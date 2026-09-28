#!/usr/bin/env python3
"""GraphRAG Extraction Benchmark — Research Comparison Script.

Runs BOTH the spaCy extractor and the Ollama LLM extractor on the same PDF
and produces a side-by-side Markdown comparison report.

Usage:
    python benchmark.py --pdf "uploads/myfile.pdf"
    python benchmark.py --pdf "uploads/myfile.pdf" --method spacy
    python benchmark.py --pdf "uploads/myfile.pdf" --method llm
    python benchmark.py --pdf "uploads/myfile.pdf" --questions questions.txt

Output:
    benchmark_results_<timestamp>.md   — structured comparison report
    (also printed to stdout)

Requirements:
    - Neo4j must be running (for graph write + retrieval steps)
    - Ollama must be running only if method includes "llm"
    - OPENROUTER_API_KEY must be set only if --questions is provided
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

# ── Ensure project root is on path ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    OLLAMA_HOST, OLLAMA_MODEL,
    OPENROUTER_MODEL,
    EXTRACTION_METHOD,
)
from pipeline.pdf_loader import load_pdf
from pipeline.chunker import chunk_text
from pipeline.graph_extractor import get_extractor, check_ollama_connection
from pipeline.entity_resolver import resolve_entities
from pipeline.neo4j_writer import check_neo4j_connection, write_to_neo4j, get_driver
from pipeline.embedder import embed_chunks, get_collection_count


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner for one extraction method
# ─────────────────────────────────────────────────────────────────────────────

def run_single_method(
    pdf_path: Path,
    method: str,
    questions: list[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> dict:
    """Run the full ingestion pipeline for one extraction method.

    Returns a result dict with all timing and quality metrics.
    """
    result: dict = {
        "method": method,
        "pdf": pdf_path.name,
        "status": "pending",
        "errors": [],
    }

    doc_id = f"bench_{method}_{uuid.uuid4().hex[:6]}"
    result["document_id"] = doc_id

    print(f"\n{'='*60}")
    print(f" METHOD: {method.upper()}  |  doc_id: {doc_id}")
    print(f"{'='*60}")

    # ── Step 1: PDF Loading ──────────────────────────────────────────────────
    print("[1/7] Loading PDF...")
    t = time.perf_counter()
    try:
        content = load_pdf(pdf_path)
        result["load_time_s"] = round(time.perf_counter() - t, 3)
        result["total_pages"] = content.total_pages
        result["total_chars"] = sum(len(p.text) for p in content.pages)
        print(f"      Pages: {content.total_pages}  |  Chars: {result['total_chars']}")
    except Exception as e:
        result["status"] = "error"
        result["errors"].append(f"PDF load failed: {e}")
        return result

    # ── Step 2: Chunking ─────────────────────────────────────────────────────
    print("[2/7] Chunking text...")
    t = time.perf_counter()
    chunks = chunk_text(content, document_id=doc_id,
                        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    result["chunking_time_s"] = round(time.perf_counter() - t, 3)
    result["chunk_count"] = len(chunks)
    print(f"      Chunks: {len(chunks)}")

    # ── Step 3: Embeddings ───────────────────────────────────────────────────
    print("[3/7] Embedding chunks into Qdrant...")
    t = time.perf_counter()
    try:
        vectors = embed_chunks(chunks, extraction_method=method)
        result["embedding_time_s"] = round(time.perf_counter() - t, 3)
        result["vectors_stored"] = vectors
        result["qdrant_total_vectors"] = get_collection_count()
        print(f"      Vectors stored: {vectors}")
    except Exception as e:
        result["embedding_time_s"] = 0
        result["vectors_stored"] = 0
        result["errors"].append(f"Embedding failed: {e}")
        print(f"      WARNING: Embedding failed — {e}")

    # ── Step 4: Graph Extraction ─────────────────────────────────────────────
    print(f"[4/7] Extracting entities & relationships ({method})...")
    extractor = get_extractor(method)
    chunk_times: list[float] = []
    llm_calls = 0

    if method == "llm":
        # For LLM, time each chunk individually (to get per-chunk stats)
        extraction_results = []
        for chunk in chunks:
            t_chunk = time.perf_counter()
            try:
                res = extractor(chunks=[chunk], document_id=doc_id)
                extraction_results.extend(res)
                llm_calls += 1
            except Exception as e:
                result["errors"].append(f"LLM extraction error chunk {chunk.chunk_index}: {e}")
                print(f"      ERROR on chunk {chunk.chunk_index}: {e}")
            chunk_times.append(time.perf_counter() - t_chunk)
    else:
        # spaCy — time the entire batch
        t_all = time.perf_counter()
        try:
            extraction_results = extractor(chunks=chunks, document_id=doc_id)
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(f"spaCy extraction failed: {e}")
            return result
        total_ext = time.perf_counter() - t_all
        chunk_times = [total_ext / len(chunks)] * len(chunks) if chunks else [0]

    result["extraction_time_s"] = round(sum(chunk_times), 3)
    result["avg_chunk_extraction_s"] = round(statistics.mean(chunk_times), 4) if chunk_times else 0
    result["std_chunk_extraction_s"] = round(statistics.stdev(chunk_times), 4) if len(chunk_times) > 1 else 0
    result["llm_calls_ingestion"] = llm_calls

    raw_entities = sum(len(r.entities) for r in extraction_results)
    raw_relationships = sum(len(r.relationships) for r in extraction_results)
    print(f"      Raw entities: {raw_entities}  |  Raw relationships: {raw_relationships}")

    # ── Step 5: Entity Resolution ────────────────────────────────────────────
    print("[5/7] Resolving entities...")
    t = time.perf_counter()
    resolved_entities, resolved_relationships, chunk_mentions = resolve_entities(extraction_results)
    result["entity_resolution_time_s"] = round(time.perf_counter() - t, 3)
    result["entity_count"] = len(resolved_entities)
    result["relationship_count"] = len(resolved_relationships)
    result["mention_count"] = len(chunk_mentions)
    result["avg_entities_per_chunk"] = round(len(resolved_entities) / max(len(chunks), 1), 2)
    result["avg_relationships_per_chunk"] = round(len(resolved_relationships) / max(len(chunks), 1), 2)

    # Entity type distribution
    type_counts: dict[str, int] = {}
    for e in resolved_entities:
        type_counts[e.entity_type] = type_counts.get(e.entity_type, 0) + 1
    result["entity_type_distribution"] = dict(sorted(type_counts.items(), key=lambda x: -x[1]))

    # Relationship type distribution
    rel_counts: dict[str, int] = {}
    for r in resolved_relationships:
        rel_counts[r["relationship"]] = rel_counts.get(r["relationship"], 0) + 1
    result["relationship_type_distribution"] = dict(sorted(rel_counts.items(), key=lambda x: -x[1])[:15])

    # Sample entities and relationships for qualitative review
    result["sample_entities"] = [
        {"name": e.canonical_name, "type": e.entity_type, "description": e.description[:80]}
        for e in resolved_entities[:15]
    ]
    result["sample_relationships"] = [
        {"src": r["source"], "rel": r["relationship"], "tgt": r["target"]}
        for r in resolved_relationships[:15]
    ]

    print(f"      Resolved: {len(resolved_entities)} entities, {len(resolved_relationships)} relationships")
    print(f"      Entity types: {list(type_counts.keys())}")

    # ── Step 6: Neo4j Write ──────────────────────────────────────────────────
    print("[6/7] Writing to Neo4j...")
    is_connected, msg = check_neo4j_connection()
    if not is_connected:
        result["errors"].append(f"Neo4j offline — skipping write: {msg}")
        result["neo4j_write_time_s"] = 0
        print(f"      WARNING: {msg}")
    else:
        t = time.perf_counter()
        try:
            neo4j_stats = write_to_neo4j(
                content=content,
                chunks=chunks,
                resolved_entities=resolved_entities,
                resolved_relationships=resolved_relationships,
                document_id=doc_id,
                chunk_mentions=chunk_mentions,
                extraction_method=method,
            )
            result["neo4j_write_time_s"] = round(time.perf_counter() - t, 3)
            result["neo4j_stats"] = neo4j_stats
        except Exception as e:
            result["neo4j_write_time_s"] = 0
            result["errors"].append(f"Neo4j write failed: {e}")
            print(f"      ERROR: {e}")

    # ── Step 7: Retrieval + Answer benchmarks ────────────────────────────────
    if questions and is_connected:
        print(f"[7/7] Running retrieval benchmark ({len(questions)} questions)...")
        _run_retrieval_benchmark(result, questions, doc_id)
    else:
        print("[7/7] Skipping retrieval benchmark (no questions or Neo4j offline)")

    # ── Totals ───────────────────────────────────────────────────────────────
    timed_keys = [k for k in result if k.endswith("_time_s")]
    result["total_ingestion_time_s"] = round(sum(result.get(k, 0) for k in timed_keys
                                                  if "retrieval" not in k and "answer" not in k), 3)
    result["status"] = "success" if not result["errors"] else "partial"

    print(f"\n✅  Done ({method}) — total ingestion: {result['total_ingestion_time_s']:.2f}s")
    return result


def _run_retrieval_benchmark(result: dict, questions: list[str], doc_id: str):
    """Add retrieval and answer metrics to result dict (mutates in-place)."""
    from pipeline.retriever import hybrid_retrieve

    retrieval_latencies: list[float] = []
    answer_latencies: list[float] = []
    chunks_retrieved_per_q: list[int] = []
    graph_facts_per_q: list[int] = []
    llm_answer_calls = 0
    qa_pairs: list[dict] = []

    try:
        from pipeline.answer_generator import generate_answer
        from config.settings import OPENROUTER_API_KEY
        can_answer = bool(OPENROUTER_API_KEY)
    except Exception:
        can_answer = False

    for q in questions:
        qa = {"question": q}

        # Retrieval
        t = time.perf_counter()
        try:
            chunks, graph_facts = hybrid_retrieve(q, document_id=doc_id)
            lat = round(time.perf_counter() - t, 3)
            retrieval_latencies.append(lat)
            chunks_retrieved_per_q.append(len(chunks))
            graph_facts_per_q.append(len(graph_facts))
            qa["retrieval_latency_s"] = lat
            qa["chunks_retrieved"] = len(chunks)
            qa["graph_facts"] = len(graph_facts)
        except Exception as e:
            qa["retrieval_error"] = str(e)

        # Answer generation
        if can_answer and "retrieval_error" not in qa:
            t = time.perf_counter()
            try:
                ar = generate_answer(q, chunks, graph_facts)
                lat = round(time.perf_counter() - t, 3)
                answer_latencies.append(lat)
                llm_answer_calls += 1
                qa["answer_latency_s"] = lat
                qa["answer"] = ar.answer[:500]   # truncate for report
                qa["citations"] = ar.citations
                qa["prompt_tokens"] = ar.prompt_tokens
                qa["completion_tokens"] = ar.completion_tokens
            except Exception as e:
                qa["answer_error"] = str(e)

        qa_pairs.append(qa)

    result["qa_pairs"] = qa_pairs
    result["llm_calls_answers"] = llm_answer_calls
    result["avg_retrieval_latency_s"] = round(statistics.mean(retrieval_latencies), 3) if retrieval_latencies else 0
    result["avg_answer_latency_s"] = round(statistics.mean(answer_latencies), 3) if answer_latencies else 0
    result["avg_chunks_retrieved"] = round(statistics.mean(chunks_retrieved_per_q), 1) if chunks_retrieved_per_q else 0
    result["avg_graph_facts"] = round(statistics.mean(graph_facts_per_q), 1) if graph_facts_per_q else 0


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(results: list[dict], pdf_path: Path, questions: list[str]) -> str:
    """Generate a structured Markdown comparison report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []

    lines.append("# GraphRAG Extraction Benchmark Report")
    lines.append(f"\n**Generated:** {timestamp}")
    lines.append(f"**PDF:** `{pdf_path.name}`")
    lines.append(f"**Methods compared:** {', '.join(r['method'] for r in results)}")
    if questions:
        lines.append(f"**Benchmark questions:** {len(questions)}")

    # ── Side-by-side metrics table ────────────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## Quantitative Metrics")
    lines.append("\n> These are automatically measured. Fill in the Qualitative Ratings section manually.\n")

    metrics = [
        ("Pages", "total_pages"),
        ("Characters (total)", "total_chars"),
        ("Chunks", "chunk_count"),
        ("Vectors stored (Qdrant)", "vectors_stored"),
        ("**Ingestion Timing**", None),
        ("PDF load time (s)", "load_time_s"),
        ("Chunking time (s)", "chunking_time_s"),
        ("Embedding time (s)", "embedding_time_s"),
        ("Extraction time (s)", "extraction_time_s"),
        ("Avg per-chunk extraction (s)", "avg_chunk_extraction_s"),
        ("StdDev per-chunk extraction (s)", "std_chunk_extraction_s"),
        ("Entity resolution time (s)", "entity_resolution_time_s"),
        ("Neo4j write time (s)", "neo4j_write_time_s"),
        ("**Total ingestion time (s)**", "total_ingestion_time_s"),
        ("LLM calls (ingestion)", "llm_calls_ingestion"),
        ("**Entity & Relationship Counts**", None),
        ("Unique entities", "entity_count"),
        ("Unique relationships", "relationship_count"),
        ("Chunk–entity mentions", "mention_count"),
        ("Avg entities / chunk", "avg_entities_per_chunk"),
        ("Avg relationships / chunk", "avg_relationships_per_chunk"),
    ]

    if questions:
        metrics += [
            ("**Retrieval Performance**", None),
            ("Avg retrieval latency (s)", "avg_retrieval_latency_s"),
            ("Avg chunks retrieved / question", "avg_chunks_retrieved"),
            ("Avg graph facts / question", "avg_graph_facts"),
            ("Avg answer latency (s)", "avg_answer_latency_s"),
            ("LLM calls (answers)", "llm_calls_answers"),
        ]

    # Build header
    header_cols = ["Metric"] + [f"**{r['method'].upper()}**" for r in results]
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_cols)) + " |")

    for label, key in metrics:
        if key is None:
            # Section header row
            lines.append(f"| {label} | " + " | ".join([""] * len(results)) + " |")
            continue
        vals = []
        for r in results:
            v = r.get(key, "—")
            vals.append(str(v) if v != "—" else "—")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    # ── Entity type distributions ─────────────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## Entity Type Distribution\n")
    for r in results:
        dist = r.get("entity_type_distribution", {})
        lines.append(f"### {r['method'].upper()}")
        if dist:
            lines.append("| Type | Count |")
            lines.append("| --- | --- |")
            for etype, count in dist.items():
                lines.append(f"| {etype} | {count} |")
        else:
            lines.append("_No entities extracted._")
        lines.append("")

    # ── Relationship type distributions ───────────────────────────────────────
    lines.append("---\n")
    lines.append("## Relationship Type Distribution (Top 15)\n")
    for r in results:
        dist = r.get("relationship_type_distribution", {})
        lines.append(f"### {r['method'].upper()}")
        if dist:
            lines.append("| Relationship | Count |")
            lines.append("| --- | --- |")
            for rtype, count in dist.items():
                lines.append(f"| {rtype} | {count} |")
        else:
            lines.append("_No relationships extracted._")
        lines.append("")

    # ── Sample entities ────────────────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## Sample Entities (First 15)\n")
    for r in results:
        lines.append(f"### {r['method'].upper()}")
        samples = r.get("sample_entities", [])
        if samples:
            lines.append("| Name | Type | Description |")
            lines.append("| --- | --- | --- |")
            for e in samples:
                lines.append(f"| {e['name']} | {e['type']} | {e['description']} |")
        else:
            lines.append("_No entities._")
        lines.append("")

    # ── Sample relationships ───────────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## Sample Relationships (First 15)\n")
    for r in results:
        lines.append(f"### {r['method'].upper()}")
        samples = r.get("sample_relationships", [])
        if samples:
            lines.append("| Source | Relationship | Target |")
            lines.append("| --- | --- | --- |")
            for rel in samples:
                lines.append(f"| {rel['src']} | {rel['rel']} | {rel['tgt']} |")
        else:
            lines.append("_No relationships._")
        lines.append("")

    # ── Per-question Q&A comparison ────────────────────────────────────────────
    if questions:
        lines.append("---\n")
        lines.append("## Per-Question Retrieval & Answer Comparison\n")
        for qi, q in enumerate(questions):
            lines.append(f"### Q{qi+1}: {q}\n")
            for r in results:
                qa_pairs = r.get("qa_pairs", [])
                qa = qa_pairs[qi] if qi < len(qa_pairs) else {}
                lines.append(f"**{r['method'].upper()}**")
                lines.append(f"- Retrieval: {qa.get('retrieval_latency_s', '—')}s | "
                              f"Chunks: {qa.get('chunks_retrieved', '—')} | "
                              f"Graph facts: {qa.get('graph_facts', '—')}")
                if "answer" in qa:
                    lines.append(f"- Answer latency: {qa.get('answer_latency_s', '—')}s")
                    lines.append(f"- Answer:\n\n  > {qa['answer'][:400].replace(chr(10), chr(10) + '  > ')}")
                elif "answer_error" in qa:
                    lines.append(f"- Answer error: {qa['answer_error']}")
                lines.append("")

    # ── Qualitative ratings (manual fill-in) ──────────────────────────────────
    lines.append("---\n")
    lines.append("## Qualitative Ratings (Manual Assessment)\n")
    lines.append("> Fill in these ratings (1=poor, 5=excellent) after reviewing the samples above.\n")

    q_metrics = [
        "Entity meaningfulness (are entities real/useful domain concepts?)",
        "Entity completeness (are important entities missing?)",
        "Relationship accuracy (do relationships reflect actual document statements?)",
        "Relationship meaningfulness (are CO_OCCURS_WITH / generic rels useful?)",
        "Retrieval relevance (do retrieved chunks answer the question?)",
        "Answer groundedness (does the answer stay within the evidence?)",
        "Answer completeness (does the answer fully address the question?)",
        "Answer citation accuracy (are citations correct?)",
    ]

    header_cols = ["Dimension"] + [r["method"].upper() for r in results] + ["Notes"]
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_cols)) + " |")
    for qm in q_metrics:
        vals = ["_/5_"] * len(results) + [""]
        lines.append(f"| {qm} | " + " | ".join(vals) + " |")

    # ── Errors ────────────────────────────────────────────────────────────────
    any_errors = any(r.get("errors") for r in results)
    if any_errors:
        lines.append("\n---\n")
        lines.append("## Errors / Warnings\n")
        for r in results:
            errs = r.get("errors", [])
            if errs:
                lines.append(f"### {r['method'].upper()}")
                for err in errs:
                    lines.append(f"- {err}")
                lines.append("")

    # ── Raw JSON ──────────────────────────────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## Raw Results (JSON)\n")
    lines.append("<details><summary>Click to expand</summary>\n")
    lines.append("```json")
    # Strip qa pairs from raw JSON to keep it manageable
    raw_copy = [{k: v for k, v in r.items() if k != "qa_pairs"} for r in results]
    lines.append(json.dumps(raw_copy, indent=2, default=str))
    lines.append("```\n</details>")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark spaCy vs LLM extraction on the same PDF."
    )
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument(
        "--method",
        choices=["spacy", "llm", "both"],
        default="both",
        help="Which extractor(s) to run (default: both)",
    )
    parser.add_argument(
        "--questions",
        help="Path to a text file with one question per line (for retrieval benchmark)",
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--output", help="Output report path (default: auto-named .md)")

    args = parser.parse_args()
    pdf_path = Path(args.pdf)

    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        sys.exit(1)

    # Load questions
    questions: list[str] = []
    if args.questions:
        q_path = Path(args.questions)
        if q_path.exists():
            questions = [l.strip() for l in q_path.read_text().splitlines() if l.strip()]
            print(f"Loaded {len(questions)} benchmark questions from {q_path}")
        else:
            print(f"WARNING: Questions file not found: {q_path}")

    # Determine which methods to run
    methods: list[str] = []
    if args.method == "both":
        methods = ["spacy", "llm"]
    else:
        methods = [args.method]

    # Check LLM availability if needed
    if "llm" in methods:
        llm_ok, llm_msg = check_ollama_connection()
        if not llm_ok:
            print(f"\nWARNING: Ollama is not reachable: {llm_msg}")
            print("LLM benchmarking will be skipped.")
            methods = [m for m in methods if m != "llm"]
        else:
            print(f"Ollama: {llm_msg}")

    if not methods:
        print("No methods available to benchmark. Exiting.")
        sys.exit(1)

    print(f"\nBenchmarking method(s): {methods}")
    print(f"PDF: {pdf_path}")
    print(f"Chunk size: {args.chunk_size}, overlap: {args.chunk_overlap}")

    # Run each method
    results: list[dict] = []
    for method in methods:
        res = run_single_method(
            pdf_path=pdf_path,
            method=method,
            questions=questions,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        results.append(res)

    # Generate report
    report = generate_report(results, pdf_path, questions)

    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else Path(f"benchmark_results_{timestamp}.md")
    out_path.write_text(report, encoding="utf-8")

    print(f"\n{'='*60}")
    print(f" BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f" Report saved: {out_path.resolve()}")
    for r in results:
        print(f"\n [{r['method'].upper()}]")
        print(f"   Status          : {r['status']}")
        print(f"   Ingestion time  : {r.get('total_ingestion_time_s', '?')}s")
        print(f"   Entities        : {r.get('entity_count', '?')}")
        print(f"   Relationships   : {r.get('relationship_count', '?')}")
        print(f"   LLM calls (ingest): {r.get('llm_calls_ingestion', 0)}")
        if questions:
            print(f"   Avg retrieval   : {r.get('avg_retrieval_latency_s', '?')}s")
            print(f"   Avg answer      : {r.get('avg_answer_latency_s', '?')}s")
        if r.get("errors"):
            print(f"   Errors          : {r['errors']}")


if __name__ == "__main__":
    main()
