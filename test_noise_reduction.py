"""Before/after noise-reduction benchmark for the spaCy graph extractor.

Runs entity+relationship extraction on the GPT-4 technical report PDF in
two modes:
  - BEFORE: raw spaCy extraction (EntityFilter/Normalizer/RelationshipFilter disabled)
  - AFTER:  filtered + normalized extraction (current production behaviour)

Prints a detailed comparison including counts, examples of removed noise,
and examples of preserved meaningful entities.

NO side effects: does NOT write to Neo4j or Qdrant.

Usage:
    .venv/bin/python test_noise_reduction.py
    .venv/bin/python test_noise_reduction.py --pdf uploads/d08defaf_gpt4\ technical\ report.pdf
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from collections import Counter

# ── Bootstrap the package path ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.pdf_loader import load_pdf
from pipeline.chunker import chunk_text
from pipeline.graph_extractor import Entity, Relationship, ExtractionResult
from pipeline.spacy_extractor import (
    _get_nlp,
    _extract_entities_from_doc,
    _extract_relationships_from_doc,
)
from pipeline.entity_filter import EntityFilter, EntityNormalizer, RelationshipFilter


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_raw(chunks) -> tuple[list[Entity], list[Relationship]]:
    """Run spaCy extraction WITHOUT any filtering/normalisation."""
    nlp = _get_nlp()
    texts = [c.text for c in chunks]
    all_entities: list[Entity] = []
    all_relationships: list[Relationship] = []
    for chunk, doc in zip(chunks, nlp.pipe(texts, batch_size=32)):
        ents = _extract_entities_from_doc(doc)
        rels = _extract_relationships_from_doc(doc, ents, source_text=chunk.text)
        all_entities.extend(ents)
        all_relationships.extend(rels)
    return all_entities, all_relationships


def _extract_filtered(chunks) -> tuple[list[Entity], list[Relationship], list[Entity], list[Relationship]]:
    """Run spaCy extraction WITH filtering/normalisation.

    Returns:
        kept_entities, kept_relationships, removed_entities, removed_relationships
    """
    nlp = _get_nlp()
    filt  = EntityFilter()
    norm  = EntityNormalizer()
    rfilt = RelationshipFilter()

    texts = [c.text for c in chunks]
    kept_ents: list[Entity] = []
    kept_rels: list[Relationship] = []
    removed_ents: list[Entity] = []
    removed_rels: list[Relationship] = []

    for chunk, doc in zip(chunks, nlp.pipe(texts, batch_size=32)):
        raw_ents = _extract_entities_from_doc(doc)
        kept_e, removed_e = filt.filter(raw_ents)
        normed_ents = norm.normalize(kept_e)

        raw_rels = _extract_relationships_from_doc(doc, normed_ents, source_text=chunk.text)
        entity_name_set = {e.name for e in normed_ents}
        kept_r, removed_r = rfilt.filter(raw_rels, entity_names=entity_name_set)

        kept_ents.extend(normed_ents)
        kept_rels.extend(kept_r)
        removed_ents.extend(removed_e)
        removed_rels.extend(removed_r)

    return kept_ents, kept_rels, removed_ents, removed_rels


def _dedup_names(entities: list[Entity]) -> list[str]:
    seen = set()
    result = []
    for e in entities:
        k = e.name.lower()
        if k not in seen:
            seen.add(k)
            result.append(e.name)
    return result


def _section(title: str) -> None:
    width = 70
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def _bullet_list(items: list[str], limit: int = 15, indent: str = "    ") -> None:
    for item in items[:limit]:
        print(f"{indent}• {item}")
    if len(items) > limit:
        print(f"{indent}  … and {len(items) - limit} more")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Graph extraction noise-reduction benchmark")
    parser.add_argument(
        "--pdf",
        default="uploads/d08defaf_gpt4 technical report.pdf",
        help="Path to the PDF to test against",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Limit to N chunks (0 = all). Use a small number for a quick test.",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        # Try finding the GPT-4 PDF automatically
        candidates = list(Path("uploads").glob("*gpt4*")) + list(Path("uploads").glob("*GPT*"))
        if candidates:
            pdf_path = candidates[0]
            print(f"[auto-detected] Using: {pdf_path}")
        else:
            print(f"[ERROR] PDF not found: {pdf_path}")
            sys.exit(1)

    print(f"\n{'═' * 70}")
    print(f"  GRAPH EXTRACTION NOISE-REDUCTION BENCHMARK")
    print(f"  PDF: {pdf_path.name}")
    print(f"{'═' * 70}")

    # ── Load & chunk ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading PDF …")
    content = load_pdf(pdf_path)
    print(f"      {content.total_pages} page(s) loaded")

    print("[2/4] Chunking …")
    all_chunks = chunk_text(content, document_id="benchmark", chunk_size=1000, chunk_overlap=200)
    chunks = all_chunks[: args.max_chunks] if args.max_chunks else all_chunks
    print(f"      {len(chunks)} chunk(s) {'(limited)' if args.max_chunks else ''}")

    # ── BEFORE ────────────────────────────────────────────────────────────────
    print("[3/4] Running BEFORE extraction (raw spaCy, no filter) …")
    raw_ents, raw_rels = _extract_raw(chunks)
    before_unique_ents = _dedup_names(raw_ents)
    before_unique_rels_set = {(r.source, r.target, r.relationship) for r in raw_rels}

    # ── AFTER ─────────────────────────────────────────────────────────────────
    print("[4/4] Running AFTER extraction (filtered + normalised) …")
    kept_ents, kept_rels, removed_ents, removed_rels = _extract_filtered(chunks)
    after_unique_ents = _dedup_names(kept_ents)
    after_unique_rels_set = {(r.source, r.target, r.relationship) for r in kept_rels}

    # ─────────────────────────────────────────────────────────────────────────
    # Summary counts
    # ─────────────────────────────────────────────────────────────────────────
    _section("SUMMARY COUNTS")
    n_before_ent = len(before_unique_ents)
    n_after_ent  = len(after_unique_ents)
    n_before_rel = len(before_unique_rels_set)
    n_after_rel  = len(after_unique_rels_set)
    n_rem_ent    = len({e.name.lower() for e in removed_ents})
    n_rem_rel    = len(removed_rels)

    ent_pct  = 100 * (n_before_ent - n_after_ent)  / max(n_before_ent, 1)
    rel_pct  = 100 * (n_before_rel - n_after_rel)  / max(n_before_rel, 1)

    print(f"\n  {'Metric':<35} {'BEFORE':>8}  {'AFTER':>8}  {'Removed':>9}")
    print(f"  {'─'*35} {'─'*8}  {'─'*8}  {'─'*9}")
    print(f"  {'Unique entity names':<35} {n_before_ent:>8}  {n_after_ent:>8}  "
          f"{n_before_ent - n_after_ent:>8}  ({ent_pct:.0f}% noise removed)")
    print(f"  {'Unique relationships':<35} {n_before_rel:>8}  {n_after_rel:>8}  "
          f"{n_before_rel - n_after_rel:>8}  ({rel_pct:.0f}% noise removed)")

    # ─────────────────────────────────────────────────────────────────────────
    # Entity type breakdown
    # ─────────────────────────────────────────────────────────────────────────
    _section("ENTITY TYPE BREAKDOWN  (BEFORE → AFTER)")
    before_type_counts = Counter(e.type for e in raw_ents)
    after_type_counts  = Counter(e.type for e in kept_ents)
    all_types = sorted(before_type_counts.keys() | after_type_counts.keys())
    print(f"\n  {'Type':<20} {'Before':>8}  {'After':>8}")
    print(f"  {'─'*20} {'─'*8}  {'─'*8}")
    for t in all_types:
        print(f"  {t:<20} {before_type_counts[t]:>8}  {after_type_counts[t]:>8}")

    # ─────────────────────────────────────────────────────────────────────────
    # Removed noise examples
    # ─────────────────────────────────────────────────────────────────────────
    _section("EXAMPLES OF REMOVED NOISE (entities)")
    rem_names = sorted({e.name for e in removed_ents}, key=lambda x: x.lower())
    _bullet_list(rem_names, limit=30)

    _section("EXAMPLES OF REMOVED NOISE (relationships)")
    rem_rel_strs = sorted({
        f"({r.source}) --[{r.relationship}]--> ({r.target})" for r in removed_rels
    })
    _bullet_list(rem_rel_strs, limit=20)

    # ─────────────────────────────────────────────────────────────────────────
    # Preserved meaningful entities
    # ─────────────────────────────────────────────────────────────────────────
    _section("PRESERVED MEANINGFUL ENTITIES (sample)")
    meaningful_types = {"Person", "Organization", "Product", "Location", "Group", "Language"}
    meaningful = [n for e in kept_ents if e.type in meaningful_types for n in [e.name]]
    meaningful_dedup = list(dict.fromkeys(meaningful))  # preserve order, deduplicate
    _bullet_list(meaningful_dedup, limit=30)

    # ─────────────────────────────────────────────────────────────────────────
    # Kept relationship examples
    # ─────────────────────────────────────────────────────────────────────────
    _section("SAMPLE KEPT RELATIONSHIPS")
    kept_rel_strs = sorted({
        f"({r.source}) --[{r.relationship}]--> ({r.target})" for r in kept_rels
    })
    _bullet_list(kept_rel_strs, limit=20)

    # ─────────────────────────────────────────────────────────────────────────
    # Normalisation examples
    # ─────────────────────────────────────────────────────────────────────────
    _section("NORMALISATION EXAMPLES  (raw → canonical)")
    from pipeline.entity_filter import EntityNormalizer
    norm = EntityNormalizer()
    raw_names_all = {e.name for e in raw_ents}
    collapsed: list[tuple[str, str]] = []
    for raw_name in sorted(raw_names_all):
        canonical = norm.normalize_name(raw_name)
        if canonical != raw_name:
            collapsed.append((raw_name, canonical))
    if collapsed:
        for raw_name, canonical in collapsed[:20]:
            print(f"    {raw_name!r:30s} → {canonical!r}")
    else:
        print("    (no variant normalisations detected in this sample)")

    print(f"\n{'═' * 70}")
    print("  BENCHMARK COMPLETE")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
