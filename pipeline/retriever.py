"""Hybrid retrieval module.

Combines Qdrant semantic search with Neo4j graph traversal to retrieve
the most relevant evidence for a user question.

Pipeline for a single query:
  1. Qdrant semantic search  → top-K text chunks
  2. spaCy NER on the question → entity/keyword hints
  3. Neo4j Cypher query       → entities + their neighbours + linked chunks
  4. Merge, deduplicate, rank → combined evidence list
  5. Return list of RetrievedChunk objects with provenance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from config.settings import RETRIEVAL_TOP_K, GRAPH_HOPS
from .embedder import semantic_search
from .neo4j_writer import get_driver


@dataclass
class RetrievedChunk:
    """A single piece of retrieved evidence with full provenance."""
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    score: float
    source: str                       # "qdrant_semantic" | "neo4j_graph"
    matched_entities: list[str] = field(default_factory=list)


# ── Lightweight keyword/entity extraction for queries ─────────────────────────

def _extract_query_keywords(question: str) -> list[str]:
    """
    Extract entity names and noun phrases from the question using spaCy.
    Reuses the cached model from spacy_extractor to avoid reload overhead.
    Falls back to simple word-level extraction if spaCy is unavailable.
    """
    try:
        # Reuse the already-loaded + cached model from spacy_extractor
        from .spacy_extractor import _get_nlp
        nlp = _get_nlp()
        doc = nlp(question)
        terms: list[str] = []

        # Named entities first (highest quality)
        for ent in doc.ents:
            terms.append(ent.text.strip())

        # Noun chunks as fallback
        for chunk in doc.noun_chunks:
            text = chunk.root.text.strip()
            if len(text) > 2 and text not in terms:
                terms.append(text)

        return terms if terms else _fallback_keywords(question)
    except Exception:
        return _fallback_keywords(question)


def _fallback_keywords(question: str) -> list[str]:
    """Simple word tokenizer if spaCy is unavailable."""
    stopwords = {
        "what", "who", "where", "when", "how", "why", "is", "are", "was",
        "were", "the", "a", "an", "of", "in", "on", "at", "to", "for",
        "and", "or", "but", "does", "did", "do", "this", "that", "with",
    }
    return [w for w in question.split() if len(w) > 2 and w.lower() not in stopwords]


# ── Neo4j graph retrieval ─────────────────────────────────────────────────────

def _neo4j_retrieve(
    keywords: list[str],
    document_id: Optional[str] = None,
    hops: int = 1,
) -> list[dict]:
    """
    Query Neo4j for:
      - Entities matching keywords
      - Their direct neighbours (up to `hops` hops)
      - Chunks that MENTION those entities

    Returns list of chunk payload dicts (same format as Qdrant results).
    """
    if not keywords:
        return []

    try:
        driver = get_driver()
        chunks_found: dict[str, dict] = {}  # keyed by chunk_id

        with driver.session() as session:
            for keyword in keywords[:8]:   # limit to 8 keywords to keep queries fast
                # Find entities matching keyword
                ent_result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS toLower($keyword)
                    RETURN e.name AS name
                    LIMIT 5
                    """,
                    keyword=keyword,
                )
                matched_names = [r["name"] for r in ent_result]

                if not matched_names:
                    continue

                for ent_name in matched_names:
                    # For each matched entity, find chunks that mention it
                    doc_filter = "AND c.document_id = $document_id" if document_id else ""
                    chunk_result = session.run(
                        f"""
                        MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {{name: $name}})
                        {doc_filter}
                        RETURN c.chunk_id AS chunk_id,
                               c.document_id AS document_id,
                               c.text AS text,
                               c.page_number AS page_number,
                               c.chunk_index AS chunk_index,
                               e.name AS matched_entity
                        LIMIT 10
                        """,
                        name=ent_name,
                        document_id=document_id,
                    )

                    for row in chunk_result:
                        cid = row["chunk_id"]
                        if cid not in chunks_found:
                            chunks_found[cid] = {
                                "chunk_id": cid,
                                "document_id": row["document_id"],
                                "filename": "",          # filled below if needed
                                "page_number": row["page_number"],
                                "chunk_index": row["chunk_index"],
                                "text": row["text"],
                                "score": 0.7,            # fixed score for graph hits
                                "source": "neo4j_graph",
                                "matched_entities": [],
                            }
                        chunks_found[cid]["matched_entities"].append(row["matched_entity"])

        driver.close()
        return list(chunks_found.values())

    except Exception as exc:
        # Non-fatal — return empty list so Qdrant results still work
        print(f"[retriever] Neo4j query failed (non-fatal): {exc}")
        return []


# ── Graph facts retrieval (entity + relationship snippets) ─────────────────────

def _neo4j_graph_facts(
    keywords: list[str],
    document_id: Optional[str] = None,
) -> list[str]:
    """
    Retrieve entity-relationship-entity triplet strings from Neo4j
    that can be appended to the LLM context as structured facts.

    Returns list of human-readable fact strings.
    """
    if not keywords:
        return []

    facts: list[str] = []
    try:
        driver = get_driver()
        with driver.session() as session:
            for keyword in keywords[:6]:
                doc_filter = "AND r.document_id = $document_id" if document_id else ""
                result = session.run(
                    f"""
                    MATCH (s:Entity)-[r]->(t:Entity)
                    WHERE toLower(s.name) CONTAINS toLower($keyword)
                       OR toLower(t.name) CONTAINS toLower($keyword)
                    {doc_filter}
                    RETURN s.name AS src, type(r) AS rel, t.name AS tgt,
                           r.description AS desc
                    LIMIT 8
                    """,
                    keyword=keyword,
                    document_id=document_id,
                )
                for row in result:
                    rel_str = row["rel"].replace("_", " ").title()
                    fact = f"{row['src']} — {rel_str} — {row['tgt']}"
                    if row["desc"]:
                        fact += f" ({row['desc'][:80]})"
                    if fact not in facts:
                        facts.append(fact)
        driver.close()
    except Exception:
        pass

    return facts[:20]   # cap at 20 facts


# ── Main hybrid retrieval function ────────────────────────────────────────────

def hybrid_retrieve(
    question: str,
    document_id: Optional[str] = None,
    top_k: int = RETRIEVAL_TOP_K,
) -> tuple[list[RetrievedChunk], list[str]]:
    """Retrieve the most relevant chunks and graph facts for a question.

    Args:
        question: User's natural language question.
        document_id: Optional — restrict retrieval to one document.
        top_k: Number of Qdrant semantic results to fetch.

    Returns:
        (chunks, graph_facts) where:
          - chunks is a ranked, deduplicated list of RetrievedChunk objects
          - graph_facts is a list of entity-relationship-entity fact strings
    """
    # 1. Qdrant semantic search
    qdrant_hits = semantic_search(question, top_k=top_k, document_id=document_id)

    # 2. Extract query keywords/entities for graph lookup
    keywords = _extract_query_keywords(question)

    # 3. Neo4j graph retrieval
    graph_hits = _neo4j_retrieve(keywords, document_id=document_id)

    # 4. Graph facts (entity-rel-entity strings)
    graph_facts = _neo4j_graph_facts(keywords, document_id=document_id)

    # 5. Merge and deduplicate by chunk_id
    seen_ids: set[str] = set()
    all_chunks: list[RetrievedChunk] = []

    # Qdrant results first (they have proper similarity scores)
    for hit in qdrant_hits:
        cid = hit["chunk_id"]
        if cid not in seen_ids:
            seen_ids.add(cid)
            all_chunks.append(
                RetrievedChunk(
                    chunk_id=cid,
                    document_id=hit.get("document_id", ""),
                    filename=hit.get("filename", ""),
                    page_number=hit.get("page_number", 0),
                    chunk_index=hit.get("chunk_index", 0),
                    text=hit["text"],
                    score=hit["score"],
                    source="qdrant_semantic",
                )
            )

    # Neo4j results (add only if not already in Qdrant results)
    for hit in graph_hits:
        cid = hit["chunk_id"]
        if cid not in seen_ids:
            seen_ids.add(cid)
            all_chunks.append(
                RetrievedChunk(
                    chunk_id=cid,
                    document_id=hit.get("document_id", ""),
                    filename=hit.get("filename", ""),
                    page_number=hit.get("page_number", 0),
                    chunk_index=hit.get("chunk_index", 0),
                    text=hit["text"],
                    score=hit["score"],
                    source="neo4j_graph",
                    matched_entities=hit.get("matched_entities", []),
                )
            )

    # 6. Sort by score descending
    all_chunks.sort(key=lambda c: c.score, reverse=True)

    # 7. Return top results (cap at 2*top_k to avoid bloated context)
    return all_chunks[: top_k * 2], graph_facts
