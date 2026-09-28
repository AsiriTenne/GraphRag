"""OpenRouter answer generation module.

Uses the OpenRouter API (OpenAI-compatible) with google/gemini-2.5-flash
to generate grounded answers from retrieved evidence.

Rules enforced via the system prompt:
  - Ground every claim in the provided context
  - Cite sources as [Page X, Chunk Y, doc: filename]
  - Explicitly say "I don't have enough information" when evidence is thin
  - Never invent facts not present in the context
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from openai import OpenAI

from config.settings import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    OPENROUTER_BASE_URL,
)
from .retriever import RetrievedChunk

SYSTEM_PROMPT = """\
You are a precise research assistant. You answer questions ONLY using the provided evidence.

RULES:
1. Base every sentence on the provided Evidence chunks and/or Graph Facts.
2. Cite your sources inline like this: [Page 3, Chunk 2 — filename.pdf]
3. If a piece of information comes from a Graph Fact, cite it as: [Graph: EntityA → Relationship → EntityB]
4. If the provided evidence does not contain enough information to answer, say:
   "The available documents do not contain sufficient information to answer this question."
5. Do NOT invent facts, figures, dates, or names that are not in the evidence.
6. Keep your answer concise and well-structured.
"""


@dataclass
class AnswerResult:
    """Result of a single answer generation call."""
    answer: str
    citations: list[str]
    model: str
    latency_seconds: float
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int


def _build_context(chunks: list[RetrievedChunk], graph_facts: list[str]) -> str:
    """Format retrieved evidence into a compact context string for the LLM."""
    parts: list[str] = []

    if chunks:
        parts.append("=== EVIDENCE CHUNKS ===")
        for i, chunk in enumerate(chunks, 1):
            citation = (
                f"[Page {chunk.page_number}, Chunk {chunk.chunk_index} — {chunk.filename}]"
            )
            parts.append(f"\n[{i}] {citation}\n{chunk.text.strip()}")

    if graph_facts:
        parts.append("\n=== GRAPH FACTS ===")
        for fact in graph_facts:
            parts.append(f"• {fact}")

    return "\n".join(parts)


def _extract_citations(chunks: list[RetrievedChunk], graph_facts: list[str]) -> list[str]:
    """Build a deduplicated citation list from the retrieved evidence."""
    citations: list[str] = []
    seen: set[str] = set()

    for chunk in chunks:
        ref = f"Page {chunk.page_number}, Chunk {chunk.chunk_index} — {chunk.filename}"
        if ref not in seen:
            seen.add(ref)
            citations.append(ref)

    for fact in graph_facts:
        ref = f"Graph: {fact}"
        if ref not in seen:
            seen.add(ref)
            citations.append(ref)

    return citations


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    graph_facts: list[str],
) -> AnswerResult:
    """Generate a grounded answer from retrieved evidence using OpenRouter.

    Args:
        question: The user's natural language question.
        chunks: Retrieved text chunks (from hybrid retriever).
        graph_facts: Entity-relationship-entity strings from Neo4j.

    Returns:
        AnswerResult with answer text, citations, timing, and token usage.

    Raises:
        ValueError: If OPENROUTER_API_KEY is not configured.
        RuntimeError: If the API call fails.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
            "Add it to your .env file: OPENROUTER_API_KEY=your_key_here"
        )

    if not chunks and not graph_facts:
        return AnswerResult(
            answer=(
                "No relevant evidence was found in the documents for this question. "
                "Please upload a relevant PDF and try again."
            ),
            citations=[],
            model=OPENROUTER_MODEL,
            latency_seconds=0.0,
            llm_calls=0,
            prompt_tokens=0,
            completion_tokens=0,
        )

    context = _build_context(chunks, graph_facts)
    citations = _extract_citations(chunks, graph_facts)

    user_message = (
        f"Question: {question}\n\n"
        f"{context}\n\n"
        "Answer the question using only the evidence above. "
        "Cite sources inline."
    )

    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

    t_start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,   # low temperature for factual, grounded answers
            max_tokens=1024,
        )
    except Exception as exc:
        raise RuntimeError(f"OpenRouter API call failed: {exc}") from exc

    latency = time.perf_counter() - t_start

    answer_text = response.choices[0].message.content or ""
    usage = response.usage

    return AnswerResult(
        answer=answer_text,
        citations=citations,
        model=OPENROUTER_MODEL,
        latency_seconds=round(latency, 3),
        llm_calls=1,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
    )
