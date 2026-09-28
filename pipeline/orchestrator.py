"""Pipeline Orchestrator module.

Coordinates the end-to-end flow:
  1. PDF Loading
  2. Chunking
  3. Graph Extraction (configurable: spaCy or Ollama LLM)
  4. Entity Resolution
  5. Neo4j Graph Persistence
  6. Qdrant Vector Embedding

The extraction method is selected via the EXTRACTION_METHOD environment
variable ("spacy" or "llm").  Both paths produce identical data structures
so the rest of the pipeline is unchanged.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Callable, Optional


from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, EXTRACTION_METHOD
from .pdf_loader import load_pdf
from .chunker import chunk_text
from .graph_extractor import get_extractor
from .entity_resolver import resolve_entities
from .neo4j_writer import write_to_neo4j, check_neo4j_connection
from .embedder import embed_chunks
from .neo4j_writer import get_document_stats


class PipelineOrchestrator:
    """Orchestrates the execution of the PDF to Knowledge Graph pipeline."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        extraction_method: str = EXTRACTION_METHOD,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extraction_method = extraction_method

    def process_pdf(
        self,
        file_path: str | Path,
        document_id: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> dict:
        """Process a PDF file end-to-end into a Knowledge Graph + Qdrant index.

        Args:
            file_path: Path to the PDF file.
            document_id: Optional custom document ID (generated if not provided).
            progress_callback: Optional function ``callback(status_message, percentage)``.

        Returns:
            Dict summarising processing metrics and created graph elements,
            including timing data for benchmarking.

        Raises:
            FileNotFoundError: If PDF doesn't exist.
            ValueError: If PDF text extraction fails.
            ConnectionError: If Neo4j is offline or unreachable.
        """
        timings: dict[str, float] = {}

        def report(msg: str, percent: int):
            if progress_callback:
                progress_callback(msg, percent)
            print(f"[{percent:3d}%] {msg}")

        def timed_report(stage: str, msg: str, percent: int, t_start: float):
            elapsed = time.perf_counter() - t_start
            timings[stage] = elapsed
            report(f"{msg} ({elapsed:.2f}s)", percent)

        file_path = Path(file_path)
        if not document_id:
            document_id = f"doc_{uuid.uuid4().hex[:10]}"

        pipeline_start = time.perf_counter()

        # ── Step 1: Load PDF ─────────────────────────────────────────────────
        report(f"Loading PDF: {file_path.name}", 5)
        t = time.perf_counter()
        content = load_pdf(file_path)
        timed_report("load_pdf", f"Loaded {content.total_pages} page(s)", 10, t)

        # ── Step 2: Chunk Text ───────────────────────────────────────────────
        report(f"Chunking text (size={self.chunk_size}, overlap={self.chunk_overlap})...", 15)
        t = time.perf_counter()
        chunks = chunk_text(
            content=content,
            document_id=document_id,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        timed_report("chunking", f"Generated {len(chunks)} chunks", 25, t)

        # ── Step 3: Vector Embeddings (Qdrant) ───────────────────────────────
        report(f"Embedding {len(chunks)} chunks into Qdrant...", 30)
        t = time.perf_counter()
        vectors_stored = embed_chunks(chunks, extraction_method=self.extraction_method)
        timed_report("embedding", f"Stored {vectors_stored} vectors in Qdrant", 45, t)

        # ── Step 4: Graph Extraction ─────────────────────────────────────────
        report(
            f"Extracting entities & relationships using '{self.extraction_method}' "
            f"across {len(chunks)} chunk(s)...",
            50,
        )
        t = time.perf_counter()
        extractor = get_extractor(self.extraction_method)
        extraction_results = extractor(chunks=chunks, document_id=document_id)
        timed_report(
            "extraction",
            f"Extracted entities/relationships from {len(chunks)} chunks",
            65,
            t,
        )

        # ── Step 5: Entity Resolution ────────────────────────────────────────
        report("Resolving & deduplicating entities and relationships...", 70)
        t = time.perf_counter()
        resolved_entities, resolved_relationships, chunk_mentions = resolve_entities(
            extraction_results
        )
        timed_report(
            "entity_resolution",
            f"Resolved {len(resolved_entities)} unique entities, "
            f"{len(resolved_relationships)} relationships, "
            f"{len(chunk_mentions)} chunk mentions",
            80,
            t,
        )

        # ── Step 6: Neo4j Connectivity Check ────────────────────────────────
        report("Verifying Neo4j connectivity...", 83)
        is_connected, conn_msg = check_neo4j_connection()
        if not is_connected:
            raise ConnectionError(
                f"Neo4j is offline or unreachable: {conn_msg}. Graph creation aborted."
            )

        # ── Step 7: Neo4j Persistence ────────────────────────────────────────
        report("Persisting document, chunks, entities, and relationships in Neo4j...", 87)
        t = time.perf_counter()
        neo4j_stats = write_to_neo4j(
            content=content,
            chunks=chunks,
            resolved_entities=resolved_entities,
            resolved_relationships=resolved_relationships,
            document_id=document_id,
            chunk_mentions=chunk_mentions,
            extraction_method=self.extraction_method,
        )
        timed_report("neo4j_write", "Neo4j graph persisted", 98, t)

        # ── Step 8: Verify Neo4j write counts ───────────────────────────────
        neo4j_verified = get_document_stats(document_id)

        timings["total"] = time.perf_counter() - pipeline_start
        report(
            f"Pipeline complete! Total time: {timings['total']:.2f}s "
            f"(method: {self.extraction_method})",
            100,
        )

        return {
            "status": "success",
            "document_id": document_id,
            "filename": content.filename,
            "total_pages": content.total_pages,
            "chunk_count": len(chunks),
            "vectors_stored": vectors_stored,
            # In-memory extraction counts (before Neo4j MERGE deduplication)
            "entity_count": len(resolved_entities),
            "relationship_count": len(resolved_relationships),
            "mention_count": len(chunk_mentions),
            # Verified Neo4j write counts (after MERGE deduplication)
            "neo4j_chunks": neo4j_verified.get("chunks_in_neo4j", 0),
            "neo4j_entities": neo4j_verified.get("entities_reachable", 0),
            "neo4j_mentions": neo4j_verified.get("mentions_in_neo4j", 0),
            "neo4j_relationships": neo4j_verified.get("relationships_in_neo4j", 0),
            "extraction_method": self.extraction_method,
            "timings_seconds": timings,
            "neo4j_stats": neo4j_stats,
            "entities": [
                {
                    "name": e.canonical_name,
                    "type": e.entity_type,
                    "description": e.description,
                    "aliases": e.aliases,
                }
                for e in resolved_entities
            ],
            "relationships": resolved_relationships,
        }
