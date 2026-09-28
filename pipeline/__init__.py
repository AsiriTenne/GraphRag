"""Pipeline modules for PDF to Knowledge Graph conversion."""

from .pdf_loader import load_pdf
from .chunker import chunk_text
from .graph_extractor import extract_entities_and_relationships, check_ollama_connection, get_extractor
from .spacy_extractor import extract_entities_and_relationships_spacy
from .entity_resolver import resolve_entities
from .orchestrator import PipelineOrchestrator
from .embedder import embed_chunks, semantic_search, get_collection_count
from .retriever import hybrid_retrieve, RetrievedChunk
from .answer_generator import generate_answer, AnswerResult
from .neo4j_writer import (
    write_to_neo4j,
    check_neo4j_connection,
    get_graph_stats,
    get_graph_data,
    get_document_stats,
    search_entities,
    get_documents,
    clear_database,
)

__all__ = [
    # PDF ingestion
    "load_pdf",
    "chunk_text",
    # Graph extraction — both methods
    "extract_entities_and_relationships",   # original Ollama/LLM extractor
    "extract_entities_and_relationships_spacy",  # new spaCy extractor
    "get_extractor",                        # factory: get_extractor("spacy"|"llm")
    "check_ollama_connection",
    # Entity resolution
    "resolve_entities",
    # Orchestrator
    "PipelineOrchestrator",
    # Qdrant embeddings
    "embed_chunks",
    "semantic_search",
    "get_collection_count",
    # Hybrid retrieval
    "hybrid_retrieve",
    "RetrievedChunk",
    # Answer generation
    "generate_answer",
    "AnswerResult",
    # Neo4j
    "write_to_neo4j",
    "check_neo4j_connection",
    "get_graph_stats",
    "get_graph_data",
    "get_document_stats",
    "search_entities",
    "get_documents",
    "clear_database",
]
