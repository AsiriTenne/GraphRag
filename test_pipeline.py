"""Incremental test script for RAG GraphRAG pipeline components."""

import sys
from pathlib import Path

from pipeline.pdf_loader import PageContent, PDFContent
from pipeline import (
    chunk_text,
    resolve_entities,
    check_neo4j_connection,
    write_to_neo4j,
    PipelineOrchestrator,
)

from pipeline.graph_extractor import Entity, Relationship, ExtractionResult
from config.settings import OLLAMA_MODEL, OLLAMA_HOST


def test_imports_and_settings():
    print("=== Test 1: Settings & Config ===")
    print(f"Ollama Configured Host: {OLLAMA_HOST}")
    print(f"Ollama Configured Model: {OLLAMA_MODEL}")
    assert OLLAMA_MODEL != "", "OLLAMA_MODEL should be configured"
    print("  PASSED: Config loaded successfully.\n")



def test_pdf_struct_and_chunker():
    print("=== Test 2: PDF Loader & Chunker Structures ===")
    mock_pdf = PDFContent(
        filename="test_sample.pdf",
        pages=[
            PageContent(
                page_number=1,
                text="GraphRAG combines Knowledge Graphs with Retrieval Augmented Generation. Neo4j is a popular graph database created by Neo4j Inc."
            ),
            PageContent(
                page_number=2,
                text="OpenAI produces language models like GPT-4o. Antigravity is an AI coding assistant designed by Google DeepMind."
            )
        ]
    )

    chunks = chunk_text(mock_pdf, document_id="doc_test_123", chunk_size=200, chunk_overlap=30)
    print(f"  Total chunks generated: {len(chunks)}")
    for c in chunks:
        print(f"    - Chunk {c.chunk_index} (Page {c.page_number}): {c.text[:50]}...")
    assert len(chunks) >= 2, "Expected at least 2 chunks"
    print("  PASSED: Chunker test successful.\n")
    return mock_pdf, chunks


def test_entity_resolution(chunks):
    print("=== Test 3: Entity Resolution & Chunk Mentions ===")
    mock_extractions = [
        ExtractionResult(
            chunk_id=chunks[0].chunk_id,
            document_id="doc_test_123",
            entities=[
                Entity(name="GraphRAG", type="Technology", description="Knowledge graph RAG framework"),
                Entity(name="Neo4j", type="Database", description="Graph database system"),
                Entity(name="Neo4j Inc", type="Organization", description="Company behind Neo4j"),
            ],
            relationships=[
                Relationship(source="GraphRAG", target="Neo4j", relationship="USES", description="Uses graph database"),
                Relationship(source="Neo4j Inc", target="Neo4j", relationship="CREATED", description="Created database"),
            ]
        ),
        ExtractionResult(
            chunk_id=chunks[1].chunk_id,
            document_id="doc_test_123",
            entities=[
                Entity(name="OpenAI", type="Organization", description="AI Research Company"),
                Entity(name="GPT-4o", type="Technology", description="LLM model"),
                Entity(name="Antigravity", type="Technology", description="AI agent"),
                Entity(name="Neo4j", type="Database", description="Graph DB"),
            ],
            relationships=[
                Relationship(source="OpenAI", target="GPT-4o", relationship="PRODUCED", description="Produces LLM"),
            ]
        )
    ]

    resolved_entities, resolved_relationships, chunk_mentions = resolve_entities(mock_extractions)

    print(f"  Resolved entities ({len(resolved_entities)}): {[e.canonical_name for e in resolved_entities]}")
    print(f"  Resolved relationships ({len(resolved_relationships)}): {[(r['source'], r['relationship'], r['target']) for r in resolved_relationships]}")
    print(f"  Chunk mentions tracked ({len(chunk_mentions)}): {chunk_mentions}")

    assert len(resolved_entities) > 0, "Resolved entities should not be empty"
    assert len(chunk_mentions) >= 6, "Chunk mentions should link all entities in chunks"
    print("  PASSED: Entity resolution & provenance test successful.\n")
    return resolved_entities, resolved_relationships, chunk_mentions


def test_neo4j_offline_enforcement(mock_pdf, chunks, resolved_entities, resolved_relationships, chunk_mentions):
    print("=== Test 4: Neo4j Connection Check & Offline Enforcement ===")
    is_connected, msg = check_neo4j_connection()
    print(f"  Neo4j Connection Status: Connected={is_connected} ({msg})")

    if not is_connected:
        print("  Neo4j is offline. Verifying write_to_neo4j raises ConnectionError as required...")
        try:
            write_to_neo4j(
                content=mock_pdf,
                chunks=chunks,
                resolved_entities=resolved_entities,
                resolved_relationships=resolved_relationships,
                document_id="doc_test_123",
                chunk_mentions=chunk_mentions,
            )
            print("  FAILED: write_to_neo4j should have raised ConnectionError when Neo4j is offline!")
            sys.exit(1)
        except ConnectionError as e:
            print(f"  PASSED: ConnectionError correctly raised when Neo4j is offline: {e}\n")
    else:
        print("  Neo4j is online! Writing test graph...")
        stats = write_to_neo4j(
            content=mock_pdf,
            chunks=chunks,
            resolved_entities=resolved_entities,
            resolved_relationships=resolved_relationships,
            document_id="doc_test_123",
            chunk_mentions=chunk_mentions,
        )
        print(f"  PASSED: Written stats to Neo4j: {stats}\n")


def run_all_tests():
    print("Starting incremental pipeline tests...\n")
    test_imports_and_settings()
    pdf, chunks = test_pdf_struct_and_chunker()
    res_ent, res_rel, mentions = test_entity_resolution(chunks)
    test_neo4j_offline_enforcement(pdf, chunks, res_ent, res_rel, mentions)
    print("All incremental pipeline tests finished successfully!")


if __name__ == "__main__":
    run_all_tests()
