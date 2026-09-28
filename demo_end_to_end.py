"""End-to-end demonstration script: PDF → Chunks → Ollama Extraction → Entities/Relationships → Neo4j."""

import json
from pathlib import Path
from config.settings import UPLOAD_DIR, OLLAMA_MODEL, OLLAMA_HOST
from pipeline import (
    load_pdf,
    chunk_text,
    extract_entities_and_relationships,
    resolve_entities,
    check_neo4j_connection,
    check_ollama_connection,
    write_to_neo4j,
    clear_database,
)


def run_demonstration():
    pdf_path = UPLOAD_DIR / "AI_Knowledge_Graph_Research.pdf"
    print("==========================================================================")
    print("   END-TO-END DEMONSTRATION: PDF TO NEO4J KNOWLEDGE GRAPH PIPELINE")
    print("==========================================================================")
    print(f"Target PDF File: {pdf_path}")
    print(f"Configured Ollama Host: {OLLAMA_HOST}")
    print(f"Configured Ollama Model: {OLLAMA_MODEL}")

    # Check connections
    is_neo4j_online, neo4j_msg = check_neo4j_connection()
    print(f"Neo4j Status: Connected={is_neo4j_online} ({neo4j_msg})")

    is_ollama_online, ollama_msg = check_ollama_connection()
    print(f"Ollama Status: Connected={is_ollama_online} ({ollama_msg})\n")

    # Step 1: Clear Database first
    print("--- Stage 0: Clearing Database ---")
    print(clear_database())

    # Step 2: Load PDF
    print("\n--- Stage 1: PDF Text Extraction ---")
    content = load_pdf(pdf_path)
    print(f"Filename: {content.filename}")
    print(f"Total Pages: {content.total_pages}")
    for page in content.pages:
        print(f"  [Page {page.page_number}] Character count: {len(page.text)}")

    # Step 3: Chunking
    print("\n--- Stage 2: Text Chunking ---")
    doc_id = "doc_ollama_demo_001"
    chunks = chunk_text(content, document_id=doc_id, chunk_size=1000, chunk_overlap=200)
    print(f"Generated Chunks: {len(chunks)}")
    for chunk in chunks:
        print(f"  [Chunk {chunk.chunk_index}] (Page {chunk.page_number}): {chunk.text[:80]}...")

    # Step 4: Ollama LLM Extraction
    print("\n--- Stage 3: Ollama LLM Extraction (llama3.2) ---")
    print(f"Extracting structured entities & relationships using local Ollama model {OLLAMA_MODEL}...")
    extraction_results = extract_entities_and_relationships(chunks, document_id=doc_id)
    total_extracted_entities = sum(len(res.entities) for res in extraction_results)
    total_extracted_rels = sum(len(res.relationships) for res in extraction_results)
    print(f"Ollama Extraction Output: Extracted {total_extracted_entities} entities and {total_extracted_rels} relationships across chunks.")

    # Step 5: Entity Resolution
    print("\n--- Stage 4: Entity Resolution & Provenance Mapping ---")
    resolved_entities, resolved_relationships, chunk_mentions = resolve_entities(extraction_results)

    print("\nResolved Canonical Entities:")
    for e in resolved_entities:
        print(f"  • [{e.entity_type}] {e.canonical_name} — {e.description}")

    print("\nResolved Relationships:")
    for r in resolved_relationships:
        print(f"  • {r['source']}  --[{r['relationship']}]-->  {r['target']} ({r['description']})")

    print(f"\nTracked Chunk Mentions (Chunk -> MENTIONS -> Entity): {len(chunk_mentions)} total links.")

    # Step 6: Neo4j Persistence
    print("\n--- Stage 5: Neo4j Knowledge Graph Persistence ---")
    stats = write_to_neo4j(
        content=content,
        chunks=chunks,
        resolved_entities=resolved_entities,
        resolved_relationships=resolved_relationships,
        document_id=doc_id,
        chunk_mentions=chunk_mentions,
    )
    print("Successfully written to Neo4j database!")
    print(json.dumps(stats, indent=2))

    print("\n==========================================================================")
    print("   DEMONSTRATION COMPLETE: Pipeline executed successfully!")
    print("==========================================================================")


if __name__ == "__main__":
    run_demonstration()
