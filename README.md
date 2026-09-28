# GraphRAG Studio

A hybrid GraphRAG system that combines **spaCy NER graph extraction**, **Qdrant local vector storage**, **Neo4j knowledge graph traversal**, and **OpenRouter LLM answer generation** — with full page/chunk citations.

---

## Architecture Overview

```
PDF Upload
    │
    ▼
┌──────────────────────┐
│   PDF Loader         │  pdfplumber — page-level text extraction
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│   Chunker            │  Character-based, sentence-boundary-aware
└──────────────────────┘
    │
    ├──────────────────────────────────────────────┐
    ▼                                              ▼
┌──────────────────────┐              ┌──────────────────────────┐
│  spaCy Extractor     │              │  Qdrant Embedder          │
│  (graph_extractor)   │              │  BAAI/bge-small-en-v1.5  │
│  NER + dep. parse    │              │  384-dim cosine vectors   │
│  SVO triples +       │              │  Local file persistence   │
│  co-occurrence edges │              │  (no Docker needed)       │
└──────────────────────┘              └──────────────────────────┘
    │                                              │
    ▼                                             (stored)
┌──────────────────────┐
│  Entity Resolver     │  Deduplication, canonical names, aliases
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│  Neo4j Writer        │  Document → Chunk → Entity graph
│                      │  MENTIONS + SVO/CO_OCCURS_WITH edges
└──────────────────────┘

───────── QUERY TIME ─────────────────────────────────────────────

User Question
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Hybrid Retriever                                │
│  1. Qdrant semantic search (top-K chunks)        │
│  2. spaCy NER on question → keywords             │
│  3. Neo4j Cypher → entities + linked chunks      │
│  4. Neo4j graph facts (entity-rel-entity triples)│
│  5. Merge, deduplicate, rank                     │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│  OpenRouter Answer Generator                     │
│  model: google/gemini-2.5-flash (configurable)   │
│  Grounded answer with page/chunk citations       │
│  Zero hallucination policy via system prompt     │
└──────────────────────────────────────────────────┘
```

---

## Setup

### 1. Prerequisites

- Python 3.9+
- Neo4j (local or AuraDB) running at `bolt://localhost:7687`
- An [OpenRouter](https://openrouter.ai/) API key

### 2. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Configure `.env`

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=google/gemini-2.5-flash

# Fast extraction (no LLM calls during ingestion)
EXTRACTION_METHOD=spacy

# Qdrant local storage (no Docker needed)
QDRANT_PATH=qdrant_storage
QDRANT_COLLECTION=graphrag_chunks
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

RETRIEVAL_TOP_K=5
GRAPH_HOPS=1
```

### 4. Run the Web App

```bash
python app.py
# → http://localhost:5001
```

---

## Pipeline Modules

| Module | Purpose |
|--------|---------|
| `pipeline/pdf_loader.py` | Extract text + page metadata from PDFs |
| `pipeline/chunker.py` | Sentence-boundary-aware text chunking |
| `pipeline/spacy_extractor.py` | **Fast** NER + SVO + co-occurrence graph extraction |
| `pipeline/graph_extractor.py` | Original Ollama LLM extractor (+ `get_extractor()` factory) |
| `pipeline/entity_resolver.py` | Deduplicate & canonicalize entities across chunks |
| `pipeline/neo4j_writer.py` | Persist graph to Neo4j (Documents, Chunks, Entities, Rels) |
| `pipeline/embedder.py` | Encode chunks → Qdrant local vector store |
| `pipeline/retriever.py` | Hybrid Qdrant + Neo4j retrieval with entity extraction |
| `pipeline/answer_generator.py` | OpenRouter grounded Q&A with citations |
| `pipeline/orchestrator.py` | End-to-end pipeline coordinator with timing |

---

## Benchmarking

```bash
# Compare spaCy vs LLM extraction on a PDF
python benchmark.py --pdf uploads/myfile.pdf --method both

# With retrieval + answer benchmark questions
python benchmark.py --pdf uploads/myfile.pdf --questions questions.txt --method spacy

# spaCy only (no Ollama required)
python benchmark.py --pdf uploads/myfile.pdf --method spacy
```

Output: `benchmark_results_<timestamp>.md` with:
- Side-by-side timing metrics
- Entity/relationship counts and distributions
- Per-question retrieval latency and answer quality
- Manual qualitative rating template

---

## Extraction Methods

| Method | Speed | LLM Calls | Quality |
|--------|-------|-----------|---------|
| `spacy` | ⚡ Fast (~ms/chunk) | 0 | Good NER, grammatical relations |
| `llm` | 🐢 Slow (~3-30s/chunk) | 1 per chunk | Richer semantic relations |

Set via `EXTRACTION_METHOD=spacy` in `.env` or pass `extraction_method` in the upload form.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/api/health` | Check Neo4j, OpenRouter, extraction method |
| `POST` | `/api/upload` | Upload PDF → run pipeline |
| `POST` | `/api/chat` | Ask question → hybrid retrieve + OpenRouter answer |
| `GET` | `/api/graph/stats` | Node/relationship counts |
| `GET` | `/api/graph/data?limit=100` | Graph data for visualization |
| `GET` | `/api/documents` | List processed documents |
| `GET` | `/api/entities?q=query` | Search entities |

### `/api/chat` Request/Response

```json
// Request
{ "question": "Who developed Neo4j?", "document_id": "doc_abc123" }

// Response
{
  "answer": "Neo4j was developed by Neo4j Inc. [Page 3, Chunk 2 — report.pdf]",
  "citations": ["Page 3, Chunk 2 — report.pdf"],
  "model": "google/gemini-2.5-flash",
  "chunks_retrieved": 5,
  "graph_facts_retrieved": 3,
  "retrieval_latency_s": 0.12,
  "answer_latency_s": 1.45,
  "llm_calls": 1,
  "retrieved_chunks": [...],
  "graph_facts": [...]
}
```
