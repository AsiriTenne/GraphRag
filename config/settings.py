"""Configuration settings loaded from environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Neo4j ────────────────────────────────────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# ── Ollama (kept for LLM extraction mode) ────────────────────────────────────
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# ── OpenRouter (final answer generation) ─────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── Qdrant (local file persistence — no Docker required) ─────────────────────
QDRANT_PATH = os.getenv("QDRANT_PATH", str(Path(__file__).parent.parent / "qdrant_storage"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "graphrag_chunks")

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = 384  # fixed for bge-small-en-v1.5

# ── Extraction ────────────────────────────────────────────────────────────────
# Values: "spacy" (fast, no LLM) | "llm" (Ollama, slow but richer)
EXTRACTION_METHOD = os.getenv("EXTRACTION_METHOD", "spacy")

# ── Chunking ──────────────────────────────────────────────────────────────────
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))   # Qdrant top-K chunks
GRAPH_HOPS = int(os.getenv("GRAPH_HOPS", "1"))              # Neo4j neighbour hops
