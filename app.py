"""PDF to Knowledge Graph — Web Application Entry-Point.

Provides Flask web interface and REST APIs for uploading PDFs,
running the GraphRAG pipeline, checking status, and exploring the knowledge graph.

New in v2:
  - /api/chat  — hybrid retrieval + OpenRouter answer generation
  - /api/health checks OpenRouter connectivity alongside Neo4j / Ollama
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

from config.settings import (
    UPLOAD_DIR,
    OLLAMA_HOST, OLLAMA_MODEL,
    OPENROUTER_MODEL, OPENROUTER_API_KEY,
    EXTRACTION_METHOD,
)
from pipeline import (
    PipelineOrchestrator,
    check_neo4j_connection,
    check_ollama_connection,
    get_graph_stats,
    get_graph_data,
    search_entities,
    get_documents,
    hybrid_retrieve,
    generate_answer,
)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB max upload limit


@app.route("/")
def index():
    """Render main application dashboard."""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    """Return connectivity status for Neo4j, Ollama, and OpenRouter."""
    neo4j_online, neo4j_msg = check_neo4j_connection()
    ollama_online, ollama_msg = check_ollama_connection()

    openrouter_status = "configured" if OPENROUTER_API_KEY else "no_api_key"

    return jsonify({
        "neo4j": {
            "status": "online" if neo4j_online else "offline",
            "message": neo4j_msg,
        },
        "ollama": {
            "status": "online" if ollama_online else "offline",
            "host": OLLAMA_HOST,
            "model": OLLAMA_MODEL,
            "message": ollama_msg,
        },
        "openrouter": {
            "status": openrouter_status,
            "model": OPENROUTER_MODEL,
        },
        "extraction_method": EXTRACTION_METHOD,
    })


@app.route("/api/upload", methods=["POST"])
def upload_pdf():
    """Upload a PDF file and trigger the GraphRAG pipeline."""
    if "file" not in request.files:
        return jsonify({"error": "No file field provided in request"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    # Allow caller to override extraction method per-request
    method = request.form.get("extraction_method", EXTRACTION_METHOD)

    filename = Path(file.filename).name
    save_path = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{filename}"
    file.save(save_path)

    neo4j_online, neo4j_msg = check_neo4j_connection()
    if not neo4j_online:
        return jsonify({
            "error": f"Neo4j database is offline. Cannot process graph: {neo4j_msg}"
        }), 503

    try:
        orchestrator = PipelineOrchestrator(extraction_method=method)
        result = orchestrator.process_pdf(
            file_path=save_path,
            progress_callback=None,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    """Hybrid retrieval + OpenRouter answer generation.

    Request body (JSON):
        {
            "question": "What products does GreenLeaf sell?",
            "document_id": "doc_abc123"   # optional — search all docs if omitted
        }

    Response:
        {
            "answer": "...",
            "citations": ["Page 3, Chunk 2 — file.pdf", ...],
            "model": "google/gemini-2.5-flash",
            "chunks_retrieved": 5,
            "graph_facts_retrieved": 3,
            "retrieval_latency_s": 0.12,
            "answer_latency_s": 1.45,
            "llm_calls": 1
        }
    """
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    document_id = data.get("document_id") or None

    if not question:
        return jsonify({"error": "No question provided"}), 400

    if not OPENROUTER_API_KEY:
        return jsonify({
            "error": "OPENROUTER_API_KEY is not configured. "
                     "Add it to your .env file to enable the chatbot."
        }), 503

    import time

    try:
        # ── Retrieval ──────────────────────────────────────────────────────
        t_ret = time.perf_counter()
        chunks, graph_facts = hybrid_retrieve(question, document_id=document_id)
        retrieval_latency = round(time.perf_counter() - t_ret, 3)

        # ── Answer generation ──────────────────────────────────────────────
        result = generate_answer(question, chunks, graph_facts)

        return jsonify({
            "answer": result.answer,
            "citations": result.citations,
            "model": result.model,
            "chunks_retrieved": len(chunks),
            "graph_facts_retrieved": len(graph_facts),
            "retrieval_latency_s": retrieval_latency,
            "answer_latency_s": result.latency_seconds,
            "llm_calls": result.llm_calls,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            # Include chunk details for the UI citation panel
            "retrieved_chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "filename": c.filename,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                    "text": c.text[:300],   # truncate for UI
                    "score": round(c.score, 4),
                    "source": c.source,
                }
                for c in chunks
            ],
            "graph_facts": graph_facts,
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/graph/stats", methods=["GET"])
def graph_stats():
    """Return database statistics."""
    stats = get_graph_stats()
    return jsonify(stats)


@app.route("/api/graph/data", methods=["GET"])
def graph_data():
    """Return graph nodes and edges for visualization.

    Optional query params:
      - limit (int): max edges to return (default 150)
      - document_id (str): restrict graph to a single document
    """
    limit = request.args.get("limit", default=150, type=int)
    document_id = request.args.get("document_id", default=None, type=str) or None
    data = get_graph_data(limit=limit, document_id=document_id)
    return jsonify(data)


@app.route("/api/documents", methods=["GET"])
def list_documents():
    """Return list of processed documents."""
    docs = get_documents()
    return jsonify(docs)


@app.route("/api/entities", methods=["GET"])
def search_entities_api():
    """Search entities in the Knowledge Graph."""
    query = request.args.get("q", default="", type=str)
    results = search_entities(query)
    return jsonify(results)


if __name__ == "__main__":
    print("Starting RAG GraphRAG Web Server on http://localhost:5001")
    print(f"Extraction method: {EXTRACTION_METHOD}")
    print(f"OpenRouter model: {OPENROUTER_MODEL}")
    print(f"OpenRouter API key: {'set' if OPENROUTER_API_KEY else 'NOT SET'}")
    app.run(host="0.0.0.0", port=5001, debug=True)
