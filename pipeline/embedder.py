"""Qdrant vector embeddings module.

Stores ALL document chunks in a local Qdrant collection using
sentence-transformers (BAAI/bge-small-en-v1.5, 384-dim).

No Docker required — uses Qdrant's local file-persistence mode.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    raise ImportError(
        "sentence-transformers is not installed. "
        "Run: pip install sentence-transformers"
    ) from exc

from config.settings import (
    QDRANT_PATH,
    QDRANT_COLLECTION,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
)

# Module-level cache so the model loads only once per process
_encoder: Optional[SentenceTransformer] = None
_client: Optional[QdrantClient] = None


def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(EMBEDDING_MODEL)
    return _encoder


def _get_client() -> QdrantClient:
    """Return (and lazily create) a Qdrant client backed by local file storage."""
    global _client
    if _client is None:
        Path(QDRANT_PATH).mkdir(parents=True, exist_ok=True)
        _client = QdrantClient(path=QDRANT_PATH)
        _ensure_collection(_client)
    return _client


def _ensure_collection(client: QdrantClient) -> None:
    """Create the Qdrant collection if it doesn't already exist."""
    existing = {c.name for c in client.get_collections().collections}
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def embed_chunks(chunks: list, extraction_method: str = "spacy") -> int:
    """Encode all chunks and upsert them into Qdrant.

    Args:
        chunks: List of Chunk objects (must have chunk_id, document_id,
                filename, page_number, chunk_index, text).
        extraction_method: Which extractor was used (stored in payload for
                           benchmark queries).

    Returns:
        Number of vectors upserted.
    """
    if not chunks:
        return 0

    encoder = _get_encoder()
    client = _get_client()

    texts = [c.text for c in chunks]
    vectors = encoder.encode(texts, show_progress_bar=False, batch_size=64)

    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id)),
            vector=vector.tolist(),
            payload={
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "extraction_method": extraction_method,
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    # Upsert in batches of 256
    batch_size = 256
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points[i : i + batch_size],
        )

    return len(points)


def semantic_search(
    query: str,
    top_k: int = 5,
    document_id: Optional[str] = None,
) -> list[dict]:
    """Search Qdrant for chunks semantically similar to query.

    Args:
        query: The user's question or search text.
        top_k: Number of results to return.
        document_id: Optional — restrict search to a specific document.

    Returns:
        List of payload dicts, each with: chunk_id, document_id, filename,
        page_number, chunk_index, text, score.
    """
    encoder = _get_encoder()
    client = _get_client()

    query_vector = encoder.encode([query], show_progress_bar=False)[0].tolist()

    search_filter = None
    if document_id:
        search_filter = Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        )

    hits = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=top_k,
        query_filter=search_filter,
        with_payload=True,
    )

    results = []
    for hit in hits:
        payload = dict(hit.payload)
        payload["score"] = hit.score
        payload["source"] = "qdrant_semantic"
        results.append(payload)

    return results


def get_collection_count(document_id: Optional[str] = None) -> int:
    """Return the number of vectors in the collection (optionally per-document)."""
    client = _get_client()
    if document_id:
        return client.count(
            collection_name=QDRANT_COLLECTION,
            count_filter=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
            exact=True,
        ).count
    return client.get_collection(QDRANT_COLLECTION).vectors_count or 0
