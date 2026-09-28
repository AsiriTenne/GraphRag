"""Text chunking module.

Splits extracted text into manageable chunks while preserving metadata.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


from .pdf_loader import PDFContent


@dataclass
class Chunk:
    """A single text chunk with metadata."""
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    text: str
    chunk_index: int = 0


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks by character count.

    Tries to split at sentence boundaries when possible.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # Try to break at a sentence boundary
            for sep in [". ", ".\n", "! ", "? ", "\n\n", "\n", " "]:
                last_sep = text.rfind(sep, start + chunk_size // 2, end)
                if last_sep != -1:
                    end = last_sep + len(sep)
                    break

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)

        start = end - chunk_overlap

    return chunks


def chunk_text(
    content: PDFContent,
    document_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """Split PDF content into chunks.

    Args:
        content: The extracted PDF content.
        document_id: Unique identifier for this document.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of Chunk objects with metadata.
    """
    chunks: list[Chunk] = []
    global_index = 0

    for page in content.pages:
        if not page.text.strip():
            continue

        page_chunks = _split_text(page.text, chunk_size, chunk_overlap)

        for chunk_text_piece in page_chunks:
            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                filename=content.filename,
                page_number=page.page_number,
                text=chunk_text_piece,
                chunk_index=global_index,
            )
            chunks.append(chunk)
            global_index += 1

    return chunks
