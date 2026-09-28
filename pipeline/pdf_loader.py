"""PDF text extraction module.

Extracts text from uploaded PDFs while preserving page information.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass


import pdfplumber


@dataclass
class PageContent:
    """A single page's extracted text and metadata."""
    page_number: int
    text: str


@dataclass
class PDFContent:
    """Full extracted PDF content."""
    filename: str
    pages: list[PageContent]

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    @property
    def has_content(self) -> bool:
        return any(p.text.strip() for p in self.pages)


def load_pdf(file_path: str | Path) -> PDFContent:
    """Extract text from a PDF file.

    Args:
        file_path: Path to the PDF file.

    Returns:
        PDFContent with extracted text per page.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the PDF contains no extractable text.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    pages: list[PageContent] = []

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append(PageContent(page_number=i, text=text))

    content = PDFContent(filename=file_path.name, pages=pages)

    if not content.has_content:
        raise ValueError(
            f"The PDF '{file_path.name}' contains no extractable text. "
            "It may be a scanned/image-based PDF."
        )

    return content
