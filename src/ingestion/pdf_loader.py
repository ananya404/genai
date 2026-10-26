"""Extract searchable text and source metadata from local PDF files."""

from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass(frozen=True)
class PageText:
    """Text extracted from one PDF page with citation-ready metadata."""

    source_path: str
    source_name: str
    page_number: int
    text: str


def extract_pdf_pages(pdf_path: Path) -> list[PageText]:
    """Extract non-empty pages from a local PDF.

    The function intentionally accepts only a local path. Downloading official
    sources is kept outside the application so runtime inference stays local.
    """

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path.name}")

    pages: list[PageText] = []
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            text = page.get_text("text").strip()
            if text:
                pages.append(
                    PageText(
                        source_path=str(pdf_path.resolve()),
                        source_name=pdf_path.name,
                        page_number=page_index + 1,
                        text=text,
                    )
                )
    return pages
