from pathlib import Path

import fitz
import pytest

from src.ingestion.chunker import chunk_pages
from src.ingestion.pdf_loader import extract_pdf_pages


def create_test_pdf(path: Path) -> None:
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "PM-KISAN official information page.")
        document.save(path)


def test_extract_pdf_pages_preserves_page_metadata(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scheme.pdf"
    create_test_pdf(pdf_path)

    pages = extract_pdf_pages(pdf_path)

    assert len(pages) == 1
    assert pages[0].source_name == "scheme.pdf"
    assert pages[0].page_number == 1
    assert "PM-KISAN" in pages[0].text


def test_chunk_pages_preserves_source_page(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scheme.pdf"
    create_test_pdf(pdf_path)
    page = extract_pdf_pages(pdf_path)[0]

    chunks = chunk_pages([page], max_characters=10)

    assert len(chunks) > 1
    assert all(chunk.source_name == "scheme.pdf" for chunk in chunks)
    assert all(chunk.page_number == 1 for chunk in chunks)


def test_loader_rejects_missing_or_non_pdf_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_pdf_pages(tmp_path / "missing.pdf")

    text_file = tmp_path / "notes.txt"
    text_file.write_text("not a PDF", encoding="utf-8")
    with pytest.raises(ValueError):
        extract_pdf_pages(text_file)
