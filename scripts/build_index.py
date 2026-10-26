"""Build the local FAISS index from PDFs placed in data/raw/government_pdfs."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import RAW_DOCUMENTS_DIR, SCHEMES_PATH, VECTOR_STORE_DIR
from src.ingestion.chunker import chunk_pages
from src.ingestion.pdf_loader import extract_pdf_pages
from src.ingestion.scheme_loader import load_scheme_records
from src.retrieval.embeddings import LocalEmbeddingModel
from src.retrieval.vector_store import FaissVectorStore


def main() -> None:
    pdf_paths = sorted(RAW_DOCUMENTS_DIR.glob("*.pdf"))
    scheme_chunks = load_scheme_records(SCHEMES_PATH)
    pdf_chunks = []
    if pdf_paths:
        pages = [page for path in pdf_paths for page in extract_pdf_pages(path)]
        pdf_chunks = chunk_pages(pages)
    chunks = scheme_chunks + pdf_chunks
    if not chunks:
        raise SystemExit(
            f"No verified scheme records or PDF files found. Add records to "
            f"{SCHEMES_PATH} or PDFs to {RAW_DOCUMENTS_DIR}."
        )

    store = FaissVectorStore(LocalEmbeddingModel())
    store.build(chunks)
    store.save(VECTOR_STORE_DIR)
    print(
        f"Indexed {len(chunks)} chunks: {len(scheme_chunks)} curated scheme "
        f"records and {len(pdf_chunks)} PDF chunks from {len(pdf_paths)} files."
    )


if __name__ == "__main__":
    main()
