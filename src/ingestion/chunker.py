"""Create bounded RAG chunks while retaining page-level citations."""

from dataclasses import dataclass

from src.ingestion.pdf_loader import PageText


@dataclass(frozen=True)
class TextChunk:
    """A searchable text chunk and its source page."""

    chunk_id: str
    source_path: str
    source_name: str
    page_number: int
    text: str
    source_type: str = "pdf"
    source_url: str | None = None
    scheme_name: str | None = None
    structured_data: dict[str, str] | None = None


def chunk_pages(pages: list[PageText], max_characters: int = 1200) -> list[TextChunk]:
    """Split page text into deterministic word-boundary chunks.

    A page remains the citation unit even when it produces multiple chunks.
    """

    if max_characters < 1:
        raise ValueError("max_characters must be positive")

    chunks: list[TextChunk] = []
    for page in pages:
        words = page.text.split()
        current_words: list[str] = []
        current_length = 0
        chunk_number = 0

        for word in words:
            proposed_length = current_length + len(word) + bool(current_words)
            if current_words and proposed_length > max_characters:
                chunks.append(
                    TextChunk(
                        chunk_id=f"{page.source_name}:{page.page_number}:{chunk_number}",
                        source_path=page.source_path,
                        source_name=page.source_name,
                        page_number=page.page_number,
                        text=" ".join(current_words),
                    )
                )
                chunk_number += 1
                current_words = []
                current_length = 0

            current_words.append(word)
            current_length += len(word) + (1 if len(current_words) > 1 else 0)

        if current_words:
            chunks.append(
                TextChunk(
                    chunk_id=f"{page.source_name}:{page.page_number}:{chunk_number}",
                    source_path=page.source_path,
                    source_name=page.source_name,
                    page_number=page.page_number,
                    text=" ".join(current_words),
                )
            )
    return chunks
