"""FAISS vector index with citation metadata persisted as JSON."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

from src.ingestion.chunker import TextChunk


class TextEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode text into row-major vectors."""


class SearchResult:
    """A retrieved chunk and its similarity score."""

    def __init__(self, chunk: TextChunk, score: float) -> None:
        self.chunk = chunk
        self.score = score


class FaissVectorStore:
    """Build and query a cosine-similarity FAISS index."""

    _METADATA_FIELDS = frozenset(
        {
            "chunk_id",
            "source_path",
            "source_name",
            "page_number",
            "text",
            "source_type",
            "source_url",
            "scheme_name",
            "structured_data",
        }
    )

    def __init__(self, encoder: TextEncoder) -> None:
        self.encoder = encoder
        self._index: faiss.Index | None = None
        self._chunks: list[TextChunk] = []

    def build(self, chunks: Sequence[TextChunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build an index with no chunks")
        vectors = self.encoder.encode([chunk.text for chunk in chunks])
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError("Encoder returned an invalid vector matrix")
        if vectors.shape[1] == 0:
            raise ValueError("Encoder returned zero-dimensional vectors")

        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self._chunks = list(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        if self._index is None:
            raise RuntimeError("Build the vector index before searching")
        if not query.strip():
            raise ValueError("Query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1")

        query_vector = self.encoder.encode([query])
        scores, indices = self._index.search(query_vector, min(top_k, len(self._chunks)))
        return [
            SearchResult(self._chunks[index], float(score))
            for score, index in zip(scores[0], indices[0])
            if index >= 0 and float(score) >= min_score
        ]

    def save(self, directory: Path) -> None:
        if self._index is None:
            raise RuntimeError("Build the vector index before saving")
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / "index.faiss"))
        metadata = [chunk.__dict__ for chunk in self._chunks]
        (directory / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, directory: Path) -> None:
        index_path = directory / "index.faiss"
        metadata_path = directory / "metadata.json"
        if not index_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("FAISS index files are incomplete")

        index = faiss.read_index(str(index_path))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, list):
            raise ValueError("FAISS metadata must contain a list of chunk records")

        chunks: list[TextChunk] = []
        for record_number, item in enumerate(metadata):
            if not isinstance(item, dict):
                raise ValueError(
                    f"FAISS metadata record {record_number} must be an object"
                )
            unknown_fields = set(item) - self._METADATA_FIELDS
            missing_fields = self._METADATA_FIELDS - set(item)
            if unknown_fields:
                raise ValueError(
                    f"FAISS metadata record {record_number} contains unsupported "
                    f"fields: {', '.join(sorted(unknown_fields))}"
                )
            if missing_fields:
                raise ValueError(
                    f"FAISS metadata record {record_number} is missing fields: "
                    f"{', '.join(sorted(missing_fields))}"
                )
            try:
                chunks.append(TextChunk(**item))
            except TypeError as error:
                raise ValueError(
                    f"FAISS metadata record {record_number} is incompatible with "
                    "the current TextChunk schema"
                ) from error
        if index.ntotal != len(chunks):
            raise ValueError("FAISS index and metadata have different sizes")
        self._index = index
        self._chunks = chunks
