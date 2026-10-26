from pathlib import Path
from collections.abc import Sequence
import json

import numpy as np
import pytest

from src.ingestion.chunker import TextChunk
from src.retrieval.vector_store import FaissVectorStore


class KeywordEncoder:
    """Small deterministic encoder used to test FAISS behavior without downloads."""

    vocabulary = ("insurance", "cooperative", "kisan")

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            [
                [float(text.lower().count(term)) for term in self.vocabulary]
                for text in texts
            ],
            dtype=np.float32,
        )


def sample_chunks() -> list[TextChunk]:
    return [
        TextChunk("a:1:0", "a", "insurance.pdf", 1, "crop insurance claim process"),
        TextChunk("b:2:0", "b", "cooperative.pdf", 2, "cooperative society membership"),
        TextChunk("c:3:0", "c", "kisan.pdf", 3, "kisan scheme eligibility"),
    ]


def test_search_returns_relevant_chunk() -> None:
    store = FaissVectorStore(KeywordEncoder())
    store.build(sample_chunks())

    results = store.search("insurance", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.source_name == "insurance.pdf"


def test_save_and_load_preserve_citations(tmp_path: Path) -> None:
    store = FaissVectorStore(KeywordEncoder())
    store.build(sample_chunks())
    store.save(tmp_path)

    restored = FaissVectorStore(KeywordEncoder())
    restored.load(tmp_path)

    result = restored.search("cooperative", top_k=1)[0]
    assert result.chunk.page_number == 2
    assert result.chunk.source_name == "cooperative.pdf"


def test_load_rejects_incompatible_metadata_with_clear_error(tmp_path: Path) -> None:
    store = FaissVectorStore(KeywordEncoder())
    store.build(sample_chunks())
    store.save(tmp_path)

    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[0]["old_field"] = "incompatible"
    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    restored = FaissVectorStore(KeywordEncoder())
    with pytest.raises(ValueError, match="unsupported fields: old_field"):
        restored.load(tmp_path)
