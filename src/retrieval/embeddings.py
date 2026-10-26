"""Local multilingual embedding model wrapper."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from config.settings import EMBEDDING_MODEL_DIR


class LocalEmbeddingModel:
    """Encode text with a locally loaded Sentence Transformers model."""

    def __init__(
        self,
        model_name: str | Path = EMBEDDING_MODEL_DIR,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        is_local_path = isinstance(model_name, Path)
        self._model = SentenceTransformer(
            str(model_name),
            local_files_only=is_local_path,
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return normalized float32 vectors suitable for cosine search."""

        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
