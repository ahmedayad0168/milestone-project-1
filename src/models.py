"""Shared model singletons (embedder), so every component reuses one copy in RAM."""
from __future__ import annotations

import hashlib
import threading
from typing import List, Optional

import numpy as np

from src.config import settings
from src.logging_setup import get_logger

log = get_logger(__name__)
_lock = threading.Lock()
_embedder: Optional["BaseEmbedder"] = None


class BaseEmbedder:
    dim: int = 0
    name: str = "base"

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:  # -> (n, dim) L2-normalised
        raise NotImplementedError


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model %s", model_name)
        self.name = model_name
        self._model = SentenceTransformer(model_name, device="cpu")
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        return np.asarray(
            self._model.encode(
                list(texts), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
            ),
            dtype=np.float32,
        )


class HashEmbedder(BaseEmbedder):
    """Deterministic bag-of-words hashing embedder. Offline; used for CI and smoke tests only."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hash-{dim}"

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        from src.text_utils import tokenize

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 64) % 2 == 0 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


def get_embedder() -> BaseEmbedder:
    global _embedder
    if _embedder is None:
        with _lock:
            if _embedder is None:
                if settings.EMBEDDING_BACKEND == "hash":
                    _embedder = HashEmbedder()
                else:
                    _embedder = SentenceTransformerEmbedder(settings.EMBEDDING_MODEL_NAME)
    return _embedder


def set_embedder(embedder: Optional[BaseEmbedder]) -> None:
    """Inject an embedder (tests)."""
    global _embedder
    _embedder = embedder
