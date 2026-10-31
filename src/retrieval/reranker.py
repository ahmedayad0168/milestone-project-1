"""Cross-encoder reranking."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.config import settings
from src.logging_setup import get_logger

log = get_logger(__name__)


class Reranker:
    """Scores (query, passage) pairs with a cross-encoder. Falls back to RRF order if disabled/unavailable."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = settings.RERANKER_MODEL_NAME if model_name is None else model_name
        self._model = None
        self._failed = self.model_name.lower() in {"", "none", "off"}

    @property
    def model(self):
        if self._model is None and not self._failed:
            try:
                from sentence_transformers import CrossEncoder

                log.info("Loading cross-encoder %s", self.model_name)
                self._model = CrossEncoder(self.model_name, max_length=512, device="cpu")
            except Exception as e:
                log.warning("Reranker unavailable (%s); falling back to RRF ordering.", e)
                self._failed = True
        return self._model

    @property
    def is_cross_encoder(self) -> bool:
        return self.model is not None

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = settings.RERANK_TOP_K) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        ranked = [dict(c) for c in candidates]
        model = self.model
        if model is not None:
            scores = model.predict([[query, c["text"]] for c in ranked], batch_size=16, show_progress_bar=False)
            for c, s in zip(ranked, scores):
                c["rerank_score"] = float(s)
        else:
            for c in ranked:
                c["rerank_score"] = float(c.get("rrf_score", c.get("score", 0.0)))
        ranked.sort(key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k]
