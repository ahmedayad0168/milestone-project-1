"""Download the embedding + reranker models into the HF cache (used at Docker build time)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402

if __name__ == "__main__":
    from sentence_transformers import CrossEncoder, SentenceTransformer

    SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    if settings.RERANKER_MODEL_NAME.lower() not in {"", "none", "off"}:
        CrossEncoder(settings.RERANKER_MODEL_NAME)
    print("Models cached.")
