import os

# Must be set before importing src.* (settings are read at import time). Fully offline test mode.
os.environ.update({"EMBEDDING_BACKEND": "hash", "LLM_BACKEND": "extractive", "RERANKER_MODEL_NAME": "none",
                   "USE_ELASTICSEARCH": "false"})

import pytest  # noqa: E402

from src.config import settings  # noqa: E402


@pytest.fixture()
def kb(tmp_path, monkeypatch):
    """A fresh, isolated knowledge base built from the bundled seed corpus."""
    empty_raw = tmp_path / "raw"
    empty_raw.mkdir()
    monkeypatch.setattr(settings, "DATA_RAW_DIR", empty_raw)
    monkeypatch.setattr(settings, "CHROMA_PERSIST_DIR", tmp_path / "chroma")
    monkeypatch.setattr(settings, "PARENT_STORE_PATH", tmp_path / "parents.json")
    monkeypatch.setattr(settings, "BM25_INDEX_PATH", tmp_path / "bm25.pkl")
    from src.bootstrap import ensure_index

    info = ensure_index()
    return {"info": info, "tmp": tmp_path}


@pytest.fixture()
def pipeline(kb):
    from src.generation.chain import FinConRAGPipeline

    return FinConRAGPipeline()
