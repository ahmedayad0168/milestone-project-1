"""First-run index bootstrap: makes a fresh deployment (e.g. Streamlit Cloud) self-sufficient."""
from __future__ import annotations

import threading

from src.config import settings
from src.ingestion.indexer import Indexer
from src.ingestion.run_ingestion import run_ingestion_pipeline
from src.logging_setup import get_logger

log = get_logger(__name__)
_lock = threading.Lock()


def index_is_ready(indexer: Indexer = None) -> bool:
    indexer = indexer or Indexer()
    return indexer.count() > 0 and settings.PARENT_STORE_PATH.exists() and indexer.bm25_path.exists()


def ensure_index() -> dict:
    """Build the indexes if they are missing. Safe to call from many threads/sessions."""
    with _lock:
        indexer = Indexer()
        if index_is_ready(indexer):
            return {"built": False, "chunks": indexer.count()}
        log.info("Index not found - building from data/raw (or bundled seed corpus)...")
        stats = run_ingestion_pipeline(indexer=indexer)
        return {"built": True, "chunks": stats["children"], **stats}
