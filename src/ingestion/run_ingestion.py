"""Ingestion pipeline: parse -> hierarchical chunking -> dense + sparse indexes."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from src.config import settings
from src.ingestion.hierarchical import HierarchicalChunker
from src.ingestion.indexer import Indexer
from src.ingestion.parser import Document, DocumentParser
from src.logging_setup import get_logger

log = get_logger(__name__)


def collect_documents(raw_dir: Path = None, include_processed: bool = False, allow_seed: bool = True) -> List[Document]:
    raw_dir = Path(raw_dir or settings.DATA_RAW_DIR)
    docs = DocumentParser.parse_directory(raw_dir, exclude_globs=settings.INGEST_EXCLUDE_GLOBS)
    if include_processed:
        docs += DocumentParser.parse_directory(settings.DATA_PROCESSED_DIR, exclude_globs=("*.csv",))
    if not docs and allow_seed:
        log.warning("No documents in %s - falling back to the bundled seed corpus in %s", raw_dir, settings.DATA_SEED_DIR)
        docs = DocumentParser.parse_directory(settings.DATA_SEED_DIR)
    return docs


def run_ingestion_pipeline(
    raw_dir: Path = None,
    include_processed: bool = False,
    max_docs: Optional[int] = None,
    indexer: Optional[Indexer] = None,
) -> dict:
    docs = collect_documents(raw_dir, include_processed)
    if max_docs:
        docs = docs[:max_docs]
    if not docs:
        raise RuntimeError("No documents found to ingest (data/raw and data/seed are both empty).")
    log.info("Parsed %d documents.", len(docs))

    chunker = HierarchicalChunker()
    child_chunks, parent_store = chunker.process_documents(docs)
    log.info("Generated %d parent and %d child chunks.", len(parent_store), len(child_chunks))

    chunker.save_parent_store(parent_store)
    (indexer or Indexer()).build_indexes(child_chunks, reset=True)
    return {"documents": len(docs), "parents": len(parent_store), "children": len(child_chunks)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the FinCon RAG indexes.")
    ap.add_argument("--raw-dir", type=Path, default=None)
    ap.add_argument("--include-processed", action="store_true", help="also ingest data/processed/*.json")
    ap.add_argument("--max-docs", type=int, default=None)
    args = ap.parse_args()
    print(run_ingestion_pipeline(args.raw_dir, args.include_processed, args.max_docs))
