"""Parent-Child hierarchical chunking with deterministic IDs."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.ingestion.parser import Document


def sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """ChromaDB only accepts str/int/float/bool metadata values: coerce or drop everything else."""
    clean: Dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            if isinstance(v, float) and v != v:  # NaN
                continue
            clean[str(k)] = v
        elif isinstance(v, (list, tuple, set)):
            clean[str(k)] = ", ".join(str(x) for x in v)
        else:
            clean[str(k)] = json.dumps(v, ensure_ascii=False, default=str)[:500]
    return clean


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")[:40] or "doc"


class HierarchicalChunker:
    """Splits documents into large parent chunks (LLM context) and small child chunks (search units).

    IDs are content-derived, so re-ingesting the same corpus is idempotent (no duplicates).
    """

    def __init__(
        self,
        parent_chunk_size: int = settings.PARENT_CHUNK_SIZE,
        parent_chunk_overlap: int = settings.PARENT_CHUNK_OVERLAP,
        child_chunk_size: int = settings.CHILD_CHUNK_SIZE,
        child_chunk_overlap: int = settings.CHILD_CHUNK_OVERLAP,
    ):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def process_documents(self, documents: List[Document]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        child_chunks: List[Dict[str, Any]] = []
        parent_store: Dict[str, Dict[str, Any]] = {}

        for doc in documents:
            source = str(doc.metadata.get("source", "doc"))
            record = str(doc.metadata.get("record_id", ""))
            base_meta = sanitize_metadata(doc.metadata)

            for p_idx, p_text in enumerate(self.parent_splitter.split_text(doc.page_content)):
                digest = hashlib.sha1(f"{source}|{record}|{p_idx}|{p_text}".encode("utf-8")).hexdigest()[:10]
                parent_id = f"{_slug(source)}_p{p_idx}_{digest}"
                parent_store[parent_id] = {
                    "text": p_text,
                    "metadata": {**base_meta, "parent_id": parent_id, "parent_index": p_idx},
                }
                for c_idx, c_text in enumerate(self.child_splitter.split_text(p_text)):
                    if not c_text.strip():
                        continue
                    child_id = f"{parent_id}_c{c_idx}"
                    child_chunks.append(
                        {
                            "id": child_id,
                            "text": c_text,
                            "parent_id": parent_id,
                            "metadata": {**base_meta, "parent_id": parent_id, "child_id": child_id, "child_index": c_idx},
                        }
                    )
        return child_chunks, parent_store

    @staticmethod
    def save_parent_store(parent_store: Dict[str, Dict[str, Any]], path: Path = None):
        path = Path(path or settings.PARENT_STORE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(parent_store, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        tmp.replace(path)  # atomic: readers never see a half-written file

    @staticmethod
    def load_parent_store(path: Path = None) -> Dict[str, Dict[str, Any]]:
        path = Path(path or settings.PARENT_STORE_PATH)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
