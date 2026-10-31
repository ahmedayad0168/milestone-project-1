"""Parse JSON / CSV / PDF / TXT / MD files into Document objects."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.logging_setup import get_logger

log = get_logger(__name__)
SUPPORTED = {".json", ".csv", ".pdf", ".txt", ".md"}


@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any]


def _clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        import math

        if isinstance(value, float) and math.isnan(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


class DocumentParser:
    @staticmethod
    def parse_file(file_path: Path) -> List[Document]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        ext = file_path.suffix.lower()

        if ext == ".json":
            return DocumentParser._parse_json(file_path)
        if ext == ".csv":
            return DocumentParser._parse_csv(file_path)
        if ext == ".pdf":
            return DocumentParser._parse_pdf(file_path)
        if ext in {".txt", ".md"}:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            return [Document(text, {"source": file_path.name, "file_type": ext.lstrip(".")})] if text.strip() else []
        raise ValueError(f"Unsupported file format: {ext}")

    @staticmethod
    def _record_to_doc(item: Dict[str, Any], idx: int, file_path: Path, file_type: str) -> Optional[Document]:
        content = _clean(item.get("content") or item.get("text") or item.get("response"))
        title = _clean(item.get("title") or item.get("topic") or item.get("instruction")) or f"Record {idx + 1}"
        if not content:
            return None
        full_text = content if content == title else f"Title: {title}\n\n{content}"
        extra = {k: v for k, v in item.items() if k not in {"content", "text", "response", "source"}}
        meta = {
            **extra,
            "record_id": _clean(item.get("id")) or f"{file_path.stem}_{idx}",
            "title": title,
            "file_type": file_type,
            "source": file_path.name,
            "origin_source": _clean(item.get("source")) or file_path.name,
        }
        return Document(full_text, meta)

    @staticmethod
    def _parse_json(file_path: Path) -> List[Document]:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("records") or data.get("documents") or [data]
        docs = []
        for idx, item in enumerate(data if isinstance(data, list) else []):
            if isinstance(item, str):
                item = {"content": item}
            if not isinstance(item, dict):
                continue
            doc = DocumentParser._record_to_doc(item, idx, file_path, "json")
            if doc:
                docs.append(doc)
        return docs

    @staticmethod
    def _parse_csv(file_path: Path) -> List[Document]:
        import pandas as pd

        df = pd.read_csv(file_path)
        docs = []
        for idx, row in enumerate(df.to_dict(orient="records")):
            if not any(k in row for k in ("content", "text", "response")):
                row = {**row, "content": " | ".join(f"{k}: {_clean(v)}" for k, v in row.items() if _clean(v))}
            doc = DocumentParser._record_to_doc(row, idx, file_path, "csv")
            if doc:
                docs.append(doc)
        return docs

    @staticmethod
    def _parse_pdf(file_path: Path) -> List[Document]:
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        docs = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(text, {"source": file_path.name, "page": i + 1, "file_type": "pdf"}))
        return docs

    @classmethod
    def parse_directory(cls, directory_path: Path, exclude_globs: Iterable[str] = ()) -> List[Document]:
        directory_path = Path(directory_path)
        if not directory_path.exists():
            return []
        excluded = set()
        for pattern in exclude_globs:
            excluded.update(directory_path.rglob(pattern))
        docs: List[Document] = []
        for file in sorted(directory_path.rglob("*")):
            if not file.is_file() or file.suffix.lower() not in SUPPORTED or file in excluded:
                continue
            try:
                docs.extend(cls.parse_file(file))
            except Exception as e:  # one bad file must not abort the whole ingestion
                log.warning("Skipping %s: %s", file.name, e)
        return docs
