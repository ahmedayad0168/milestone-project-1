"""FinCon & Regulatory RAG Assistant.

Importing any ``src.*`` module first applies a small compatibility shim: some
hosts (older Streamlit Cloud images, slim containers) ship a SQLite that is too
old for ChromaDB. If ``pysqlite3-binary`` is installed we transparently use it.
"""
import sys

try:  # pragma: no cover - environment dependent
    import pysqlite3  # type: ignore

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass
