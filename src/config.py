"""Central configuration.

Values are read (in order) from: real environment variables, Streamlit secrets
(when running on Streamlit Cloud), then the defaults below. A local ``.env`` file
is loaded for development but never overrides variables already set.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=False)


def _secret(name: str):
    """Return a Streamlit secret if running under Streamlit and it exists."""
    try:
        import streamlit as st  # local import: keeps API/CLI free of Streamlit

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return None


def _get(name: str, default: str = "") -> str:
    val = os.getenv(name)
    if val is None:
        val = _secret(name)
    return default if val is None or val == "" else val


def _bool(name: str, default: bool = False) -> bool:
    return _get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(_get(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_get(name, str(default)))
    except ValueError:
        return default


@dataclass
class Settings:
    # ── Paths ────────────────────────────────────────────────────────────
    BASE_DIR: Path = BASE_DIR
    DATA_RAW_DIR: Path = field(default_factory=lambda: BASE_DIR / "data" / "raw")
    DATA_SEED_DIR: Path = field(default_factory=lambda: BASE_DIR / "data" / "seed")
    DATA_PROCESSED_DIR: Path = field(default_factory=lambda: BASE_DIR / "data" / "processed")
    CHROMA_PERSIST_DIR: Path = field(
        default_factory=lambda: Path(_get("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db")))
    )
    PARENT_STORE_PATH: Path = field(
        default_factory=lambda: Path(_get("PARENT_STORE_PATH", str(BASE_DIR / "data" / "parent_docstore.json")))
    )
    BM25_INDEX_PATH: Path = field(
        default_factory=lambda: Path(_get("BM25_INDEX_PATH", str(BASE_DIR / "data" / "bm25_index.pkl")))
    )
    GOLDEN_EVAL_PATH: Path = field(
        default_factory=lambda: Path(_get("GOLDEN_EVAL_PATH", str(BASE_DIR / "data" / "golden_eval_curated.json")))
    )
    # Files matching these globs inside data/raw are NOT ingested (e.g. the 75k raw dump).
    INGEST_EXCLUDE_GLOBS: tuple = ("*_raw_75k*.csv",)

    # ── Chunking (hierarchical) ──────────────────────────────────────────
    PARENT_CHUNK_SIZE: int = 1200
    PARENT_CHUNK_OVERLAP: int = 150
    CHILD_CHUNK_SIZE: int = 300
    CHILD_CHUNK_OVERLAP: int = 50

    # ── Embedding & reranking models ─────────────────────────────────────
    # EMBEDDING_BACKEND: "sentence_transformers" (default) | "hash" (offline, for CI/tests)
    EMBEDDING_BACKEND: str = field(default_factory=lambda: _get("EMBEDDING_BACKEND", "sentence_transformers"))
    EMBEDDING_MODEL_NAME: str = field(
        default_factory=lambda: _get(
            "EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    )
    # Light default (~90 MB) so the app fits in Streamlit Cloud's ~2.7 GB RAM.
    # For higher accuracy / multilingual use: BAAI/bge-reranker-base (~1.1 GB). "none" disables reranking.
    RERANKER_MODEL_NAME: str = field(
        default_factory=lambda: _get("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )

    # ── LLM ──────────────────────────────────────────────────────────────
    # LLM_BACKEND: "gemini" (default) | "groq" (optional) | "extractive" (no LLM, offline)
    LLM_BACKEND: str = field(default_factory=lambda: _get("LLM_BACKEND", "gemini").lower())
    GEMINI_API_KEY: str = field(default_factory=lambda: _get("GEMINI_API_KEY", _get("GOOGLE_API_KEY", "")))
    GEMINI_MODEL: str = field(default_factory=lambda: _get("GEMINI_MODEL", "gemini-3.5-flash"))
    # Tried in order if the primary model errors out (quota, overload, retired model id).
    GEMINI_FALLBACK_MODELS: str = field(default_factory=lambda: _get("GEMINI_FALLBACK_MODELS", "gemini-3-flash"))
    GEMINI_THINKING_LEVEL: str = field(default_factory=lambda: _get("GEMINI_THINKING_LEVEL", "low"))  # "" disables
    LLM_TEMPERATURE: float = field(default_factory=lambda: _float("LLM_TEMPERATURE", 0.1))
    LLM_MAX_TOKENS: int = field(default_factory=lambda: _int("LLM_MAX_TOKENS", 1024))
    LLM_TIMEOUT_S: int = field(default_factory=lambda: _int("LLM_TIMEOUT_S", 60))
    LLM_MAX_RETRIES: int = field(default_factory=lambda: _int("LLM_MAX_RETRIES", 2))

    # Optional alternative backend (uncomment groq in requirements.txt)
    GROQ_API_KEY: str = field(default_factory=lambda: _get("GROQ_API_KEY", ""))
    GROQ_MODEL: str = field(default_factory=lambda: _get("GROQ_MODEL", "openai/gpt-oss-120b"))

    # ── Elasticsearch (optional BM25 engine) ─────────────────────────────
    USE_ELASTICSEARCH: bool = field(default_factory=lambda: _bool("USE_ELASTICSEARCH", False))
    ELASTICSEARCH_URL: str = field(default_factory=lambda: _get("ELASTICSEARCH_URL", "http://localhost:9200"))
    ELASTICSEARCH_API_KEY: str = field(default_factory=lambda: _get("ELASTICSEARCH_API_KEY", ""))
    ELASTICSEARCH_INDEX: str = field(default_factory=lambda: _get("ELASTICSEARCH_INDEX", "fincon_child_chunks"))

    # ── Retrieval ────────────────────────────────────────────────────────
    CHROMA_COLLECTION: str = "fincon_child_chunks"
    RETRIEVAL_TOP_K_DENSE: int = 15
    RETRIEVAL_TOP_K_SPARSE: int = 15
    RETRIEVAL_CANDIDATES: int = 20      # fused candidates sent to the reranker
    RERANK_TOP_K: int = 3
    RRF_K: int = 60
    # If the best cross-encoder score is below this, the assistant abstains instead of guessing.
    # (Cross-encoder logits: clearly relevant ≈ > 0, irrelevant ≈ < -5.) Tune with the eval script.
    ABSTAIN_RERANK_THRESHOLD: float = field(default_factory=lambda: _float("ABSTAIN_RERANK_THRESHOLD", -4.0))
    MAX_CONTEXT_CHARS: int = field(default_factory=lambda: _int("MAX_CONTEXT_CHARS", 3600))

    # ── Conversation memory ──────────────────────────────────────────────
    MEMORY_TURNS: int = field(default_factory=lambda: _int("MEMORY_TURNS", 4))

    # ── Public-demo guard rails ──────────────────────────────────────────
    MAX_QUESTIONS_PER_SESSION: int = field(default_factory=lambda: _int("MAX_QUESTIONS_PER_SESSION", 40))
    MAX_QUESTION_CHARS: int = field(default_factory=lambda: _int("MAX_QUESTION_CHARS", 600))

    # ── Ops ──────────────────────────────────────────────────────────────
    LOG_LEVEL: str = field(default_factory=lambda: _get("LOG_LEVEL", "INFO"))


settings = Settings()
