"""Dense (ChromaDB, cosine) + sparse (BM25, optional Elasticsearch) indexes."""
from __future__ import annotations

import json
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import settings
from src.logging_setup import get_logger
from src.models import BaseEmbedder, get_embedder
from src.text_utils import tokenize

log = get_logger(__name__)

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    HAVE_CHROMADB = True
except ImportError:  # pragma: no cover
    HAVE_CHROMADB = False

try:
    from rank_bm25 import BM25Okapi

    HAVE_BM25 = True
except ImportError:  # pragma: no cover
    HAVE_BM25 = False


class Indexer:
    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        bm25_path: Optional[Path] = None,
        embedder: Optional[BaseEmbedder] = None,
    ):
        self.persist_dir = Path(persist_dir or settings.CHROMA_PERSIST_DIR)
        self.bm25_path = Path(bm25_path or settings.BM25_INDEX_PATH)
        self.chunks_path = self.persist_dir / "child_chunks_store.json"
        self._embedder = embedder
        self._client = None
        self._collection = None
        self._bm25_cache = None  # (BM25Okapi, child_chunks)
        self._lock = threading.RLock()
        self.es = None
        if settings.USE_ELASTICSEARCH:
            self._init_elasticsearch()

    # ── lazy resources ────────────────────────────────────────────────────
    @property
    def embedder(self) -> BaseEmbedder:
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    @property
    def collection(self):
        if self._collection is None and HAVE_CHROMADB:
            with self._lock:
                if self._collection is None:
                    try:
                        self.persist_dir.mkdir(parents=True, exist_ok=True)
                        self._client = chromadb.PersistentClient(
                            path=str(self.persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
                        )
                        self._collection = self._client.get_or_create_collection(
                            name=settings.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
                        )
                    except Exception as e:
                        log.error("ChromaDB init failed (%s). Using lexical fallback.", e)
        return self._collection

    def count(self) -> int:
        try:
            return int(self.collection.count()) if self.collection is not None else 0
        except Exception:
            return 0

    # ── Elasticsearch (optional) ──────────────────────────────────────────
    def _init_elasticsearch(self):
        try:
            from elasticsearch import Elasticsearch

            kwargs = {"api_key": settings.ELASTICSEARCH_API_KEY} if settings.ELASTICSEARCH_API_KEY else {}
            es = Elasticsearch(hosts=[settings.ELASTICSEARCH_URL], request_timeout=5, **kwargs)
            if es.ping():
                self.es = es
                if not es.indices.exists(index=settings.ELASTICSEARCH_INDEX):
                    es.indices.create(
                        index=settings.ELASTICSEARCH_INDEX,
                        body={"mappings": {"properties": {"content": {"type": "text"}, "id": {"type": "keyword"},
                                                          "parent_id": {"type": "keyword"}, "source": {"type": "keyword"}}}},
                    )
                log.info("Connected to Elasticsearch at %s", settings.ELASTICSEARCH_URL)
            else:
                log.warning("Elasticsearch ping failed; using local BM25.")
        except Exception as e:
            log.warning("Elasticsearch unavailable (%s); using local BM25.", e)

    # ── building ──────────────────────────────────────────────────────────
    def build_indexes(self, child_chunks: List[Dict[str, Any]], max_index_chunks: Optional[int] = None, reset: bool = True):
        """(Re)build all indexes from child chunks. ``reset=True`` drops old entries so the index mirrors the corpus."""
        if not child_chunks:
            log.warning("No child chunks to index.")
            return
        chunks = child_chunks[:max_index_chunks] if max_index_chunks else child_chunks
        total = len(chunks)
        log.info("Indexing %d child chunks...", total)

        with self._lock:
            self._bm25_cache = None
            # 1. Dense
            if self.collection is not None:
                if reset and self._client is not None:
                    try:
                        self._client.delete_collection(settings.CHROMA_COLLECTION)
                    except Exception:
                        pass
                    self._collection = None
                col = self.collection
                batch = 256
                try:
                    batch = min(batch, int(self._client.get_max_batch_size()))
                except Exception:
                    pass
                for start in range(0, total, batch):
                    part = chunks[start : start + batch]
                    vectors = self.embedder.encode([c["text"] for c in part])
                    col.upsert(
                        ids=[c["id"] for c in part],
                        embeddings=vectors.tolist(),
                        documents=[c["text"] for c in part],
                        metadatas=[c["metadata"] for c in part],
                    )
                    done = min(start + batch, total)
                    if done == total or (start // batch) % 10 == 0:
                        log.info("Dense index: %d/%d", done, total)

            # 2. Chunk store (lexical fallback + BM25 source of truth)
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self.chunks_path.write_text(json.dumps(chunks, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

            # 3. Elasticsearch
            if self.es:
                try:
                    from elasticsearch.helpers import bulk

                    bulk(self.es, ({"_index": settings.ELASTICSEARCH_INDEX, "_id": c["id"],
                                    "_source": {"id": c["id"], "content": c["text"], "parent_id": c.get("parent_id"),
                                                "source": c["metadata"].get("source", ""), "metadata": c["metadata"]}}
                                   for c in chunks))
                except Exception as e:
                    log.error("Elasticsearch bulk index failed: %s", e)

            # 4. Local BM25
            if HAVE_BM25:
                bm25 = BM25Okapi([tokenize(c["text"]) or ["_"] for c in chunks])
                self.bm25_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.bm25_path, "wb") as f:
                    pickle.dump({"bm25": bm25, "child_chunks": chunks}, f)
                self._bm25_cache = (bm25, chunks)
        log.info("Indexing complete (%d chunks).", total)

    # ── searching ─────────────────────────────────────────────────────────
    def search_dense(self, query: str, top_k: int = settings.RETRIEVAL_TOP_K_DENSE, where: Optional[Dict] = None) -> List[Dict[str, Any]]:
        col = self.collection
        if col is not None and self.count() > 0:
            vec = self.embedder.encode([query])[0].tolist()
            res = col.query(query_embeddings=[vec], n_results=min(top_k, self.count()), where=where or None,
                            include=["documents", "metadatas", "distances"])
            hits = []
            for i, cid in enumerate((res.get("ids") or [[]])[0]):
                meta = res["metadatas"][0][i] or {}
                hits.append({
                    "id": cid, "text": res["documents"][0][i], "metadata": meta,
                    "parent_id": meta.get("parent_id"),
                    "score": 1.0 - float(res["distances"][0][i]),  # cosine similarity
                    "retrieval_type": "dense",
                })
            return hits
        return self._search_fallback(query, top_k, "dense_fallback")

    def _load_bm25(self):
        if self._bm25_cache is None and HAVE_BM25 and self.bm25_path.exists():
            with self._lock:
                if self._bm25_cache is None:
                    with open(self.bm25_path, "rb") as f:
                        data = pickle.load(f)  # trusted: written by build_indexes
                    self._bm25_cache = (data["bm25"], data["child_chunks"])
        return self._bm25_cache

    def search_sparse(self, query: str, top_k: int = settings.RETRIEVAL_TOP_K_SPARSE) -> List[Dict[str, Any]]:
        if self.es:
            try:
                res = self.es.search(index=settings.ELASTICSEARCH_INDEX, body={"size": top_k, "query": {"match": {"content": query}}})
                hits = [{"id": h["_source"]["id"], "text": h["_source"]["content"], "metadata": h["_source"].get("metadata", {}),
                         "parent_id": h["_source"].get("parent_id"), "score": float(h["_score"]), "retrieval_type": "elasticsearch_bm25"}
                        for h in res["hits"]["hits"]]
                if hits:
                    return hits
            except Exception as e:
                log.warning("Elasticsearch search failed (%s); using local BM25.", e)

        cache = self._load_bm25()
        if cache:
            bm25, chunks = cache
            scores = bm25.get_scores(tokenize(query))
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            return [{"id": chunks[i]["id"], "text": chunks[i]["text"], "metadata": chunks[i]["metadata"],
                     "parent_id": chunks[i].get("parent_id"), "score": float(scores[i]), "retrieval_type": "rank_bm25"}
                    for i in order if scores[i] > 0]
        return self._search_fallback(query, top_k, "sparse_fallback")

    def _search_fallback(self, query: str, top_k: int, retrieval_type: str) -> List[Dict[str, Any]]:
        if not self.chunks_path.exists():
            return []
        chunks = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        q = set(tokenize(query))
        scored = [(len(q & set(tokenize(c["text"]))), c) for c in chunks]
        scored = sorted((x for x in scored if x[0] > 0), key=lambda x: x[0], reverse=True)[:top_k]
        return [{"id": c["id"], "text": c["text"], "metadata": c["metadata"], "parent_id": c.get("parent_id"),
                 "score": float(s), "retrieval_type": retrieval_type} for s, c in scored]
