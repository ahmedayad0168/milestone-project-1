"""Hybrid retrieval: dense + BM25 fused with Reciprocal Rank Fusion."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from src.config import settings
from src.ingestion.indexer import Indexer


def rrf_fuse(rankings: Dict[str, List[Dict[str, Any]]], k: int = settings.RRF_K, top_k: int = 20) -> List[Dict[str, Any]]:
    """Fuse several ranked lists. ``rankings`` maps a label (e.g. 'dense') to its ordered hits.

    RRF(d) = sum over lists of 1 / (k + rank_in_list(d)). Input dicts are copied, never mutated.
    """
    scores: Dict[str, float] = defaultdict(float)
    items: Dict[str, Dict[str, Any]] = {}
    for label, hits in rankings.items():
        for rank, hit in enumerate(hits, start=1):
            cid = hit["id"]
            scores[cid] += 1.0 / (k + rank)
            if cid not in items:
                items[cid] = dict(hit)
            items[cid][f"{label}_rank"] = rank
    ordered = sorted(scores, key=lambda cid: (-scores[cid], cid))[:top_k]
    return [{**items[cid], "rrf_score": scores[cid]} for cid in ordered]


class HybridRetriever:
    def __init__(self, indexer: Optional[Indexer] = None, rrf_k: int = settings.RRF_K):
        self.indexer = indexer or Indexer()
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = settings.RETRIEVAL_CANDIDATES, where_filter: Optional[Dict] = None,
                 mode: str = "hybrid", sparse_query: Optional[str] = None) -> List[Dict[str, Any]]:
        """mode: 'hybrid' | 'dense' | 'sparse' (the last two exist for ablation studies)."""
        rankings: Dict[str, List[Dict[str, Any]]] = {}
        if mode in ("hybrid", "dense"):
            rankings["dense"] = self.indexer.search_dense(query, settings.RETRIEVAL_TOP_K_DENSE, where=where_filter)
        if mode in ("hybrid", "sparse"):
            rankings["sparse"] = self.indexer.search_sparse(sparse_query or query, settings.RETRIEVAL_TOP_K_SPARSE)
        return rrf_fuse(rankings, k=self.rrf_k, top_k=top_k)
