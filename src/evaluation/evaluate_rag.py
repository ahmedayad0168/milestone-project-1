"""RAG evaluation.

Retrieval metrics are judged on the *parent passage* a chunk expands to (what the LLM actually sees):
a passage is relevant to a question in proportion to how many of its ``expected_keywords`` it contains.

    python -m src.evaluation.evaluate_rag                 # retrieval metrics for the full pipeline (no LLM calls)
    python -m src.evaluation.evaluate_rag --ablation      # dense vs sparse vs hybrid vs hybrid+rerank table
    python -m src.evaluation.evaluate_rag --mode full     # + faithfulness, correctness, abstention (calls the LLM)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import settings
from src.text_utils import content_terms


# ── metric primitives ─────────────────────────────────────────────────────
def keyword_relevance(text: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    low = text.lower()
    return sum(1 for k in keywords if k.lower() in low) / len(keywords)


def is_relevant(rel: float, threshold: float = 0.5) -> bool:
    return rel >= threshold


def precision_at_k(rels: List[float], k: int) -> float:
    return sum(is_relevant(r) for r in rels[:k]) / k if k > 0 else 0.0


def recall_at_k(rels: List[float], k: int) -> float:
    """Relevant found in top-k / relevant available in the retrieved pool."""
    total = sum(is_relevant(r) for r in rels)
    return (sum(is_relevant(r) for r in rels[:k]) / total) if total else 0.0


def hit_at_k(rels: List[float], k: int) -> float:
    return 1.0 if any(is_relevant(r) for r in rels[:k]) else 0.0


def mrr(rels: List[float]) -> float:
    for i, r in enumerate(rels, start=1):
        if is_relevant(r):
            return 1.0 / i
    return 0.0


def ndcg_at_k(rels: List[float], k: int) -> float:
    def dcg(xs):
        return sum((2.0 ** x - 1.0) / math.log2(i + 2) for i, x in enumerate(xs[:k]))

    ideal = dcg(sorted(rels, reverse=True))
    return min(1.0, dcg(rels) / ideal) if ideal > 0 else 0.0


def faithfulness(answer: str, context: str, embedder=None) -> float:
    """Share of the answer's content terms found in the context, blended with embedding similarity."""
    gen = content_terms(answer)
    if not gen:
        return 1.0
    ctx = context.lower()
    token_score = sum(1 for w in gen if w in ctx) / len(gen)
    if embedder is not None:
        try:
            a, c = embedder.encode([answer[:1000], context[:2500]])
            sem = max(0.0, min(1.0, (float(np.dot(a, c)) + 0.2) / 1.2))
            return round(0.5 * token_score + 0.5 * sem, 4)
        except Exception:
            pass
    return round(token_score, 4)


def correctness(answer: str, ground_truth: str, embedder=None) -> Optional[float]:
    """Lexical F1 blended with embedding similarity. Returns None (not 1.0!) when there is no ground truth."""
    if not ground_truth:
        return None
    gt, gen = content_terms(ground_truth), content_terms(answer)
    common = len(gt & gen)
    p, r = (common / len(gen) if gen else 0.0), (common / len(gt) if gt else 0.0)
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    if embedder is not None:
        try:
            a, g = embedder.encode([answer, ground_truth])
            sem = max(0.0, min(1.0, (float(np.dot(a, g)) + 0.1) / 1.1))
            return round(0.4 * f1 + 0.6 * sem, 4)
        except Exception:
            pass
    return round(f1, 4)


# ── retrieval under different configurations ──────────────────────────────
def ranked_parent_texts(pipeline, query: str, mode: str, pool: int = 20) -> List[str]:
    """Return de-duplicated parent passages for a retrieval configuration (dense | sparse | hybrid | hybrid+rerank)."""
    expanded = pipeline.expander.expand_query(query)
    base = "hybrid" if mode.startswith("hybrid") else mode
    cands = pipeline.retriever.retrieve(expanded, top_k=pool, mode=base, sparse_query=expanded)
    if mode == "hybrid+rerank":
        cands = pipeline.reranker.rerank(query, cands, top_k=pool)
    texts, seen = [], set()
    for c in cands:
        pid = c.get("parent_id") or c["metadata"].get("parent_id")
        if pid in seen:
            continue
        seen.add(pid)
        texts.append((pipeline.parent_store.get(pid) or {"text": c["text"]})["text"])
    return texts


def evaluate_retrieval(pipeline, golden: List[Dict[str, Any]], mode: str, k: int) -> Dict[str, float]:
    rows = []
    for item in golden:
        kws = item.get("expected_keywords") or []
        if item.get("should_abstain") or not kws:
            continue
        rels = [keyword_relevance(t, kws) for t in ranked_parent_texts(pipeline, item["query"], mode)]
        rows.append((hit_at_k(rels, k), mrr(rels), precision_at_k(rels, k), recall_at_k(rels, k), ndcg_at_k(rels, k)))
    if not rows:
        return {}
    avg = np.mean(rows, axis=0)
    return {"n": len(rows), f"hit@{k}": round(float(avg[0]), 4), "mrr": round(float(avg[1]), 4),
            f"precision@{k}": round(float(avg[2]), 4), f"recall@{k}": round(float(avg[3]), 4), f"ndcg@{k}": round(float(avg[4]), 4)}


def evaluate_generation(pipeline, golden: List[Dict[str, Any]], k: int) -> Dict[str, Any]:
    from src.models import get_embedder

    try:
        embedder = get_embedder()
    except Exception:
        embedder = None
    faith, corr, abst_ok, abst_n, details = [], [], 0, 0, []
    for item in golden:
        res = pipeline.run(item["query"], top_k_rerank=k)
        if item.get("should_abstain"):
            abst_n += 1
            abst_ok += int(res["abstained"] or "don't have enough information" in res["answer"].lower())
            details.append({"query": item["query"], "abstained": res["abstained"]})
            continue
        f = faithfulness(res["answer"], res["context_used"], embedder)
        c = correctness(res["answer"], item.get("ground_truth_answer", ""), embedder)
        faith.append(f)
        if c is not None:
            corr.append(c)
        details.append({"query": item["query"], "faithfulness": f, "correctness": c, "citations": len(res["citations"]),
                        "abstained": res["abstained"]})
    return {
        "faithfulness": round(float(np.mean(faith)), 4) if faith else None,
        "correctness": round(float(np.mean(corr)), 4) if corr else None,
        "abstention_accuracy": round(abst_ok / abst_n, 4) if abst_n else None,
        "details": details,
    }


def _print_table(rows: Dict[str, Dict[str, float]]):
    cols = [c for c in next(iter(rows.values())) if c != "n"]
    print(f"\n{'configuration':<16}" + "".join(f"{c:>14}" for c in cols))
    for name, m in rows.items():
        print(f"{name:<16}" + "".join(f"{m.get(c, 0):>14.4f}" for c in cols))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", type=Path, default=None)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    ap.add_argument("--min-hit-rate", type=float, default=None, help="exit 1 if hit@k is below this (CI guard)")
    args = ap.parse_args(argv)

    from src.generation.chain import FinConRAGPipeline

    golden_path = args.golden or settings.GOLDEN_EVAL_PATH
    if not golden_path.exists():
        golden_path = settings.BASE_DIR / "data" / "golden_eval_set.json"
    golden = json.loads(Path(golden_path).read_text(encoding="utf-8"))
    pipeline = FinConRAGPipeline()
    print(f"Evaluating on {len(golden)} questions from {golden_path.name} (k={args.k})")

    report: Dict[str, Any] = {"k": args.k, "golden_file": golden_path.name}
    modes = ["dense", "sparse", "hybrid", "hybrid+rerank"] if args.ablation else ["hybrid+rerank"]
    report["retrieval"] = {m: evaluate_retrieval(pipeline, golden, m, args.k) for m in modes}
    _print_table(report["retrieval"])

    if args.mode == "full":
        report["generation"] = evaluate_generation(pipeline, golden, args.k)
        g = report["generation"]
        print(f"\nFaithfulness: {g['faithfulness']}  Correctness: {g['correctness']}  Abstention accuracy: {g['abstention_accuracy']}")

    if args.out:
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    hit = report["retrieval"].get("hybrid+rerank", {}).get(f"hit@{args.k}")
    if args.min_hit_rate is not None and (hit is None or hit < args.min_hit_rate):
        print(f"FAIL: hit@{args.k}={hit} < {args.min_hit_rate}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
