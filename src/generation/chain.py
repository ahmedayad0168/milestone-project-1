"""End-to-end RAG pipeline: rewrite -> expand -> hybrid retrieve -> rerank -> parent expansion -> Gemini."""
from __future__ import annotations

import re
from typing import Any, Dict, Iterator, List, Optional

from src.config import settings
from src.generation.conversation import History, rewrite_followup
from src.generation.llm import BaseLLM, LLMUnavailable, build_llm
from src.generation.prompts import ABSTAIN_MESSAGE, FINCON_SYSTEM_PROMPT, FINCON_USER_PROMPT_TEMPLATE
from src.ingestion.hierarchical import HierarchicalChunker
from src.logging_setup import get_logger
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.query_expansion import QueryExpander
from src.retrieval.reranker import Reranker
from src.text_utils import tokenize

log = get_logger(__name__)
_CITE_RE = re.compile(r"\[(\d{1,2})\]")


def extract_citations(answer: str, parents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Map [n] markers in the answer back to the retrieved parent passages."""
    seen, out = set(), []
    for m in _CITE_RE.finditer(answer or ""):
        n = int(m.group(1))
        if n in seen or not 1 <= n <= len(parents):
            continue
        seen.add(n)
        p = parents[n - 1]
        out.append({"index": n, "source": p["metadata"].get("source", "Unknown"), "title": p["metadata"].get("title", ""),
                    "parent_id": p["metadata"].get("parent_id")})
    return sorted(out, key=lambda c: c["index"])


def extractive_answer(query: str, parents: List[Dict[str, Any]], max_sentences: int = 4) -> str:
    """LLM-free answer: the most query-relevant sentences from the retrieved passages, with citations."""
    q = set(tokenize(query))
    scored = []
    for i, p in enumerate(parents, start=1):
        for sent in re.split(r"(?<=[.!?])\s+|\n+", p["text"]):
            sent = sent.strip()
            if len(sent) < 25 or sent.lower().startswith("title:"):
                continue
            overlap = len(q & set(tokenize(sent)))
            if overlap:
                bonus = 0.5 if re.search(r"\d", sent) else 0.0  # figures/thresholds are usually what finance questions ask for
                scored.append((overlap + bonus, -i, sent, i))
    if not scored:
        return ABSTAIN_MESSAGE
    best = sorted(scored, reverse=True)[:max_sentences]
    best.sort(key=lambda x: (x[3], x[2]))
    return "\n".join(f"- {s} [{i}]" for _, _, s, i in best)


class FinConRAGPipeline:
    def __init__(
        self,
        expander: Optional[QueryExpander] = None,
        retriever: Optional[HybridRetriever] = None,
        reranker: Optional[Reranker] = None,
        parent_store: Optional[Dict[str, Dict[str, Any]]] = None,
        llm: Optional[BaseLLM] = None,
    ):
        self.expander = expander or QueryExpander()
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or Reranker()
        self.parent_store = parent_store if parent_store is not None else HierarchicalChunker.load_parent_store()
        self.llm = llm or build_llm()

    def refresh_parent_store(self):
        self.parent_store = HierarchicalChunker.load_parent_store()

    # ── 1. retrieval (everything before generation) ───────────────────────
    def retrieve(self, query: str, top_k: int = settings.RERANK_TOP_K, history: Optional[History] = None,
                 where_filter: Optional[Dict] = None) -> Dict[str, Any]:
        if not self.parent_store:
            self.refresh_parent_store()

        standalone = rewrite_followup(query, history, self.llm)
        entities = self.expander.extract_entities(standalone)
        expanded = self.expander.expand_query(standalone)

        candidates = self.retriever.retrieve(expanded, top_k=settings.RETRIEVAL_CANDIDATES, where_filter=where_filter,
                                             sparse_query=expanded)
        reranked = self.reranker.rerank(standalone, candidates, top_k=top_k)

        # Parent expansion + de-duplication; citation numbers are sequential over the *kept* parents.
        parents: List[Dict[str, Any]] = []
        seen = set()
        for child in reranked:
            pid = child.get("parent_id") or child["metadata"].get("parent_id")
            if pid in seen:
                continue
            seen.add(pid)
            info = self.parent_store.get(pid) or {"text": child["text"], "metadata": child["metadata"]}
            parents.append({"text": info["text"], "metadata": {**info["metadata"], "parent_id": pid}})

        blocks, budget = [], settings.MAX_CONTEXT_CHARS
        for n, p in enumerate(parents, start=1):
            text = p["text"][: max(400, budget)] if budget < len(p["text"]) else p["text"]
            budget -= len(text)
            title = p["metadata"].get("title", "")
            blocks.append(f"[{n}] Source: {p['metadata'].get('source', 'Unknown')}" + (f" — {title}" if title else "") + f"\n{text}\n")
        context = "\n".join(blocks) if blocks else "No relevant documents found."

        top_score = reranked[0]["rerank_score"] if reranked else None
        abstain = not reranked or (self.reranker.is_cross_encoder and top_score < settings.ABSTAIN_RERANK_THRESHOLD)

        return {
            "query": query,
            "standalone_query": standalone,
            "was_rewritten": standalone != query,
            "entities": entities,
            "expanded_query": expanded,
            "retrieved_child_chunks": reranked,
            "retrieved_parent_chunks": parents,
            "context_used": context,
            "sources": list(dict.fromkeys(p["metadata"].get("source", "Unknown") for p in parents)),
            "top_rerank_score": top_score,
            "abstained": bool(abstain),
        }

    # ── 2. generation ─────────────────────────────────────────────────────
    def _messages(self, r: Dict[str, Any]) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": FINCON_SYSTEM_PROMPT},
            {"role": "user", "content": FINCON_USER_PROMPT_TEMPLATE.format(context_blocks=r["context_used"], query=r["standalone_query"])},
        ]

    def generate_stream(self, r: Dict[str, Any]) -> Iterator[str]:
        """Yield the answer incrementally. Never raises: failures degrade to an extractive answer."""
        if r["abstained"]:
            yield ABSTAIN_MESSAGE
            return
        if not self.llm.is_generative:
            yield extractive_answer(r["standalone_query"], r["retrieved_parent_chunks"])
            return
        emitted = False
        try:
            for token in self.llm.stream(self._messages(r)):
                emitted = True
                yield token
        except Exception as e:  # noqa: BLE001
            log.error("LLM generation failed: %s", e)
            if emitted:
                yield "\n\n_(Generation was interrupted. Please retry.)_"
            else:
                yield "_(The language model is unavailable right now, showing the most relevant passages instead.)_\n\n"
                yield extractive_answer(r["standalone_query"], r["retrieved_parent_chunks"])

    def finalize(self, r: Dict[str, Any], answer: str) -> Dict[str, Any]:
        out = dict(r)
        out["answer"] = answer
        out["citations"] = extract_citations(answer, r["retrieved_parent_chunks"])
        out["llm"] = self.llm.name
        return out

    def run(self, query: str, top_k_rerank: int = settings.RERANK_TOP_K, history: Optional[History] = None,
            where_filter: Optional[Dict] = None) -> Dict[str, Any]:
        r = self.retrieve(query, top_k=top_k_rerank, history=history, where_filter=where_filter)
        return self.finalize(r, "".join(self.generate_stream(r)))
