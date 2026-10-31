from src.bootstrap import ensure_index, index_is_ready
from src.config import settings
from src.generation.conversation import needs_rewrite, rewrite_followup
from src.ingestion.run_ingestion import run_ingestion_pipeline


def test_bootstrap_builds_from_seed_then_is_noop(kb):
    assert kb["info"]["built"] and kb["info"]["chunks"] > 10
    assert index_is_ready()
    assert ensure_index()["built"] is False


def test_reingestion_is_idempotent(kb, pipeline):
    before = pipeline.retriever.indexer.count()
    run_ingestion_pipeline(indexer=pipeline.retriever.indexer)
    assert pipeline.retriever.indexer.count() == before  # no duplicates


def test_retrieval_finds_right_passage(pipeline):
    r = pipeline.retrieve("How does IFRS 16 affect EBITDA?", top_k=3)
    assert any("right-of-use" in p["text"] for p in r["retrieved_parent_chunks"])
    assert r["context_used"].startswith("[1] Source:")
    assert r["sources"] and not r["abstained"]


def test_citation_numbers_are_sequential_after_dedup(pipeline):
    r = pipeline.retrieve("What is the Form 8-K Item 1.05 deadline?", top_k=6)
    n = len(r["retrieved_parent_chunks"])
    for i in range(1, n + 1):
        assert f"[{i}] Source:" in r["context_used"]


def test_full_run_returns_answer_and_citations(pipeline):
    out = pipeline.run("What confidence level does Expected Shortfall use under FRTB?")
    assert "97.5%" in out["answer"] and out["citations"] and out["llm"].startswith("Extractive")


def test_abstains_when_reranker_score_is_low(pipeline, monkeypatch):
    class LowScore:
        is_cross_encoder = True

        def rerank(self, q, cands, top_k=3):
            return [{**c, "rerank_score": -9.0} for c in cands[:top_k]]

    pipeline.reranker = LowScore()
    out = pipeline.run("What is the capital of France?")
    assert out["abstained"] and "don't have enough information" in out["answer"]


def test_conversation_heuristics():
    hist = [{"role": "user", "content": "What does IFRS S2 require?"}, {"role": "assistant", "content": "It requires climate disclosures."}]
    assert needs_rewrite("and what about Scope 3?", hist)
    assert not needs_rewrite("and what about Scope 3?", [])
    assert not needs_rewrite("What are the CSRD size thresholds for large undertakings in the EU?", hist)


class FakeLLM:
    name, is_generative = "fake", True

    def __init__(self, reply=None, boom=False):
        self.reply, self.boom = reply, boom

    def chat(self, messages, max_tokens=None, temperature=None):
        if self.boom:
            raise RuntimeError("down")
        return self.reply


HIST = [{"role": "user", "content": "What does IFRS S2 require for emissions?"}, {"role": "assistant", "content": "Scope 1 and 2."}]


def test_llm_rewrite_used_and_cleaned():
    out = rewrite_followup("and Scope 3?", HIST, FakeLLM('Rewritten question: "What does IFRS S2 require for Scope 3 emissions?"\nextra'))
    assert out == "What does IFRS S2 require for Scope 3 emissions?"


def test_rewrite_falls_back_when_llm_fails():
    out = rewrite_followup("and Scope 3?", HIST, FakeLLM(boom=True))
    assert "and Scope 3?" in out and "IFRS S2" in out


def test_followup_flows_through_pipeline(pipeline):
    out = pipeline.run("and what about Scope 3?", history=HIST)
    assert out["was_rewritten"] and "IFRS S2" in out["standalone_query"]
    assert any("Scope 3" in p["text"] for p in out["retrieved_parent_chunks"])
