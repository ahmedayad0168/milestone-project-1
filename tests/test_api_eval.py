import pytest
from fastapi.testclient import TestClient

from src.config import settings
from src.evaluation import evaluate_rag as ev


@pytest.fixture()
def client(kb):
    from api.app import app

    with TestClient(app) as c:
        yield c


def test_health_ready_query(client):
    assert client.get("/health").json()["status"] == "healthy"
    ready = client.get("/ready").json()
    assert ready["ready"] and ready["indexed_chunks"] > 10
    r = client.post("/query", json={"query": "What is the Basel III output floor percentage for risk-weighted assets?", "top_k": 3})
    body = r.json()
    assert r.status_code == 200 and "72.5%" in body["answer"] and body["citations"]


def test_query_with_history_and_validation(client):
    hist = [{"role": "user", "content": "What does IFRS S2 require?"}, {"role": "assistant", "content": "Climate disclosures."}]
    assert client.post("/query", json={"query": "and Scope 3?", "history": hist}).json()["standalone_query"] != "and Scope 3?"
    assert client.post("/query", json={"query": "x"}).status_code == 422
    assert client.post("/query", json={"query": "ok question", "history": [{"role": "bad", "content": "z"}]}).status_code == 422


def test_ingest_is_open(client):
    r = client.post("/ingest")
    assert r.status_code == 200 and r.json()["stats"]["children"] > 10


def test_metrics():
    assert ev.keyword_relevance("The output floor is 72.5%", ["72.5%", "output floor", "nope"]) == pytest.approx(2 / 3)
    assert ev.hit_at_k([0, 0, 1], 2) == 0 and ev.hit_at_k([0, 0, 1], 3) == 1
    assert ev.mrr([0, 1, 1]) == 0.5
    assert ev.ndcg_at_k([1, 0, 0], 3) == 1.0 and ev.ndcg_at_k([0, 0, 1], 3) < 1.0
    assert ev.correctness("anything", "") is None  # no more free 100% scores
    assert ev.correctness("EBITDA increases", "EBITDA increases under IFRS 16") > 0.4


def test_retrieval_eval_and_ablation_run(pipeline):
    import json

    golden = json.loads(settings.GOLDEN_EVAL_PATH.read_text())
    res = {m: ev.evaluate_retrieval(pipeline, golden, m, 3) for m in ["dense", "sparse", "hybrid", "hybrid+rerank"]}
    print(res)
    assert res["hybrid"]["hit@3"] >= 0.7
    assert res["hybrid"]["n"] == 14  # the out-of-scope question is excluded from retrieval metrics
