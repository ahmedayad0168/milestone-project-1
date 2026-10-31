# 💼 FinCon & Financial Regulatory Intelligence Assistant

A **hierarchical hybrid RAG** assistant for financial controlling and regulatory accounting (IFRS, ASC, CSRD/ESRS, Basel/FRTB, SEC rules), powered by **Google Gemini**.
Streamlit chat UI + FastAPI backend, deployable to Streamlit Community Cloud or Docker.

## Architecture

```
question ──► follow-up rewrite (chat memory, Gemini) ──► NER + domain query expansion
                                                              │
            ┌─────────────────────────────────────────────────┘
            ▼
   Hybrid retrieval over CHILD chunks (300 chars)
   ├─ dense  : ChromaDB (cosine) + multilingual MiniLM embeddings
   └─ sparse : BM25 (local rank_bm25, or Elasticsearch)
            │  Reciprocal Rank Fusion (top 20)
            ▼
   Cross-encoder rerank ──► abstain if best score < threshold ("I don't know")
            │
            ▼
   Expand to PARENT chunks (1200 chars) → numbered context [1] [2] [3]
            │
            ▼
   Gemini (streaming) → grounded answer with inline [n] citations
```

### Feature: conversational follow-ups
Ask *"What does IFRS S2 require for emissions?"* then simply *"and what about Scope 3?"*. The last few turns are used to rewrite the follow-up into a standalone question **before** retrieval (a cheap heuristic skips the extra LLM call for self-contained questions). The UI shows `↪ Interpreted as: …` so the rewrite is transparent. The API supports it via the `history` field.

### What changed in v2
| Area | Change |
|---|---|
| LLM | Groq/LangChain → **Gemini** (`google-genai`), streaming, retries, model fallback, graceful "extractive" mode when no key is set |
| Feature | **Chat memory + follow-up rewriting** (UI and API) |
| Answers | Sequential `[n]` citations mapped to sources, **abstention** on weak evidence, tighter grounding prompt |
| Fixes | Deterministic chunk IDs (re-ingestion no longer duplicates), Chroma metadata sanitising (lists/None crashed it), cosine space (scores were wrong with L2), BM25 cached in RAM and punctuation-aware, retriever no longer mutates inputs, broken `settings.LLM_MODEL` reference removed, Elasticsearch only contacted when enabled, atomic parent-store writes |
| Evaluation | No more free 100% correctness for questions without ground truth, non-circular Recall, Hit@k/MRR, **ablation table** (dense vs sparse vs hybrid vs +rerank), curated golden set |
| Ops | Auto-builds the index on first start, `/ready` endpoint, per-session question limit, CI workflow, 29 offline tests, lean CPU-only dependencies |

## Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add your GEMINI_API_KEY (free key: https://aistudio.google.com/apikey)

streamlit run ui/streamlit_app.py                     # UI  (index is built automatically on first start)
uvicorn api.app:app --reload                          # API → http://localhost:8000/docs
python -m src.ingestion.run_ingestion                 # (optional) rebuild the index manually
python -m pytest -q                                   # tests (offline)
```

**Your documents:** drop `.json` / `.csv` / `.pdf` / `.txt` / `.md` files into `data/raw/` and re-ingest. JSON files should be a list of objects with `title` and `content` (extra fields become metadata). If `data/raw/` has nothing ingestible, a small bundled seed corpus (`data/seed/`) is used so the app always starts.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (replace the old files; delete `app_1.py` and `env-credentials.md`).
2. On [share.streamlit.io](https://share.streamlit.io) → **Create app** → pick the repo/branch, set **Main file path** to `ui/streamlit_app.py`.
3. **Advanced settings → Python version: 3.12**, and under **Secrets** paste:
   ```toml
   GEMINI_API_KEY = "your_key"
   ```
4. Deploy. On the first start the app downloads two small Hugging Face models (~0.6 GB total) and builds the index from `data/raw` (or the seed corpus) — expect 1–3 minutes. Later starts reuse the cached resources.

Memory: the default reranker (`ms-marco-MiniLM-L-6-v2`, ~90 MB) keeps the app within the free tier. Switching to `BAAI/bge-reranker-base` (better, multilingual) needs ~1 GB more and may exceed it.

## Docker

```bash
cp .env.example .env && docker compose up -d --build ui     # Streamlit on :8501
docker compose up -d --build api                            # FastAPI   on :8000
```

## API

```bash
curl -s localhost:8000/query -H 'content-type: application/json' -d '{
  "query": "and what about Scope 3?",
  "top_k": 3,
  "history": [
    {"role": "user", "content": "What does IFRS S2 require for emissions?"},
    {"role": "assistant", "content": "Scope 1 and 2 ..."}
  ]}'
```
`GET /health` (liveness) · `GET /ready` (index + LLM status) · `POST /query` · `POST /ingest` (open: no key required).

## Evaluation

```bash
python -m src.evaluation.evaluate_rag --ablation             # dense vs sparse vs hybrid vs hybrid+rerank (no LLM calls)
python -m src.evaluation.evaluate_rag --mode full --out eval_report.json   # + faithfulness, correctness, abstention (uses Gemini)
```
The curated golden set is `data/golden_eval_curated.json` (includes an out-of-scope question to test abstention). Add your own questions with `expected_keywords`; retrieval relevance is judged on those keywords. Tune `ABSTAIN_RERANK_THRESHOLD` with it. Run the ablation with the real models and put the table in your portfolio.

## Configuration

All settings are environment variables (or Streamlit secrets); see `.env.example`. Key ones: `GEMINI_API_KEY`, `GEMINI_MODEL` (default `gemini-3.5-flash`; change it if Google retires the id), `GEMINI_FALLBACK_MODELS`, `RERANKER_MODEL_NAME`, `ABSTAIN_RERANK_THRESHOLD`, `MEMORY_TURNS`, `MAX_QUESTIONS_PER_SESSION`.

## Project layout

```
ui/streamlit_app.py        chat UI                     api/app.py             FastAPI service
src/config.py              settings (env + secrets)    src/bootstrap.py       first-run index build
src/ingestion/             parser · hierarchical chunker · indexer · pipeline · BAAI loader
src/retrieval/             query expansion (NER) · hybrid RRF · cross-encoder reranker
src/generation/            llm (Gemini) · conversation memory · prompts · chain
src/evaluation/            metrics, ablation, CLI      tests/                 offline test-suite
data/seed/                 fallback corpus             data/golden_eval_curated.json
```

## Troubleshooting

* **"No LLM configured" in the sidebar** – `GEMINI_API_KEY` is missing; the app falls back to extractive answers.
* **Gemini 404 / model not found** – set `GEMINI_MODEL` to a currently available id (see Google AI Studio); `GEMINI_FALLBACK_MODELS` is tried automatically.
* **Out of memory on Streamlit Cloud** – keep the light reranker, or set `RERANKER_MODEL_NAME=none`.
* **ChromaDB complains about SQLite** – add `pysqlite3-binary` to `requirements.txt`; `src/__init__.py` picks it up automatically.
* Never commit `.env` or real keys; if a key ever was committed, rotate it.

> Seed-corpus notes are summaries for demonstration, not legal or accounting advice. The CSRD entry describes the *original* thresholds; EU "Omnibus" changes may apply — verify against current law.
