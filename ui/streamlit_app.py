"""Streamlit chat UI for the FinCon & Regulatory RAG Assistant."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(page_title="FinCon & Regulatory RAG Assistant", page_icon="💼", layout="wide")

from src.bootstrap import ensure_index  # noqa: E402
from src.config import settings  # noqa: E402
from src.generation.chain import FinConRAGPipeline  # noqa: E402

SAMPLE_QUESTIONS = [
    "What is the deadline for Form 8-K Item 1.05 after a material cybersecurity incident?",
    "What are Scope 1, Scope 2 and Scope 3 emissions under IFRS S2?",
    "How does IFRS 16 affect EBITDA?",
    "What is the Basel III Endgame output floor percentage?",
    "What confidence level does Expected Shortfall use under FRTB?",
]
FOLLOWUPS_HINT = "Try a follow-up such as “and what about Scope 3?” — the assistant remembers the conversation."


@st.cache_resource(show_spinner=False)
def load_pipeline():
    index_info = ensure_index()  # builds the index on first start (e.g. fresh Streamlit Cloud container)
    return FinConRAGPipeline(), index_info


def render_details(meta: dict):
    if meta.get("sources"):
        cited = {c["index"]: c for c in meta.get("citations", [])}
        cols = st.columns(min(len(meta["sources"]), 4))
        for i, src in enumerate(meta["sources"]):
            cols[i % len(cols)].caption(f"📄 {src}")
    with st.expander("🔎 How this answer was produced"):
        if meta.get("entities"):
            st.markdown("**Entities:** " + "  ".join(f"`{e['text']} ({e['label']})`" for e in meta["entities"]))
        if meta.get("expanded_query") and meta["expanded_query"] != meta.get("standalone_query"):
            st.markdown(f"**Expanded query:** `{meta['expanded_query']}`")
        if meta.get("top_rerank_score") is not None:
            st.markdown(f"**Best reranker score:** `{meta['top_rerank_score']:.2f}`")
        for i, ch in enumerate(meta.get("chunks", []), start=1):
            st.markdown(f"**#{i}** `{ch['id']}` · rerank `{ch.get('rerank_score', 0):.3f}` · RRF `{ch.get('rrf_score', 0):.4f}`")
            st.code(ch["text"], language=None)
        with st.popover("Full context sent to the LLM"):
            st.code(meta.get("context", ""), language="markdown")


def make_meta(out: dict) -> dict:
    return {
        "sources": out["sources"], "citations": out["citations"], "entities": out["entities"],
        "expanded_query": out["expanded_query"], "standalone_query": out["standalone_query"],
        "top_rerank_score": out["top_rerank_score"], "context": out["context_used"],
        "chunks": [{k: c.get(k) for k in ("id", "text", "rerank_score", "rrf_score")} for c in out["retrieved_child_chunks"]],
    }


# ── page ──────────────────────────────────────────────────────────────────
st.title("💼 FinCon & Regulatory Intelligence Assistant")
st.caption("Hierarchical hybrid RAG · BM25 + dense retrieval · cross-encoder reranking · numbered citations · conversational follow-ups")

with st.spinner("Starting up — building the knowledge index on first launch (this can take a minute)…"):
    pipeline, index_info = load_pipeline()

st.session_state.setdefault("messages", [])
st.session_state.setdefault("n_questions", 0)

with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Passages used (top-k)", 1, 6, settings.RERANK_TOP_K)
    st.caption(f"🤖 LLM: **{pipeline.llm.name}**")
    if not pipeline.llm.is_generative:
        st.warning("No LLM configured (`GEMINI_API_KEY` missing) — showing extractive answers from the retrieved passages.")
    st.caption(f"📚 Knowledge base: **{pipeline.retriever.indexer.count():,}** indexed chunks")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages, st.session_state.n_questions = [], 0
        st.rerun()
    st.divider()
    st.subheader("Sample questions")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True, key=f"sample_{q[:20]}"):
            st.session_state.pending = q
    st.divider()
    if st.button("🔄 Rebuild index", use_container_width=True):
        from src.ingestion.run_ingestion import run_ingestion_pipeline

        with st.spinner("Re-ingesting documents…"):
            st.success(run_ingestion_pipeline(indexer=pipeline.retriever.indexer))
            pipeline.refresh_parent_store()

if not st.session_state.messages:
    st.info(FOLLOWUPS_HINT)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("meta", {}).get("standalone_query") and msg["meta"].get("was_rewritten"):
                st.caption(f"↪ Interpreted as: {msg['meta']['standalone_query']}")
            render_details(msg["meta"])

prompt = st.chat_input("Ask about IFRS, ASC, CSRD, Basel, SEC rules, controlling…") or st.session_state.pop("pending", None)
if prompt:
    prompt = prompt.strip()[: settings.MAX_QUESTION_CHARS]
    if st.session_state.n_questions >= settings.MAX_QUESTIONS_PER_SESSION:
        st.warning("Session question limit reached. Clear the conversation to continue.")
        st.stop()
    st.session_state.n_questions += 1

    with st.chat_message("user"):
        st.markdown(prompt)
    history = [{"role": m["role"], "content": m["content"], "standalone": m.get("standalone")} for m in st.session_state.messages]

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base…"):
            retrieval = pipeline.retrieve(prompt, top_k=top_k, history=history)
        if retrieval["was_rewritten"]:
            st.caption(f"↪ Interpreted as: {retrieval['standalone_query']}")
        answer = st.write_stream(pipeline.generate_stream(retrieval))
        out = pipeline.finalize(retrieval, answer)
        meta = {**make_meta(out), "was_rewritten": retrieval["was_rewritten"]}
        render_details(meta)

    st.session_state.messages.append({"role": "user", "content": prompt, "standalone": retrieval["standalone_query"]})
    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})
