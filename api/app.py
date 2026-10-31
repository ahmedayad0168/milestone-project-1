"""FastAPI backend for the FinCon RAG assistant."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from src.bootstrap import ensure_index
from src.config import settings
from src.generation.chain import FinConRAGPipeline
from src.ingestion.run_ingestion import run_ingestion_pipeline
from src.logging_setup import get_logger

log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_index()
    app.state.pipeline = FinConRAGPipeline()
    log.info("Pipeline ready (LLM: %s)", app.state.pipeline.llm.name)
    yield


app = FastAPI(
    title="FinCon & Regulatory RAG Assistant API",
    description="Hierarchical hybrid RAG (BM25 + dense + cross-encoder reranking) for financial controlling, powered by Gemini.",
    version="2.0.0",
    lifespan=lifespan,
)


def get_pipeline(request: Request) -> FinConRAGPipeline:
    return request.app.state.pipeline


class ChatTurn(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., max_length=4000)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=settings.MAX_QUESTION_CHARS, examples=["What is the impact of IFRS 16 on EBITDA?"])
    top_k: int = Field(default=3, ge=1, le=10)
    history: List[ChatTurn] = Field(default_factory=list, description="Previous turns; enables follow-up questions.")


class Citation(BaseModel):
    index: int
    source: str
    title: str = ""
    parent_id: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    standalone_query: str
    entities: List[Dict[str, Any]] = []
    expanded_query: Optional[str] = None
    answer: str
    abstained: bool
    sources: List[str]
    citations: List[Citation] = []
    retrieved_child_chunks: List[Dict[str, Any]]
    context_used: str
    llm: str


class IngestionResponse(BaseModel):
    status: str
    message: str
    stats: Dict[str, int] = {}


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "healthy", "service": "FinCon RAG Assistant"}


@app.get("/ready")
def ready(pipeline: FinConRAGPipeline = Depends(get_pipeline)):
    chunks = pipeline.retriever.indexer.count()
    return {"ready": chunks > 0, "indexed_chunks": chunks, "llm": pipeline.llm.name}


@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest, pipeline: FinConRAGPipeline = Depends(get_pipeline)):
    try:
        res = pipeline.run(req.query, top_k_rerank=req.top_k, history=[t.model_dump() for t in req.history])
    except Exception:
        log.exception("Query failed")
        raise HTTPException(status_code=500, detail="Internal error while answering the question.")
    return QueryResponse(**{k: res[k] for k in QueryResponse.model_fields if k in res})


@app.post("/ingest", response_model=IngestionResponse)
def trigger_ingest(pipeline: FinConRAGPipeline = Depends(get_pipeline)):
    try:
        stats = run_ingestion_pipeline(indexer=pipeline.retriever.indexer)
        pipeline.refresh_parent_store()
    except Exception:
        log.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail="Ingestion failed; see server logs.")
    return IngestionResponse(status="success", message="Documents parsed and indexed.", stats=stats)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=False)
