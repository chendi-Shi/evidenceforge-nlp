"""Bounded, read-only API using the same retrieval/ranking path as evaluation."""
from __future__ import annotations

import asyncio
import re
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

NOTICE = "Relevance does not establish support, contradiction, or medical validity."


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    method: Literal["bm25", "dense", "rrf", "ltr"] = "rrf"
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def nonempty(cls, value: str) -> str:
        value = value.strip()
        if not re.search(r"\w", value):
            raise ValueError("query must contain at least one word or number")
        return value


def create_app(data_dir: Path = Path("data/scifact"), cache_dir: Path = Path("cache"),
               model_dir: Path = Path("results/scifact/model"), dense: bool = True,
               threads: int = 2, index=None, ranker=None) -> FastAPI:
    """Inject index/ranker for offline contract tests; load real artifacts at startup."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal index, ranker
        if index is None:
            from evidenceforge.data import load_dataset
            from evidenceforge.retrieval import RetrievalIndex
            index = RetrievalIndex(load_dataset(data_dir), cache_dir, dense=dense, threads=threads)
            await run_in_threadpool(index.build)
        if ranker is None and model_dir.exists() and dense:
            from evidenceforge.ranker import EvidenceRanker
            ranker = EvidenceRanker.load(model_dir)
        if ranker is not None:
            ranker.validate_index(index.metadata)
        app.state.ready = True
        yield
        app.state.ready = False

    app = FastAPI(title="EvidenceForge", version="0.1.0", lifespan=lifespan,
                  description="Scientific evidence search. " + NOTICE)
    app.state.ready = False
    # One active ranking request avoids oversubscribing the small CPU deployment.
    lock = asyncio.Lock()

    @app.get("/health")
    def health():
        return {"ready": app.state.ready, "dense": dense, "ranker": ranker is not None}

    @app.get("/metadata")
    def metadata():
        if not app.state.ready:
            raise HTTPException(503, "Index is not ready")
        return {"index": index.metadata, "notice": NOTICE,
                "candidate_depth": 50, "max_query_characters": 2000}

    @app.post("/search")
    async def search(request: SearchRequest):
        if not app.state.ready:
            raise HTTPException(503, "Index is not ready")
        if request.method != "bm25" and not dense:
            raise HTTPException(409, "This instance was started with --lexical-only")
        if request.method == "ltr" and ranker is None:
            raise HTTPException(409, "No trained ranker loaded; run the experiment or choose rrf")
        if lock.locked():
            raise HTTPException(429, "Search worker is busy; retry shortly", headers={"Retry-After": "1"})

        def retrieve():
            if request.method == "ltr":
                return ranker.rerank(request.query, index.search(request.query, method="rrf", k=50))
            return index.search(request.query, method=request.method, k=request.top_k)

        async with lock:
            start = perf_counter()
            results = await run_in_threadpool(retrieve)
            elapsed = (perf_counter() - start) * 1000
        return {"query": request.query, "method": request.method, "results": results[:request.top_k],
                "latency_ms": round(elapsed, 3), "notice": NOTICE}

    return app
