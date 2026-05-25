"""ChestRAG API — Weekend 1.

/health   liveness probe (used by the web container).
/query    retrieve -> generate -> return a grounded answer + the cited sources.
"""
from __future__ import annotations

import re
import time

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from services.api.generate import generate_answer
from services.api.retrieval import retrieve

load_dotenv()

# Structured JSON logs from day one (a JD requirement: "production logging").
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger()

app = FastAPI(title="ChestRAG API", version="0.1.0")

# Matches inline citation markers like [1], [2] in the model's answer.
CITATION_RE = re.compile(r"\[(\d+)\]")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class Citation(BaseModel):
    n: int
    paper_id: str
    title: str | None
    page: int | None
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe. The web container pings this to prove connectivity."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    t0 = time.perf_counter()
    nodes = retrieve(req.question, top_k=req.top_k)
    answer = generate_answer(req.question, nodes)

    # Keep only the sources the answer actually cited, in order of first appearance.
    cited_ns: list[int] = []
    for match in CITATION_RE.findall(answer):
        n = int(match)
        if 1 <= n <= len(nodes) and n not in cited_ns:
            cited_ns.append(n)

    citations = []
    for n in cited_ns:
        node = nodes[n - 1].node
        meta = node.metadata
        citations.append(
            Citation(
                n=n,
                paper_id=meta.get("paper_id", ""),
                title=meta.get("title"),
                page=meta.get("page"),
                snippet=" ".join(node.get_content().split())[:240],
            )
        )

    latency_ms = round((time.perf_counter() - t0) * 1000)
    log.info(
        "query",
        question=req.question,
        n_retrieved=len(nodes),
        n_cited=len(citations),
        latency_ms=latency_ms,
    )
    return QueryResponse(answer=answer, citations=citations)
