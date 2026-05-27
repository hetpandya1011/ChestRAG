"""ChestRAG API.

/health   liveness probe (used by the web container).
/query    retrieve -> generate -> return a grounded answer + the cited sources (fixed RAG).
/agent    tool-use loop: the LLM chooses search_corpus and/or hf_model_lookup (agentic RAG).
"""
from __future__ import annotations

import re
import time

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from services.api.agent import run_agent
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


def extract_cited(answer: str, n_nodes: int) -> list[int]:
    """Citation indices the answer actually used: in first-appearance order,
    de-duplicated, and limited to valid in-range sources."""
    cited: list[int] = []
    for match in CITATION_RE.findall(answer):
        n = int(match)
        if 1 <= n <= n_nodes and n not in cited:
            cited.append(n)
    return cited


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


class ToolCall(BaseModel):
    tool: str
    args: dict


class AgentResponse(BaseModel):
    answer: str
    citations: list[Citation]
    steps: list[ToolCall]  # the tools the agent chose to call, in order


def build_citations(answer: str, nodes: list) -> list[Citation]:
    """Turn the sources the answer cited ([n] markers) into Citation objects."""
    citations = []
    for n in extract_cited(answer, len(nodes)):
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
    return citations


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe. The web container pings this to prove connectivity."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Fixed RAG path: always retrieve, then generate. No tool routing."""
    t0 = time.perf_counter()
    nodes = retrieve(req.question, top_k=req.top_k)
    answer = generate_answer(req.question, nodes)
    citations = build_citations(answer, nodes)

    latency_ms = round((time.perf_counter() - t0) * 1000)
    log.info(
        "query",
        question=req.question,
        n_retrieved=len(nodes),
        n_cited=len(citations),
        latency_ms=latency_ms,
    )
    return QueryResponse(answer=answer, citations=citations)


@app.post("/agent", response_model=AgentResponse)
def agent(req: QueryRequest) -> AgentResponse:
    """Agentic path: the LLM picks tools (search_corpus and/or hf_model_lookup)."""
    t0 = time.perf_counter()
    result = run_agent(req.question)
    citations = build_citations(result["answer"], result["sources"])
    steps = [ToolCall(tool=s["tool"], args=s["args"]) for s in result["steps"]]

    latency_ms = round((time.perf_counter() - t0) * 1000)
    log.info(
        "agent",
        question=req.question,
        n_tool_calls=len(steps),
        tools=[s.tool for s in steps],
        n_cited=len(citations),
        latency_ms=latency_ms,
    )
    return AgentResponse(answer=result["answer"], citations=citations, steps=steps)
