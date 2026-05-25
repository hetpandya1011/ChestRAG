"""Hybrid retrieval (+ cross-encoder re-ranking) against the pgvector store.

Pipeline: hybrid search (dense vector + Postgres full-text) over-retrieves
`candidate_k` chunks, then a cross-encoder re-ranks them and we keep the top_k.
Pass use_rerank=False to get the raw hybrid top_k (used to A/B the re-ranker in eval).

Run standalone to sanity-check retrieval:
    python services/api/retrieval.py "Compare CheXagent and CheXzero"
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
TABLE_NAME = "chunks"


@lru_cache(maxsize=1)
def _index() -> VectorStoreIndex:
    """Build the index/connection once and reuse it (don't reconnect per query)."""
    vector_store = PGVectorStore.from_params(
        host=os.getenv("POSTGRES_HOST", "localhost"),  # "db" inside Docker, localhost locally
        port=os.getenv("POSTGRES_PORT", "5432"),
        database=os.getenv("POSTGRES_DB", "chestrag"),
        user=os.getenv("POSTGRES_USER", "chestrag"),
        password=os.getenv("POSTGRES_PASSWORD", "chestrag"),
        table_name=TABLE_NAME,
        embed_dim=EMBED_DIM,
        hybrid_search=True,
        text_search_config="english",
    )
    embed_model = OpenAIEmbedding(model=EMBED_MODEL)
    return VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)


def retrieve(
    question: str,
    top_k: int = 5,
    candidate_k: int = 20,
    use_rerank: bool = False,
) -> list[NodeWithScore]:
    """Hybrid retrieve (default). Set use_rerank=True to add cross-encoder re-ranking.

    Eval (eval/run_eval.py, 50Q / 241-paper corpus) showed plain hybrid BEAT
    hybrid+rerank here -- single-answer recall@5 0.975 vs 0.925 -- because first-stage
    recall is already near-ceiling and the general MS-MARCO re-ranker mis-scores dense
    biomedical text. So re-ranking is OFF by default and kept as an evidence-backed toggle.
    """
    n = candidate_k if use_rerank else top_k
    retriever = _index().as_retriever(
        vector_store_query_mode=VectorStoreQueryMode.HYBRID,
        similarity_top_k=n,
        sparse_top_k=n,
    )
    candidates = retriever.retrieve(question)
    if not use_rerank:
        return candidates[:top_k]
    # Lazy import: keeps PyTorch out of imports that don't rerank (e.g. the test suite).
    from services.api.rerank import rerank

    return rerank(question, candidates, top_k=top_k)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is CheXzero and how does it detect pathologies?"
    print(f"Query: {question}\n")
    for i, nws in enumerate(retrieve(question), start=1):
        meta = nws.node.metadata
        score = nws.score if nws.score is not None else float("nan")
        snippet = " ".join(nws.node.get_content().split())[:200]
        loc = f"p.{meta.get('page')}, {meta.get('section')}"
        print(f"[{i}] {meta.get('title')}  ({loc})  score={score:.3f}")
        print(f"    {snippet}...\n")
