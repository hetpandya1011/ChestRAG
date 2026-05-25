"""Hybrid retrieval against the pgvector store.

Given a question, search the `data_chunks` table in two ways at once — dense
(vector similarity) + sparse (Postgres full-text) — and return the top-k chunks,
each carrying its citation metadata (paper_id, title, page).

Run standalone to sanity-check retrieval (no Claude involved):
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


def retrieve(question: str, top_k: int = 5) -> list[NodeWithScore]:
    """Return the top-k chunks for `question` via hybrid (dense + full-text) search."""
    retriever = _index().as_retriever(
        vector_store_query_mode=VectorStoreQueryMode.HYBRID,
        similarity_top_k=top_k,
        sparse_top_k=top_k,
    )
    return retriever.retrieve(question)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is CheXzero and how does it detect pathologies?"
    print(f"Query: {question}\n")
    for i, nws in enumerate(retrieve(question), start=1):
        meta = nws.node.metadata
        score = nws.score if nws.score is not None else float("nan")
        snippet = " ".join(nws.node.get_content().split())[:200]
        print(f"[{i}] {meta.get('title')}  (p.{meta.get('page')})  score={score:.3f}")
        print(f"    {snippet}...\n")
