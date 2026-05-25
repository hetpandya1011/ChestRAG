"""Cross-encoder re-ranking of retrieved candidates.

Hybrid retrieval is fast but approximate (it compares separately-embedded vectors).
A cross-encoder scores each (query, chunk) pair *jointly* -- far more accurate -- so
we over-retrieve candidates upstream, re-score them here, and keep the best top_k.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (small, CPU-friendly). Downloaded once
from HuggingFace on first use and cached locally.
"""
from __future__ import annotations

import os
from functools import lru_cache

from llama_index.core.schema import NodeWithScore
from sentence_transformers import CrossEncoder

RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    """Load the cross-encoder once and reuse it."""
    return CrossEncoder(RERANK_MODEL)


def rerank(query: str, nodes: list[NodeWithScore], top_k: int = 5) -> list[NodeWithScore]:
    """Re-score (query, chunk) pairs with the cross-encoder; return the top_k nodes."""
    if not nodes:
        return []
    pairs = [(query, n.node.get_content()) for n in nodes]
    scores = _model().predict(pairs)
    ranked = sorted(zip(scores, nodes, strict=True), key=lambda pair: float(pair[0]), reverse=True)
    out: list[NodeWithScore] = []
    for score, nws in ranked[:top_k]:
        nws.score = float(score)  # replace the hybrid score with the cross-encoder score
        out.append(nws)
    return out
