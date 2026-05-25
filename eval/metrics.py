"""Retrieval metrics for the eval harness.

Every function takes:
  retrieved -> paper_ids of the retrieved chunks, in rank order (one per chunk,
               duplicates allowed -- several chunks can come from the same paper)
  gold      -> paper_ids that should answer the question

They're pure functions (no DB, no API), so they're trivially unit-testable.

  hit_at_k    -> 1.0 if ANY gold paper is among the top-k chunks' papers, else 0.0
  recall_at_k -> fraction of gold papers found among the top-k chunks' papers
  mrr         -> 1 / rank of the first chunk from a gold paper (0.0 if none found)
"""
from __future__ import annotations

from collections.abc import Sequence


def hit_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int = 5) -> float:
    gold_set = set(gold)
    return 1.0 if any(p in gold_set for p in retrieved[:k]) else 0.0


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int = 5) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    found = {p for p in retrieved[:k] if p in gold_set}
    return len(found) / len(gold_set)


def mrr(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    gold_set = set(gold)
    for rank, paper_id in enumerate(retrieved, start=1):
        if paper_id in gold_set:
            return 1.0 / rank
    return 0.0


def aggregate(rows: list[dict]) -> dict:
    """Average the metrics across eval rows, split single-answer vs comparative.

    Each row is expected to have keys: "hit@5", "recall@5", "mrr", "type".
    """

    def mean(vals: list[float]) -> float:
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def summarize(subset: list[dict]) -> dict:
        return {
            "n": len(subset),
            "hit@5": mean([r["hit@5"] for r in subset]),
            "recall@5": mean([r["recall@5"] for r in subset]),
            "mrr": mean([r["mrr"] for r in subset]),
        }

    single = [r for r in rows if r.get("type") != "comparative"]
    comparative = [r for r in rows if r.get("type") == "comparative"]
    return {
        "overall": summarize(rows),
        "single_answer": summarize(single),
        "comparative": summarize(comparative),
    }


if __name__ == "__main__":
    # Tiny self-check: gold paper appears as the 3rd retrieved chunk.
    retrieved = ["chexpert", "chexpert", "mimic-cxr", "chexnet", "padchest"]
    gold = ["mimic-cxr"]
    print("hit@5:   ", hit_at_k(retrieved, gold))      # 1.0
    print("recall@5:", recall_at_k(retrieved, gold))   # 1.0
    print("mrr:     ", round(mrr(retrieved, gold), 3)) # 0.333
