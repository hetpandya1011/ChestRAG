"""Evaluation harness: retrieval metrics (always) + optional LLM-as-judge faithfulness.

Retrieval pass: score the question bank with re-ranking OFF vs ON (recall@5/MRR vs
gold papers). Faithfulness pass (opt-in, costs GPT-4o calls): run the full
retrieve -> generate -> judge path on the first N questions and average faithfulness.
Results are written to eval/results/<timestamp>.json.

Prereqs: db container running, OPENAI_API_KEY in .env.

Usage (from repo root, inside the venv):
    python -m eval.run_eval                    # retrieval metrics only (cheap)
    python -m eval.run_eval --faithfulness 20  # also judge faithfulness on 20 questions
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from eval.llm_judge import judge_faithfulness
from eval.metrics import aggregate, hit_at_k, mrr, recall_at_k
from services.api.generate import generate_answer
from services.api.retrieval import retrieve

QUESTIONS_FILE = Path(__file__).resolve().parent / "questions.jsonl"
ADVERSARIAL_FILE = Path(__file__).resolve().parent / "adversarial.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
TOP_K = 5
CONFIGS = {"hybrid": False, "hybrid+rerank": True}  # name -> use_rerank


def load_questions() -> list[dict]:
    lines = QUESTIONS_FILE.read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def score_question(question: dict, use_rerank: bool) -> dict:
    nodes = retrieve(question["question"], top_k=TOP_K, use_rerank=use_rerank)
    retrieved = [n.node.metadata.get("paper_id") for n in nodes]
    gold = question["gold_paper_ids"]
    return {
        "id": question["id"],
        "type": question.get("type", "single"),
        "hit@5": hit_at_k(retrieved, gold, TOP_K),
        "recall@5": recall_at_k(retrieved, gold, TOP_K),
        "mrr": mrr(retrieved, gold),
        "retrieved": retrieved,
    }


def run_retrieval(questions: list[dict]) -> tuple[dict, dict]:
    summary, per_question = {}, {}
    for name, use_rerank in CONFIGS.items():
        print(f"Scoring {len(questions)} questions [{name}] ...")
        rows = [score_question(q, use_rerank) for q in questions]
        summary[name] = aggregate(rows)
        per_question[name] = rows
    return summary, per_question


def run_faithfulness(questions: list[dict], n: int) -> tuple[dict, list]:
    subset = questions[:n]
    print(f"\nJudging faithfulness on {len(subset)} questions (retrieve -> generate -> judge) ...")
    rows = []
    for q in subset:
        nodes = retrieve(q["question"], top_k=TOP_K)  # hybrid (default)
        answer = generate_answer(q["question"], nodes)
        verdict = judge_faithfulness(q["question"], answer, nodes)
        rows.append({
            "id": q["id"],
            "faithfulness": verdict.get("faithfulness"),
            "verdict": verdict.get("verdict"),
            "n_unsupported": len(verdict.get("unsupported_claims", [])),
        })

    scored = [r["faithfulness"] for r in rows if isinstance(r["faithfulness"], (int, float))]
    supported = sum(r["verdict"] == "supported" for r in rows)
    halluc = sum(r["n_unsupported"] > 0 for r in rows)
    summary = {
        "n": len(rows),
        "mean_faithfulness": round(sum(scored) / len(scored), 4) if scored else None,
        "pct_fully_supported": round(supported / len(rows), 4) if rows else None,
        "answers_with_unsupported_claims": halluc,
    }
    return summary, rows


REFUSAL_MARKERS = (
    "don't have enough information", "do not have enough information",
    "not enough information", "cannot determine", "do not specify",
    "does not specify", "no information", "unable to",
)


def is_refusal(answer: str) -> bool:
    text = answer.lower()
    return any(marker in text for marker in REFUSAL_MARKERS)


def load_adversarial() -> list[dict]:
    if not ADVERSARIAL_FILE.exists():
        return []
    lines = ADVERSARIAL_FILE.read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def run_adversarial(questions: list[dict]) -> tuple[dict, list]:
    """Unanswerable questions: the system should REFUSE, not fabricate."""
    print(f"\nAdversarial pass on {len(questions)} unanswerable questions ...")
    rows = []
    for q in questions:
        nodes = retrieve(q["question"], top_k=TOP_K)  # hybrid (default)
        answer = generate_answer(q["question"], nodes)
        verdict = judge_faithfulness(q["question"], answer, nodes)
        rows.append({
            "id": q["id"],
            "refused": is_refusal(answer),
            "faithfulness": verdict.get("faithfulness"),
            "n_unsupported": len(verdict.get("unsupported_claims", [])),
        })
    scored = [r["faithfulness"] for r in rows if isinstance(r["faithfulness"], (int, float))]
    summary = {
        "n": len(rows),
        "refusal_rate": round(sum(r["refused"] for r in rows) / len(rows), 4) if rows else None,
        "mean_faithfulness": round(sum(scored) / len(scored), 4) if scored else None,
        "n_hallucinated": sum(r["n_unsupported"] > 0 for r in rows),
    }
    return summary, rows


def main() -> int:
    parser = argparse.ArgumentParser(description="ChestRAG evaluation harness")
    parser.add_argument(
        "--faithfulness", type=int, default=0,
        help="also judge faithfulness on the first N questions (0 = skip)",
    )
    args = parser.parse_args()

    if not QUESTIONS_FILE.exists():
        print(f"No question set at {QUESTIONS_FILE}.")
        return 1
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_questions()

    retrieval_summary, retrieval_rows = run_retrieval(questions)

    print("\n=== Retrieval eval (top-5) ===")
    print(f"{'config':<16}{'split':<15}{'n':>4}{'hit@5':>9}{'recall@5':>10}{'mrr':>8}")
    for name in CONFIGS:
        for split, m in retrieval_summary[name].items():
            print(
                f"{name:<16}{split:<15}{m['n']:>4}"
                f"{m['hit@5']:>9}{m['recall@5']:>10}{m['mrr']:>8}"
            )

    results = {
        "top_k": TOP_K,
        "n_questions": len(questions),
        "retrieval": {"summary": retrieval_summary, "per_question": retrieval_rows},
    }

    if args.faithfulness > 0:
        faith_summary, faith_rows = run_faithfulness(questions, args.faithfulness)
        results["faithfulness"] = {"summary": faith_summary, "per_question": faith_rows}
        print("\n=== Faithfulness (hybrid path, LLM-as-judge) ===")
        print(json.dumps(faith_summary, indent=2))

        adversarial = load_adversarial()
        if adversarial:
            adv_summary, adv_rows = run_adversarial(adversarial)
            results["adversarial"] = {"summary": adv_summary, "per_question": adv_rows}
            print("\n=== Adversarial / refusal (unanswerable questions) ===")
            print(json.dumps(adv_summary, indent=2))

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_file = RESULTS_DIR / f"eval_{timestamp}.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
