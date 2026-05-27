"""Multi-model generation benchmark: Claude vs GPT vs Gemini on the eval set.

Retrieval is identical for every model (we retrieve once per question); only the
GENERATION model varies. For each (model, question) we record faithfulness
(LLM-as-judge), latency, token usage, and cost, then aggregate per model into a
cost / latency / quality table. Results -> eval/results/benchmark_<ts>.json.

Caveats (printed): the objective differentiators are cost + latency; faithfulness is
secondary because the judge is gpt-4o (slight self-bias toward the OpenAI row). A
provider that errors (e.g. a wrong model id) is logged and skipped, not fatal.

Prereqs: db running; OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY in .env.

Usage (from repo root, inside the venv):
    python -m eval.benchmark            # default subset
    python -m eval.benchmark -n 25      # first 25 questions
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from eval.llm_judge import judge_faithfulness
from services.api.generate import PROMPT_TEMPLATE, SYSTEM_PROMPT, format_sources
from services.api.retrieval import retrieve

load_dotenv()

QUESTIONS_FILE = Path(__file__).resolve().parent / "questions.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
TOP_K = 5
DEFAULT_N = 15

# model id (env-overridable) + approximate USD price per 1M tokens (input, output).
# Prices are approximate and may drift -- update as needed; the comparison is the point.
MODELS = {
    "openai": {"model": os.getenv("OPENAI_MODEL", "gpt-4o"), "in": 2.50, "out": 10.00},
    "anthropic": {
        "model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        "in": 3.00,
        "out": 15.00,
    },
    # Gemini was dropped from the default run: the free tier caps at 5 req/min (the
    # benchmark needs ~15) and the paid tier requires a $20 prepay. gen_gemini below
    # still works -- re-add a "gemini" entry here with a valid GEMINI_API_KEY to re-enable.
}


def load_questions() -> list[dict]:
    lines = QUESTIONS_FILE.read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _prompt(question: str, nodes: list) -> str:
    return PROMPT_TEMPLATE.format(sources=format_sources(nodes), question=question)


def gen_openai(model: str, question: str, nodes: list) -> tuple[str, int, int]:
    from openai import OpenAI

    resp = OpenAI().chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _prompt(question, nodes)},
        ],
    )
    u = resp.usage
    return resp.choices[0].message.content, u.prompt_tokens, u.completion_tokens


def gen_anthropic(model: str, question: str, nodes: list) -> tuple[str, int, int]:
    import anthropic

    resp = anthropic.Anthropic().messages.create(
        model=model,
        max_tokens=1024,
        temperature=0.1,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _prompt(question, nodes)}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return text, resp.usage.input_tokens, resp.usage.output_tokens


def gen_gemini(model: str, question: str, nodes: list) -> tuple[str, int, int]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.1)

    def _call() -> object:
        return client.models.generate_content(
            model=model, contents=_prompt(question, nodes), config=config
        )

    try:
        resp = _call()
    except Exception as exc:
        # On 429 rate-limit, parse the retry delay and wait once before giving up.
        m = re.search(r"retryDelay.*?(\d+)s", str(exc))
        wait = int(m.group(1)) + 2 if m else 45
        print(f"    gemini 429 — waiting {wait}s ...")
        time.sleep(wait)
        resp = _call()  # raises if it fails again

    um = resp.usage_metadata
    return resp.text, um.prompt_token_count, um.candidates_token_count


GENERATORS = {"openai": gen_openai, "anthropic": gen_anthropic, "gemini": gen_gemini}


def summarize(rows: list[dict], model: str) -> dict:
    ok = [r for r in rows if "error" not in r]
    faith = [r["faithfulness"] for r in ok if isinstance(r["faithfulness"], (int, float))]
    lat = [r["latency_s"] for r in ok]
    cost = [r["cost_usd"] for r in ok]
    return {
        "model": model,
        "n_ok": len(ok),
        "n_error": len(rows) - len(ok),
        "mean_faithfulness": round(sum(faith) / len(faith), 4) if faith else None,
        "mean_latency_s": round(sum(lat) / len(lat), 3) if lat else None,
        "avg_cost_per_query_usd": round(sum(cost) / len(cost), 6) if cost else None,
        "total_cost_usd": round(sum(cost), 6) if cost else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ChestRAG multi-model benchmark")
    parser.add_argument("-n", type=int, default=DEFAULT_N, help="number of questions to benchmark")
    args = parser.parse_args()

    if not QUESTIONS_FILE.exists():
        print(f"No question set at {QUESTIONS_FILE}.")
        return 1
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_questions()[: args.n]

    # Retrieve once per question; the same chunks feed every model.
    print(f"Retrieving for {len(questions)} questions ...")
    retrieved = {q["id"]: retrieve(q["question"], top_k=TOP_K) for q in questions}

    summary = {}
    for provider, cfg in MODELS.items():
        model = cfg["model"]
        print(f"\n[{provider}] generating with {model} ...")
        rows = []
        for q in questions:
            nodes = retrieved[q["id"]]
            t0 = time.perf_counter()
            try:
                text, in_tok, out_tok = GENERATORS[provider](model, q["question"], nodes)
            except Exception as exc:  # a bad model id / SDK mismatch shouldn't kill the run
                print(f"  ! {q['id']} failed: {exc}")
                rows.append({"id": q["id"], "error": str(exc)})
                continue
            latency = time.perf_counter() - t0
            verdict = judge_faithfulness(q["question"], text, nodes)
            cost = (in_tok / 1e6) * cfg["in"] + (out_tok / 1e6) * cfg["out"]
            rows.append({
                "id": q["id"],
                "faithfulness": verdict.get("faithfulness"),
                "latency_s": round(latency, 3),
                "in_tokens": in_tok,
                "out_tokens": out_tok,
                "cost_usd": round(cost, 6),
            })
        s = summarize(rows, model)
        s["rows"] = rows
        summary[provider] = s

    print(f"\n=== Multi-model benchmark (n={len(questions)}; judge=gpt-4o) ===")
    print(f"{'provider':<11}{'model':<22}{'ok':>4}{'err':>4}{'faith':>8}{'lat(s)':>9}{'$/query':>10}")
    for provider, s in summary.items():
        print(
            f"{provider:<11}{s['model']:<22}{s['n_ok']:>4}{s['n_error']:>4}"
            f"{str(s['mean_faithfulness']):>8}{str(s['mean_latency_s']):>9}"
            f"{str(s['avg_cost_per_query_usd']):>10}"
        )
    print("\nNote: faithfulness judged by gpt-4o (slight self-bias to the OpenAI row); "
          "cost + latency are the objective comparisons.")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_file = RESULTS_DIR / f"benchmark_{timestamp}.json"
    out_file.write_text(json.dumps({"n_questions": len(questions), "models": summary}, indent=2))
    print(f"\nWrote {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
