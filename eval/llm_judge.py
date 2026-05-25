"""LLM-as-judge: score whether a generated answer is faithful to its sources.

For a (question, answer, sources) triple, a judge LLM checks whether each factual
claim in the answer is supported by the retrieved source chunks, returning a
faithfulness score (fraction of claims supported) + the unsupported claims it found
(hallucination flags).

The judge is a separate call from the generator (standard LLM-as-judge). It is still
model-grading-model, so treat the score as a strong signal, not ground truth.

Try it on one question (db running, OPENAI_API_KEY set):
    python -m eval.llm_judge "How does CheXpert handle uncertain labels?"
"""
from __future__ import annotations

import json
import os
import sys

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.schema import NodeWithScore
from llama_index.llms.openai import OpenAI

from services.api.generate import format_sources, generate_answer
from services.api.retrieval import retrieve

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o")

JUDGE_SYSTEM = (
    "You are a strict fact-checker for a retrieval-augmented QA system. You are given "
    "a QUESTION, numbered SOURCES, and an ANSWER generated from them. Judge whether each "
    "factual claim in the ANSWER is supported by the SOURCES. A claim is supported only if "
    "a source states or clearly implies it; treat unsupported specifics (numbers, model "
    "names, results) as hallucinations. "
    "IMPORTANT: if the ANSWER declines to answer or says it lacks enough information (and "
    "makes no other factual claims), that is correct behavior, NOT a hallucination -- score "
    "faithfulness 1.0, verdict \"supported\", and unsupported_claims []. "
    "Respond with ONLY a JSON object, no prose."
)

JUDGE_TEMPLATE = """QUESTION:
{question}

SOURCES:
{sources}

ANSWER:
{answer}

Return JSON exactly in this shape:
{{"faithfulness": <number 0.0-1.0, the fraction of the answer's claims supported by the sources>,
  "verdict": "supported" | "partially_supported" | "unsupported",
  "unsupported_claims": [<short strings; empty list if none>]}}"""


def _extract_json(text: str) -> str:
    """Pull the first {...} block out of the reply (in case the model adds fences/prose)."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found")
    return text[start : end + 1]


def judge_faithfulness(question: str, answer: str, nodes: list[NodeWithScore]) -> dict:
    """Ask the judge LLM whether `answer` is supported by `nodes`; return a parsed dict."""
    prompt = JUDGE_TEMPLATE.format(
        question=question, sources=format_sources(nodes), answer=answer
    )
    llm = OpenAI(model=JUDGE_MODEL, temperature=0.0)
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=JUDGE_SYSTEM),
        ChatMessage(role=MessageRole.USER, content=prompt),
    ]
    raw = llm.chat(messages).message.content or ""
    try:
        data = json.loads(_extract_json(raw))
    except (json.JSONDecodeError, ValueError):
        return {
            "faithfulness": None,
            "verdict": "parse_error",
            "unsupported_claims": [],
            "raw": raw,
        }
    return {
        "faithfulness": data.get("faithfulness"),
        "verdict": data.get("verdict"),
        "unsupported_claims": data.get("unsupported_claims", []),
    }


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How does CheXpert handle uncertain labels?"
    nodes = retrieve(question, top_k=5)
    answer = generate_answer(question, nodes)
    result = judge_faithfulness(question, answer, nodes)
    print(f"Q: {question}\n")
    print(f"A: {answer}\n")
    print("Judge:", json.dumps(result, indent=2))
