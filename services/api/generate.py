"""Generate a grounded, cited answer from retrieved chunks.

Numbers the retrieved chunks as sources [1]..[n] and asks the LLM to answer using
ONLY those sources, citing them inline. If the sources don't contain the answer,
the model is told to say so — the anti-hallucination guardrail.

Run standalone to test retrieve -> generate end to end (no API server):
    python services/api/generate.py "How does CheXpert handle uncertain labels?"
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.schema import NodeWithScore
from llama_index.llms.openai import OpenAI

from services.api.retrieval import retrieve

load_dotenv()

GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o")

SYSTEM_PROMPT = (
    "You are a research assistant for the chest X-ray AI literature. "
    "Answer the question using ONLY the numbered sources provided. "
    "Cite the sources you use inline, like [1] or [2]. "
    "If the sources do not contain the answer, say you don't have enough "
    "information. Be concise and precise; never invent facts or citations."
)

PROMPT_TEMPLATE = """Sources:
{sources}

Question: {question}

Answer (cite sources inline as [n]):"""


def format_sources(nodes: list[NodeWithScore]) -> str:
    blocks = []
    for i, nws in enumerate(nodes, start=1):
        meta = nws.node.metadata
        blocks.append(
            f"[{i}] {meta.get('title')} (p.{meta.get('page')}):\n{nws.node.get_content()}"
        )
    return "\n\n".join(blocks)


def generate_answer(question: str, nodes: list[NodeWithScore]) -> str:
    """Ask the LLM to answer the question grounded in the retrieved chunks."""
    if not nodes:
        return "I don't have enough information in the corpus to answer that."
    prompt = PROMPT_TEMPLATE.format(sources=format_sources(nodes), question=question)
    llm = OpenAI(model=GENERATION_MODEL, temperature=0.1)
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=prompt),
    ]
    return llm.chat(messages).message.content


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How does CheXpert handle uncertain labels?"
    nodes = retrieve(question, top_k=5)
    answer = generate_answer(question, nodes)
    print(f"Q: {question}\n")
    print(f"A: {answer}\n")
    print("Citations:")
    for i, nws in enumerate(nodes, start=1):
        m = nws.node.metadata
        print(f"  [{i}] {m.get('title')} (p.{m.get('page')})")
