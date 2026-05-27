"""Agentic layer over the RAG pipeline — what makes this "agentic RAG", not just RAG.

The plain /query path is a fixed pipeline: retrieve -> generate. This agent instead
hands the LLM two tools and lets *it* decide what to do:

  - search_corpus(query)        -> hybrid retrieval over the 241-paper corpus
                                   ("what does the literature say?")
  - hf_model_lookup(model_name) -> live HuggingFace metadata: license, downloads,
                                   maintenance ("what does the model ecosystem look
                                   like today?")

The loop: call the LLM with both tools available; if it requests tool calls, we run
them, feed the results back, and call again; repeat until the LLM answers with no more
tool calls. The LLM is the router — we don't hardcode "if 'license' in question". A
question like "Compare CheXagent and CheXzero, and which has a more permissive license?"
makes it call BOTH tools and synthesize. That's the agentic story.

Run standalone:
    python services/api/agent.py "What's the license and download count for BiomedCLIP?"
"""
from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from services.api.retrieval import retrieve
from services.api.tools.hf_lookup import HF_LOOKUP_SCHEMA, hf_model_lookup

load_dotenv()

AGENT_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o")
MAX_STEPS = 5  # safety cap so a misbehaving model can't loop forever

AGENT_SYSTEM = (
    "You are a research assistant for the chest X-ray AI literature. You have two tools:\n"
    "- search_corpus: retrieves passages from a curated corpus of chest X-ray AI papers. "
    "Use it for questions about what the literature says — methods, results, datasets, "
    "comparisons between approaches.\n"
    "- hf_model_lookup: returns LIVE metadata about a model on HuggingFace (license, "
    "downloads, likes, maintenance status). Use it for questions about a model's license, "
    "popularity, availability, or how current it is.\n"
    "Some questions need both tools — call them as needed. When you cite passages from "
    "search_corpus, cite them inline as [1], [2] matching the numbered sources. When you "
    "state ecosystem facts (license, downloads), say they come from HuggingFace. "
    "Answer using ONLY tool results; if they don't contain the answer, say so. Never invent "
    "facts, citations, licenses, or numbers."
)

# search_corpus schema — the corpus retrieval exposed as a tool the LLM can choose.
SEARCH_CORPUS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_corpus",
        "description": (
            "Search the curated chest X-ray AI research-paper corpus and return the most "
            "relevant passages, numbered [1]..[n] for citation. Use for any question about "
            "what the literature reports: methods, datasets, results, model comparisons."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to retrieve relevant paper passages.",
                }
            },
            "required": ["query"],
        },
    },
}

TOOLS = [SEARCH_CORPUS_SCHEMA, HF_LOOKUP_SCHEMA]


def _run_search_corpus(query: str, sources: list) -> str:
    """Retrieve from the corpus, append nodes to the shared `sources` list (so the API
    layer can build citations), and return the numbered passages for the LLM to read."""
    nodes = retrieve(query, top_k=5)
    offset = len(sources)
    sources.extend(nodes)
    # Re-number from the running offset so citation indices stay unique across calls.
    blocks = []
    for i, nws in enumerate(nodes, start=offset + 1):
        meta = nws.node.metadata
        title, page = meta.get("title"), meta.get("page")
        blocks.append(f"[{i}] {title} (p.{page}):\n{nws.node.get_content()}")
    return "\n\n".join(blocks) if blocks else "No relevant passages found."


def run_agent(question: str) -> dict:
    """Run the tool-use loop. Returns {answer, sources, steps} where `sources` are the
    corpus nodes pulled in (for citation rendering) and `steps` is the tool-call trace."""
    client = OpenAI()
    messages = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": question},
    ]
    sources: list = []  # corpus nodes accumulated across search_corpus calls
    trace: list[dict] = []  # which tools the model called, for transparency/logging

    for _ in range(MAX_STEPS):
        resp = client.chat.completions.create(
            model=AGENT_MODEL,
            temperature=0.1,
            messages=messages,
            tools=TOOLS,
        )
        msg = resp.choices[0].message

        # No tool calls -> the model is done; this is the final answer.
        if not msg.tool_calls:
            return {"answer": msg.content or "", "sources": sources, "steps": trace}

        # Echo the assistant's tool-call message back into the history (required by the API).
        messages.append(msg)
        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments or "{}")
            if name == "search_corpus":
                result = _run_search_corpus(args.get("query", ""), sources)
            elif name == "hf_model_lookup":
                result = json.dumps(hf_model_lookup(args.get("model_name", "")))
            else:
                result = f"Unknown tool: {name}"
            trace.append({"tool": name, "args": args})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    # Hit the step cap without a final answer — make one last call with no tools to force one.
    final = client.chat.completions.create(
        model=AGENT_MODEL, temperature=0.1, messages=messages
    )
    return {"answer": final.choices[0].message.content or "", "sources": sources, "steps": trace}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What's the license and download count for BiomedCLIP?"
    out = run_agent(q)
    print(f"Q: {q}\n")
    print(f"A: {out['answer']}\n")
    print(f"Tool calls: {out['steps']}")
    if out["sources"]:
        print("\nCorpus sources pulled:")
        for i, nws in enumerate(out["sources"], start=1):
            m = nws.node.metadata
            print(f"  [{i}] {m.get('title')} (p.{m.get('page')})")
