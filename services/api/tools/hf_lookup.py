"""HuggingFace model-card lookup — the one agent tool.

Why this tool exists: the corpus answers "what does the literature say?". This tool
answers "what does the model ecosystem look like *today*?" — license, current download
counts, likes, maintenance status, code library — facts the papers don't contain and
that drift after publication. The agent must do something the corpus can't; this does.

It hits the public HuggingFace Hub API (no auth needed for public models). Given a model
name, it searches the Hub, takes the best match, and returns its operational metadata.
"""
from __future__ import annotations

import httpx

HF_API = "https://huggingface.co/api"
TIMEOUT = 10.0


def _license_from_tags(tags: list[str]) -> str | None:
    """HF stores the license inside the tags list as e.g. 'license:mit'."""
    for tag in tags:
        if tag.startswith("license:"):
            return tag.split(":", 1)[1]
    return None


def _maintenance(last_modified: str | None) -> str:
    """Coarse 'is this still maintained?' signal from the last-modified date."""
    if not last_modified:
        return "unknown"
    year = last_modified[:4]
    return f"last updated {year}"


def hf_model_lookup(model_name: str) -> dict:
    """Search the HuggingFace Hub for `model_name` and return the top match's metadata.

    Returns a dict with: matched_id, license, downloads, likes, pipeline_tag,
    library, maintenance, tags, url. On no match or API error, returns {"error": ...}
    so the agent can tell the user the model wasn't found rather than crashing.
    """
    try:
        # 1) Search the Hub for the name -> list of candidate repos (ranked by relevance).
        resp = httpx.get(
            f"{HF_API}/models",
            params={"search": model_name, "limit": 5},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        candidates = resp.json()
    except httpx.HTTPError as exc:
        return {"error": f"HuggingFace API request failed: {exc}"}

    if not candidates:
        return {"error": f"No HuggingFace model found matching '{model_name}'."}

    # 2) Take the top hit and fetch its full record (search results are abbreviated).
    best_id = candidates[0]["id"]
    try:
        detail = httpx.get(f"{HF_API}/models/{best_id}", timeout=TIMEOUT)
        detail.raise_for_status()
        m = detail.json()
    except httpx.HTTPError as exc:
        return {"error": f"Could not fetch details for '{best_id}': {exc}"}

    tags = m.get("tags", [])
    return {
        "query": model_name,
        "matched_id": best_id,
        "license": _license_from_tags(tags),
        "downloads": m.get("downloads"),
        "likes": m.get("likes"),
        "pipeline_tag": m.get("pipeline_tag"),
        "library": m.get("library_name"),
        "maintenance": _maintenance(m.get("lastModified")),
        "tags": [t for t in tags if not t.startswith(("region:", "license:"))][:10],
        "url": f"https://huggingface.co/{best_id}",
        "other_candidates": [c["id"] for c in candidates[1:4]],
    }


# OpenAI function-calling schema. This is what we hand the LLM so it KNOWS the tool
# exists and when to reach for it. The description is the routing signal — the model
# reads it to decide "this question needs live model metadata, not the corpus."
HF_LOOKUP_SCHEMA = {
    "type": "function",
    "function": {
        "name": "hf_model_lookup",
        "description": (
            "Look up live operational metadata for a machine-learning model on the "
            "HuggingFace Hub: its license, current download count, likes, code library, "
            "and how recently it was updated. Use this when the user asks about a model's "
            "license, popularity, availability, maintenance status, or how to load it — "
            "facts that change over time and are NOT in the research-paper corpus. Do NOT "
            "use this for questions about what a paper reported or how a method works."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "The model name to look up, e.g. 'CheXagent' or 'BiomedCLIP'.",
                }
            },
            "required": ["model_name"],
        },
    },
}


if __name__ == "__main__":
    import json
    import sys

    name = " ".join(sys.argv[1:]) or "CheXagent"
    print(json.dumps(hf_model_lookup(name), indent=2))
