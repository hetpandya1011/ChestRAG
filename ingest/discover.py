"""Discover chest X-ray AI papers via the OpenAlex API -> data/corpus.jsonl.

Runs several targeted searches, keeps only open-access works that expose a PDF
URL (so fetch.py can actually download them), applies a light relevance filter
(must mention a chest-imaging term AND an AI term), de-duplicates, caps the set,
and writes one JSON record per line.

OpenAlex is free and needs no key. Set OPENALEX_MAILTO in .env to join the
"polite pool" (recommended, faster).

Usage (from repo root, inside the venv):
    python ingest/discover.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_FILE = REPO_ROOT / "data" / "corpus.jsonl"

OPENALEX = "https://api.openalex.org/works"
MAILTO = os.getenv("OPENALEX_MAILTO", "")

TARGET = 400          # rough cap on corpus size
FROM_YEAR = 2015      # modern CXR deep-learning era (still catches ChestX-ray8, 2017)
PER_PAGE = 200        # OpenAlex max page size

# Targeted searches over title+abstract; results are merged and de-duplicated.
QUERIES = [
    "chest radiograph deep learning",
    "chest x-ray deep learning",
    "chest x-ray classification convolutional neural network",
    "chest x-ray foundation model",
    "chest radiograph vision language model",
    "chest x-ray report generation",
    "chest x-ray self-supervised contrastive learning",
]

CHEST_TERMS = ("chest x-ray", "chest radiograph", "chest radiography", "cxr", "thoracic")
AI_TERMS = (
    "deep learning", "neural network", "machine learning", "transformer",
    "foundation model", "vision-language", "self-supervised", "contrastive",
    "artificial intelligence", "convolutional",
)


def invert_abstract(inverted_index: dict | None) -> str:
    """OpenAlex stores abstracts as a word -> positions index; rebuild the text."""
    if not inverted_index:
        return ""
    positions = [(pos, word) for word, idxs in inverted_index.items() for pos in idxs]
    return " ".join(word for _, word in sorted(positions))


def is_relevant(title: str, abstract: str) -> bool:
    text = f"{title} {abstract}".lower()
    return any(t in text for t in CHEST_TERMS) and any(t in text for t in AI_TERMS)


def pdf_url_for(work: dict) -> str | None:
    loc = work.get("best_oa_location") or work.get("primary_location") or {}
    return loc.get("pdf_url")


def search(client: httpx.Client, query: str) -> list[dict]:
    """Fetch the top page of open-access articles for one query."""
    params = {
        "filter": (
            f"title_and_abstract.search:{query},"
            "open_access.is_oa:true,type:article,"
            f"from_publication_date:{FROM_YEAR}-01-01"
        ),
        "per-page": str(PER_PAGE),
        "select": (
            "id,title,publication_year,authorships,doi,"
            "best_oa_location,primary_location,abstract_inverted_index"
        ),
    }
    if MAILTO:
        params["mailto"] = MAILTO
    resp = client.get(OPENALEX, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()["results"]


def to_record(work: dict, pdf_url: str) -> dict:
    auths = work.get("authorships") or []
    first_author = f"{auths[0]['author']['display_name']} et al." if auths else "Unknown"
    return {
        "id": work["id"].rsplit("/", 1)[-1],  # OpenAlex id, e.g. "W2099243483"
        "title": work.get("title") or "",
        "authors": first_author,
        "year": work.get("publication_year"),
        "doi": work.get("doi"),
        "source": urlparse(pdf_url).netloc.replace("www.", ""),
        "pdf_url": pdf_url,
    }


def main() -> int:
    seen: dict[str, dict] = {}
    headers = {"User-Agent": "ChestRAG corpus discovery (research/portfolio use)"}

    with httpx.Client(headers=headers) as client:
        for query in QUERIES:
            try:
                works = search(client, query)
            except httpx.HTTPError as exc:
                print(f"  ! query failed: {query!r} ({exc})")
                continue

            kept = 0
            for work in works:
                wid = work["id"].rsplit("/", 1)[-1]
                if wid in seen:
                    continue
                pdf = pdf_url_for(work)
                if not pdf:
                    continue
                abstract = invert_abstract(work.get("abstract_inverted_index"))
                if not is_relevant(work.get("title") or "", abstract):
                    continue
                seen[wid] = to_record(work, pdf)
                kept += 1
            print(f"  {query:<52} +{kept:>3} kept (total {len(seen)})")
            time.sleep(0.2)  # be polite to the API
            if len(seen) >= TARGET:
                break

    records = list(seen.values())[:TARGET]
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Quick breakdown of where the PDFs live (arxiv/pmc/publisher).
    by_source: dict[str, int] = {}
    for rec in records:
        by_source[rec["source"]] = by_source.get(rec["source"], 0) + 1
    top_sources = sorted(by_source.items(), key=lambda kv: -kv[1])[:8]

    print(f"\nWrote {len(records)} papers -> {OUT_FILE}")
    print("Top PDF sources: " + ", ".join(f"{host} ({n})" for host, n in top_sources))
    return 0


if __name__ == "__main__":
    sys.exit(main())
