"""Download the seed-corpus PDFs listed in ingest/seed_papers.yaml into data/raw/.

Idempotent: skips PDFs already present. Validates that each download is a real
PDF (some hosts return an HTML "blocked" page with HTTP 200), and prints a clear
success/failure summary so any paywalled/blocked paper can be grabbed by hand.

Usage (from repo root, inside the venv):
    python ingest/fetch.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx
import yaml

# Resolve paths from this file's location, so the script works no matter where
# it's launched from.
REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_FILE = REPO_ROOT / "ingest" / "seed_papers.yaml"
RAW_DIR = REPO_ROOT / "data" / "raw"

# A polite, browser-like User-Agent. arXiv / PMC / CVF reject empty or bot-like agents.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (ChestRAG seed fetcher; research/portfolio use)"
}


def load_papers() -> list[dict]:
    with SEED_FILE.open() as f:
        return yaml.safe_load(f)["papers"]


def looks_like_pdf(content: bytes) -> bool:
    # Real PDFs start with the magic bytes "%PDF"; HTML block pages won't.
    return content[:4] == b"%PDF"


def download_one(client: httpx.Client, paper: dict) -> tuple[bool, str]:
    dest = RAW_DIR / f"{paper['id']}.pdf"
    if dest.exists() and dest.stat().st_size > 0:
        return True, "skipped (already present)"

    try:
        resp = client.get(paper["pdf_url"])
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        return False, f"download error: {exc}"

    if not looks_like_pdf(resp.content):
        return False, "not a PDF (likely blocked/HTML) — download manually"

    dest.write_bytes(resp.content)
    return True, f"ok ({len(resp.content) // 1024} KB)"


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    papers = load_papers()
    print(f"Fetching {len(papers)} seed papers -> {RAW_DIR}\n")

    failed = []
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=60.0) as client:
        for paper in papers:
            ok, msg = download_one(client, paper)
            print(f"  {'✓' if ok else '✗'} {paper['id']:<24} {msg}")
            if not ok:
                failed.append(paper["id"])

    print(f"\nDone: {len(papers) - len(failed)}/{len(papers)} available.")
    if failed:
        print("Manual download needed for: " + ", ".join(failed))
        print("  -> save each as data/raw/<id>.pdf, then re-run to confirm.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
