"""Download the corpus PDFs into data/raw/.

Sources, merged and de-duplicated (by normalized title):
  - ingest/seed_papers.yaml  (the curated Weekend 1 seeds)
  - data/corpus.jsonl        (discovered via ingest/discover.py)

Idempotent: skips PDFs already on disk, so you can Ctrl+C and re-run to resume.
Validates each download is a real PDF (some hosts return an HTML/landing page with
HTTP 200) and prints a per-host failure breakdown -- many publisher PDFs (IEEE, some
Springer/OUP) block automated downloads, which is expected.

Usage (from repo root, inside the venv):
    python ingest/fetch.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_FILE = REPO_ROOT / "ingest" / "seed_papers.yaml"
CORPUS_FILE = REPO_ROOT / "data" / "corpus.jsonl"
RAW_DIR = REPO_ROOT / "data" / "raw"
FAILURES_FILE = REPO_ROOT / "data" / "fetch_failures.jsonl"

# A realistic browser User-Agent + Accept headers. Many open-access hosts (MDPI,
# Hindawi, PMC) sit behind Cloudflare/WAFs that reject non-browser-looking agents.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
TIMEOUT = 20.0


def _norm_title(title: str) -> str:
    return " ".join((title or "").lower().split())


def load_papers() -> list[dict]:
    """Merge curated seeds + discovered corpus; de-dupe by normalized title."""
    papers: list[dict] = []
    seen: set[str] = set()

    def add(paper_id: str, title: str, pdf_url: str) -> None:
        t = _norm_title(title)
        if t and t not in seen:
            seen.add(t)
            papers.append({"id": paper_id, "title": title, "pdf_url": pdf_url})

    # Seeds first (curated; their reliable arXiv/CVF/PMC links take priority).
    if SEED_FILE.exists():
        for p in yaml.safe_load(SEED_FILE.read_text())["papers"]:
            add(p["id"], p.get("title", ""), p["pdf_url"])

    # Then the discovered corpus.
    if CORPUS_FILE.exists():
        for line in CORPUS_FILE.read_text().splitlines():
            if line.strip():
                p = json.loads(line)
                add(p["id"], p.get("title", ""), p["pdf_url"])

    return papers


def looks_like_pdf(content: bytes) -> bool:
    return content[:4] == b"%PDF"


def download_one(client: httpx.Client, paper: dict) -> str:
    """Return one of "ok" | "skip" | "fail"."""
    dest = RAW_DIR / f"{paper['id']}.pdf"
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"
    try:
        resp = client.get(paper["pdf_url"])
        resp.raise_for_status()
    except httpx.HTTPError:
        return "fail"
    if not looks_like_pdf(resp.content):
        return "fail"
    dest.write_bytes(resp.content)
    return "ok"


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    papers = load_papers()
    if not papers:
        print("No papers to fetch. Run `python ingest/discover.py` first.")
        return 1

    print(f"Fetching {len(papers)} papers -> {RAW_DIR}\n")
    counts = {"ok": 0, "skip": 0, "fail": 0}
    fail_by_host: dict[str, int] = {}
    failures: list[dict] = []

    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=TIMEOUT) as client:
        for i, paper in enumerate(papers, start=1):
            status = download_one(client, paper)
            counts[status] += 1
            if status == "fail":
                host = urlparse(paper["pdf_url"]).netloc.replace("www.", "")
                fail_by_host[host] = fail_by_host.get(host, 0) + 1
                failures.append(paper)
            if i % 25 == 0 or i == len(papers):
                c = counts
                print(f"  [{i:>3}/{len(papers)}] ok={c['ok']} skip={c['skip']} fail={c['fail']}")

    if failures:
        FAILURES_FILE.write_text("\n".join(json.dumps(f, ensure_ascii=False) for f in failures))

    available = counts["ok"] + counts["skip"]
    print(
        f"\nCorpus: {available} PDFs in data/raw/ "
        f"({counts['ok']} new, {counts['skip']} present, {counts['fail']} failed)."
    )
    if fail_by_host:
        top = sorted(fail_by_host.items(), key=lambda kv: -kv[1])[:8]
        print("Failures by host: " + ", ".join(f"{h} ({n})" for h, n in top))
        print(f"Failed records saved to data/{FAILURES_FILE.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
