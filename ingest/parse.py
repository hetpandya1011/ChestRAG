"""Parse the seed PDFs in data/raw/ into per-paper JSON in data/parsed/.

For each PDF we extract text page-by-page with PyMuPDF and attach the metadata
from ingest/seed_papers.yaml (title, authors, year, category, source). Keeping
text page-by-page preserves page numbers, which become citation metadata later.

Usage (from repo root, inside the venv):
    python ingest/parse.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pymupdf  # PyMuPDF
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_FILE = REPO_ROOT / "ingest" / "seed_papers.yaml"
RAW_DIR = REPO_ROOT / "data" / "raw"
PARSED_DIR = REPO_ROOT / "data" / "parsed"

# Below this many characters a PDF is probably image-only/scanned (would need
# OCR) — worth flagging rather than silently emitting an empty paper.
MIN_CHARS = 1000


def load_metadata() -> dict[str, dict]:
    with SEED_FILE.open() as f:
        papers = yaml.safe_load(f)["papers"]
    return {p["id"]: p for p in papers}


def parse_pdf(path: Path) -> tuple[list[dict], int]:
    """Return (pages, total_chars); pages = [{'page': int, 'text': str}, ...]."""
    pages: list[dict] = []
    total_chars = 0
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            pages.append({"page": i, "text": text})
            total_chars += len(text)
    return pages, total_chars


def main() -> int:
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    meta_by_id = load_metadata()

    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {RAW_DIR}. Run `python ingest/fetch.py` first.")
        return 1

    print(f"Parsing {len(pdfs)} PDFs -> {PARSED_DIR}\n")
    parsed = 0
    warnings: list[str] = []

    for pdf in pdfs:
        paper_id = pdf.stem
        meta = meta_by_id.get(paper_id)
        if meta is None:
            warnings.append(f"{paper_id} (no metadata match in seed_papers.yaml)")
            meta = {}

        try:
            pages, total_chars = parse_pdf(pdf)
        except Exception as exc:
            print(f"  ✗ {paper_id:<24} parse error: {exc}")
            warnings.append(f"{paper_id} (parse error)")
            continue

        record = {
            "id": paper_id,
            "title": meta.get("title"),
            "authors": meta.get("authors"),
            "year": meta.get("year"),
            "category": meta.get("category"),
            "source": meta.get("source"),
            "arxiv_id": meta.get("arxiv_id"),
            "n_pages": len(pages),
            "pages": pages,
        }
        (PARSED_DIR / f"{paper_id}.json").write_text(json.dumps(record, ensure_ascii=False))
        parsed += 1

        flag = "  ⚠ low text (image-only?)" if total_chars < MIN_CHARS else ""
        print(f"  ✓ {paper_id:<24} {len(pages):>3} pages, {total_chars:>7} chars{flag}")
        if total_chars < MIN_CHARS:
            warnings.append(f"{paper_id} (low text)")

    print(f"\nParsed {parsed}/{len(pdfs)} PDFs.")
    if warnings:
        print("Review: " + ", ".join(warnings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
