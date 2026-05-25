"""Section-aware chunking: parsed papers -> data/chunks.jsonl.

Best-effort section detection (no GROBID): we scan for common heading lines
(Introduction / Methods / Results / Discussion / References / ...) and tag every
chunk with the section it falls in. Reference lists and boilerplate
(acknowledgments, funding, conflicts) are DROPPED -- they're noise for retrieval.
We split within each (page, section) run with LlamaIndex's SentenceSplitter, so
each chunk keeps an exact page number AND a section label for citations.

NOTE: heading detection is heuristic; papers format headings inconsistently, so
some sections will be mislabeled or missed. The printed section breakdown (and the
eval harness) tell us how well it worked.

Usage (from repo root, inside the venv):
    python ingest/chunk.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

REPO_ROOT = Path(__file__).resolve().parent.parent
PARSED_DIR = REPO_ROOT / "data" / "parsed"
CHUNKS_FILE = REPO_ROOT / "data" / "chunks.jsonl"

CHUNK_SIZE = 512      # tokens per chunk
CHUNK_OVERLAP = 64    # tokens shared between adjacent chunks

# Heading keyword -> canonical section. "drop" marks noise we exclude from the index.
SECTION_KEYWORDS = {
    "abstract": "abstract",
    "introduction": "introduction", "background": "introduction",
    "related work": "related_work",
    "method": "methods", "methods": "methods", "methodology": "methods",
    "materials and methods": "methods", "materials": "methods", "approach": "methods",
    "experiments": "experiments", "experimental setup": "experiments",
    "results": "results", "evaluation": "results", "results and discussion": "results",
    "discussion": "discussion", "limitations": "discussion",
    "conclusion": "conclusion", "conclusions": "conclusion",
    "references": "drop", "bibliography": "drop",
    "acknowledgments": "drop", "acknowledgements": "drop",
    "funding": "drop", "author contributions": "drop",
    "conflict of interest": "drop", "conflicts of interest": "drop",
    "data availability": "drop", "declaration of competing interest": "drop",
}
DROP = {"drop"}

# A heading line: short, optionally numbered ("3.", "3.1", "II."), then heading words.
HEADING_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?|[IVXLC]+\.)?\s*([A-Za-z][A-Za-z &/-]{2,40})\s*$")


def detect_section(line: str) -> str | None:
    s = line.strip()
    if not s or len(s) > 45:
        return None
    m = HEADING_RE.match(s)
    if not m:
        return None
    return SECTION_KEYWORDS.get(m.group(1).strip().lower())


def page_runs(page_text: str, current: str) -> tuple[list[tuple[str, str]], str]:
    """Split one page into (section, text) runs, carrying the section state forward."""
    runs: list[tuple[str, str]] = []
    buf: list[str] = []
    section = current
    for line in page_text.split("\n"):
        detected = detect_section(line)
        if detected:
            if buf:
                runs.append((section, "\n".join(buf)))
                buf = []
            section = detected
        else:
            buf.append(line)
    if buf:
        runs.append((section, "\n".join(buf)))
    return runs, section


def documents_for_paper(record: dict) -> list[Document]:
    """One Document per (page, section) run, carrying citation metadata; drops noise."""
    base = {
        "paper_id": record["id"],
        "title": record.get("title"),
        "category": record.get("category"),
        "year": record.get("year"),
    }
    docs: list[Document] = []
    current = "front"  # title/authors/abstract area before the first detected heading
    for page in record["pages"]:
        text = page.get("text", "")
        if not text.strip():
            continue
        runs, current = page_runs(text, current)
        for section, run_text in runs:
            if section in DROP or not run_text.strip():
                continue
            docs.append(
                Document(text=run_text, metadata={**base, "page": page["page"], "section": section})
            )
    return docs


def main() -> int:
    parsed_files = sorted(PARSED_DIR.glob("*.json"))
    if not parsed_files:
        print(f"No parsed papers in {PARSED_DIR}. Run `python ingest/parse.py` first.")
        return 1

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    total = 0
    by_section: dict[str, int] = {}

    print(f"Chunking {len(parsed_files)} papers -> {CHUNKS_FILE}\n")
    with CHUNKS_FILE.open("w", encoding="utf-8") as out:
        for pf in parsed_files:
            record = json.loads(pf.read_text())
            nodes = splitter.get_nodes_from_documents(documents_for_paper(record))
            for i, node in enumerate(nodes):
                section = node.metadata.get("section", "other")
                by_section[section] = by_section.get(section, 0) + 1
                chunk = {
                    "chunk_id": f"{record['id']}-{i:04d}",
                    "paper_id": record["id"],
                    "title": record.get("title"),
                    "page": node.metadata.get("page"),
                    "section": section,
                    "text": node.get_content(),
                }
                out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            total += len(nodes)

    print(f"Wrote {total} chunks across {len(parsed_files)} papers -> {CHUNKS_FILE}\n")
    print("Chunks by section: " + ", ".join(
        f"{s} ({n})" for s, n in sorted(by_section.items(), key=lambda kv: -kv[1])
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
