"""Chunk parsed papers into retrieval-sized passages, written to data/chunks.jsonl.

Uses LlamaIndex's SentenceSplitter (sentence-aware, token-counted). We build one
Document per page so each resulting chunk keeps its page number -> exact citations.

Usage (from repo root, inside the venv):
    python ingest/chunk.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

REPO_ROOT = Path(__file__).resolve().parent.parent
PARSED_DIR = REPO_ROOT / "data" / "parsed"
CHUNKS_FILE = REPO_ROOT / "data" / "chunks.jsonl"

CHUNK_SIZE = 512      # tokens per chunk
CHUNK_OVERLAP = 64    # tokens shared between adjacent chunks


def documents_for_paper(record: dict) -> list[Document]:
    """One Document per non-empty page, carrying citation metadata."""
    base_meta = {
        "paper_id": record["id"],
        "title": record.get("title"),
        "authors": record.get("authors"),
        "year": record.get("year"),
        "category": record.get("category"),
    }
    docs = []
    for page in record["pages"]:
        text = page["text"].strip()
        if not text:
            continue
        docs.append(Document(text=text, metadata={**base_meta, "page": page["page"]}))
    return docs


def main() -> int:
    parsed_files = sorted(PARSED_DIR.glob("*.json"))
    if not parsed_files:
        print(f"No parsed papers in {PARSED_DIR}. Run `python ingest/parse.py` first.")
        return 1

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    total = 0
    print(f"Chunking {len(parsed_files)} papers -> {CHUNKS_FILE}\n")
    with CHUNKS_FILE.open("w", encoding="utf-8") as out:
        for pf in parsed_files:
            record = json.loads(pf.read_text())
            nodes = splitter.get_nodes_from_documents(documents_for_paper(record))
            for i, node in enumerate(nodes):
                chunk = {
                    "chunk_id": f"{record['id']}-{i:04d}",
                    "paper_id": record["id"],
                    "title": record.get("title"),
                    "page": node.metadata.get("page"),
                    "text": node.get_content(),
                }
                out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            total += len(nodes)
            print(f"  {record['id']:<24} {len(nodes):>4} chunks")

    print(f"\nWrote {total} chunks across {len(parsed_files)} papers -> {CHUNKS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
