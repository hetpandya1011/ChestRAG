"""Tests for the chunking step (ingest/chunk.py)."""
from __future__ import annotations

from llama_index.core.node_parser import SentenceSplitter

from ingest.chunk import documents_for_paper

SAMPLE = {
    "id": "demo",
    "title": "Demo Paper",
    "authors": "Doe et al.",
    "year": 2024,
    "category": "dataset",
    "pages": [
        {"page": 1, "text": "First page, sentence one. First page, sentence two."},
        {"page": 2, "text": ""},  # empty page -> should be skipped
        {"page": 3, "text": "Third page content about chest x-rays and CNNs."},
    ],
}


def test_documents_skip_empty_pages_and_tag_page():
    docs = documents_for_paper(SAMPLE)
    assert len(docs) == 2  # page 2 (empty) is dropped
    assert sorted(d.metadata["page"] for d in docs) == [1, 3]


def test_documents_carry_paper_metadata():
    meta = documents_for_paper(SAMPLE)[0].metadata
    assert meta["paper_id"] == "demo"
    assert meta["title"] == "Demo Paper"
    assert meta["category"] == "dataset"


def test_chunks_preserve_page_metadata():
    docs = documents_for_paper(SAMPLE)
    nodes = SentenceSplitter(chunk_size=512, chunk_overlap=64).get_nodes_from_documents(docs)
    assert nodes
    assert all("page" in n.metadata for n in nodes)
