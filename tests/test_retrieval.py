"""Tests for the retrieval -> answer contract: citation selection + source formatting.

These are pure-logic tests (no database or API calls), so they run in CI.
"""
from __future__ import annotations

from llama_index.core.schema import NodeWithScore, TextNode

from services.api.generate import format_sources
from services.api.main import extract_cited


def test_extract_cited_orders_and_dedupes():
    assert extract_cited("Foo [2] bar [1] baz [2].", n_nodes=5) == [2, 1]


def test_extract_cited_drops_out_of_range():
    assert extract_cited("See [1] and [9].", n_nodes=3) == [1]


def test_extract_cited_handles_no_citations():
    assert extract_cited("No citations here.", n_nodes=5) == []


def _node(title: str, page: int, text: str) -> NodeWithScore:
    node = TextNode(text=text, metadata={"title": title, "page": page, "paper_id": "x"})
    return NodeWithScore(node=node)


def test_format_sources_numbers_and_labels():
    nodes = [_node("Paper A", 3, "alpha text"), _node("Paper B", 7, "beta text")]
    out = format_sources(nodes)
    assert "[1] Paper A (p.3)" in out
    assert "[2] Paper B (p.7)" in out
    assert "alpha text" in out
