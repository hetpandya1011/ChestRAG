"""Tests for the agent layer that don't require network/LLM calls.

We test the pure, deterministic parts: the tool schemas the LLM sees (the routing
contract) and the license-parsing helper. The loop itself and live HF calls are
exercised by the standalone scripts, not in CI (they need API keys + network).
"""
from __future__ import annotations

from services.api.agent import TOOLS
from services.api.tools.hf_lookup import HF_LOOKUP_SCHEMA, _license_from_tags


def test_agent_exposes_both_tools():
    names = {t["function"]["name"] for t in TOOLS}
    assert names == {"search_corpus", "hf_model_lookup"}


def test_tool_schemas_are_well_formed():
    for tool in TOOLS:
        fn = tool["function"]
        assert tool["type"] == "function"
        assert fn["name"] and fn["description"]  # description is the routing signal
        assert fn["parameters"]["required"]  # each tool requires at least one arg


def test_hf_schema_requires_model_name():
    params = HF_LOOKUP_SCHEMA["function"]["parameters"]
    assert params["required"] == ["model_name"]
    assert "model_name" in params["properties"]


def test_license_parsed_from_tags():
    assert _license_from_tags(["transformers", "license:mit", "region:us"]) == "mit"
    assert _license_from_tags(["transformers", "region:us"]) is None
    assert _license_from_tags([]) is None
