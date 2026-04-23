"""Tests for configuration module."""

import os

import pytest

from ghostgrid.config import (
    CODE_AGENT_SYSTEM_PROMPT,
    CODE_AGENT_TOOLS,
    DEFAULT_ENDPOINT,
    PROVIDER_ENV_MAP,
    WORKFLOW_CHOICES,
    get_api_key,
    get_default_endpoint,
    resolve_endpoint,
)


def test_provider_env_map():
    """Test PROVIDER_ENV_MAP contains expected providers."""
    assert "openai" in PROVIDER_ENV_MAP
    assert "anthropic" in PROVIDER_ENV_MAP
    assert "together" in PROVIDER_ENV_MAP
    assert "zai" in PROVIDER_ENV_MAP
    assert PROVIDER_ENV_MAP["openai"] == "OPENAI_API_KEY"


def test_workflow_choices():
    """Test WORKFLOW_CHOICES contains all workflows."""
    expected = ["sequential", "parallel", "conditional", "iterative", "moa", "react", "monitor"]
    for workflow in expected:
        assert workflow in WORKFLOW_CHOICES


def test_default_endpoint():
    """Test DEFAULT_ENDPOINT is OpenAI."""
    assert "openai.com" in DEFAULT_ENDPOINT


def test_get_default_endpoint():
    """Test get_default_endpoint returns correct endpoints."""
    assert "openai.com" in get_default_endpoint("openai")
    assert "anthropic.com" in get_default_endpoint("anthropic")
    assert "generativelanguage.googleapis.com" in get_default_endpoint("google")
    assert "together.xyz" in get_default_endpoint("together")
    assert "api.z.ai" in get_default_endpoint("zai")
    # Unknown provider falls back to default
    assert get_default_endpoint("unknown") == DEFAULT_ENDPOINT


def test_resolve_endpoint_requires_explicit_for_azure():
    """Known providers without a shared default endpoint must be explicit."""
    with pytest.raises(RuntimeError, match="requires an explicit --endpoint"):
        resolve_endpoint("azure")


def test_resolve_endpoint_uses_provider_default():
    """Known providers with OpenAI-compatible endpoints resolve automatically."""
    assert "anthropic.com" in resolve_endpoint("anthropic")
    assert "generativelanguage.googleapis.com" in resolve_endpoint("google")
    assert "api.z.ai" in resolve_endpoint("zai")


def test_get_api_key_missing():
    """Test get_api_key raises when key is missing."""
    # Temporarily unset the key if it exists
    original = os.environ.pop("TEST_MISSING_API_KEY", None)
    try:
        with pytest.raises(RuntimeError):
            get_api_key("openai")  # Assuming OPENAI_API_KEY is not set in test env
    except RuntimeError:
        pass  # Expected if key is actually set
    finally:
        if original:
            os.environ["TEST_MISSING_API_KEY"] = original


# ---------------------------------------------------------------------------
# Code-agent config additions
# ---------------------------------------------------------------------------


def test_code_agent_tools_contains_expected_tools():
    """CODE_AGENT_TOOLS must include all filesystem/shell tools."""
    expected = {"read_file", "write_file", "list_directory", "run_bash", "search_files"}
    assert expected.issubset(set(CODE_AGENT_TOOLS))


def test_code_agent_system_prompt_has_placeholder():
    """CODE_AGENT_SYSTEM_PROMPT must contain {tool_descriptions} for formatting."""
    assert "{tool_descriptions}" in CODE_AGENT_SYSTEM_PROMPT


def test_code_agent_system_prompt_is_distinct_from_react_prompt():
    """The code-agent prompt should be different from the vision ReAct prompt."""
    from ghostgrid.config import REACT_SYSTEM_PROMPT

    assert CODE_AGENT_SYSTEM_PROMPT != REACT_SYSTEM_PROMPT


def test_code_agent_system_prompt_mentions_coding():
    """The code-agent prompt should communicate its coding purpose."""
    assert "coding" in CODE_AGENT_SYSTEM_PROMPT.lower() or "code" in CODE_AGENT_SYSTEM_PROMPT.lower()
