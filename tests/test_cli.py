"""Tests for CLI helpers."""

import json
import sys

import pytest

from ghostgrid import cli
from ghostgrid.cli import build_agents, main


def test_build_agents_uses_provider_specific_default_endpoints(monkeypatch):
    """Each provider should resolve to its own default endpoint."""
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test")

    agents = build_agents(
        models=["gpt-5.2", "gemini-2.5-flash"],
        providers=["openai", "google"],
        endpoints=[],
    )

    assert agents[0].endpoint == "https://api.openai.com/v1/chat/completions"
    assert agents[1].endpoint == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


def test_build_agents_rejects_azure_without_endpoint(monkeypatch):
    """Azure deployments need an explicit resource-specific endpoint."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-test")

    with pytest.raises(RuntimeError, match="requires an explicit --endpoint"):
        build_agents(models=["gpt-5.2"], providers=["azure"], endpoints=[])


# ---------------------------------------------------------------------------
# CLI flag tests: --code-agent, --allow-shell, optional --images
# ---------------------------------------------------------------------------


def test_cli_images_optional_for_react(monkeypatch, capsys):
    """--images should not be required when running react workflow."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_run_react(*args, **kwargs):
        return {
            "workflow": "react",
            "content": "ok",
            "steps": [],
            "total_steps": 0,
            "stop_reason": "final_answer",
            "model": "m",
            "provider": "openai",
        }

    monkeypatch.setattr("ghostgrid.cli.run_react", fake_run_react)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ghostgrid",
            "run",
            "--workflow",
            "react",
            "--model",
            "gpt-4o",
            "--prompt",
            "list files",
            "--tools",
            "list_directory",
        ],
    )
    main()
    out = capsys.readouterr().out
    assert "react" in out


def test_cli_code_agent_flag_selects_code_tools(monkeypatch, capsys):
    """--code-agent should activate CODE_AGENT_TOOLS and CODE_AGENT_SYSTEM_PROMPT."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    captured = {}

    def fake_run_react(*args, **kwargs):
        captured["enabled_tools"] = kwargs.get("enabled_tools")
        captured["system_prompt"] = kwargs.get("system_prompt")
        captured["allow_shell"] = kwargs.get("allow_shell")
        return {
            "workflow": "react",
            "content": "ok",
            "steps": [],
            "total_steps": 0,
            "stop_reason": "final_answer",
            "model": "m",
            "provider": "openai",
        }

    monkeypatch.setattr("ghostgrid.cli.run_react", fake_run_react)

    from ghostgrid.config import CODE_AGENT_SYSTEM_PROMPT, CODE_AGENT_TOOLS

    monkeypatch.setattr(
        sys,
        "argv",
        ["ghostgrid", "run", "--workflow", "react", "--model", "gpt-4o", "--prompt", "fix bug", "--code-agent"],
    )
    main()

    assert set(captured["enabled_tools"]) == set(CODE_AGENT_TOOLS)
    assert captured["system_prompt"] == CODE_AGENT_SYSTEM_PROMPT
    assert captured["allow_shell"] is False


def test_cli_allow_shell_flag(monkeypatch, capsys):
    """--allow-shell should pass allow_shell=True to run_react."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    captured = {}

    def fake_run_react(*args, **kwargs):
        captured["allow_shell"] = kwargs.get("allow_shell")
        return {
            "workflow": "react",
            "content": "ok",
            "steps": [],
            "total_steps": 0,
            "stop_reason": "final_answer",
            "model": "m",
            "provider": "openai",
        }

    monkeypatch.setattr("ghostgrid.cli.run_react", fake_run_react)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ghostgrid",
            "run",
            "--workflow",
            "react",
            "--model",
            "gpt-4o",
            "--prompt",
            "run build",
            "--code-agent",
            "--allow-shell",
        ],
    )
    main()

    assert captured["allow_shell"] is True


def test_cli_explicit_tools_override_code_agent_defaults(monkeypatch, capsys):
    """When --tools is set alongside --code-agent, explicit tools take priority."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    captured = {}

    def fake_run_react(*args, **kwargs):
        captured["enabled_tools"] = kwargs.get("enabled_tools")
        return {
            "workflow": "react",
            "content": "ok",
            "steps": [],
            "total_steps": 0,
            "stop_reason": "final_answer",
            "model": "m",
            "provider": "openai",
        }

    monkeypatch.setattr("ghostgrid.cli.run_react", fake_run_react)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ghostgrid",
            "run",
            "--workflow",
            "react",
            "--model",
            "gpt-4o",
            "--prompt",
            "read only",
            "--code-agent",
            "--tools",
            "read_file",
        ],
    )
    main()

    assert captured["enabled_tools"] == ["read_file"]


def test_main_allows_text_only_run(monkeypatch, capsys):
    """The run command should work without --images for text-only LLM calls."""
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test")
    monkeypatch.setattr(
        cli,
        "WORKFLOW_REGISTRY",
        {
            "sequential": lambda agents, **kwargs: {
                "workflow": "sequential",
                "content": "ok",
                "image_paths": kwargs["image_paths"],
            }
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["ghostgrid", "run", "--prompt", "Hello", "--model", "gpt-5.2", "--workflow", "sequential"],
    )

    cli.main()

    output = json.loads(capsys.readouterr().out)
    assert output["content"] == "ok"
    assert output["image_paths"] == []
