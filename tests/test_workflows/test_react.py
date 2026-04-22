"""Tests for the ReAct workflow — new system_prompt and allow_shell parameters."""

from ghostgrid.config import CODE_AGENT_SYSTEM_PROMPT
from ghostgrid.models import Agent, AgentResult
from ghostgrid.workflows.react import run_react


def _agent() -> Agent:
    return Agent(
        model="test-model",
        endpoint="http://localhost/v1/chat/completions",
        api_key="EMPTY",
        provider="openai",
    )


def _make_result(agent: Agent, content: str) -> AgentResult:
    return AgentResult(
        agent_id=agent.agent_id,
        model=agent.model,
        provider=agent.provider,
        content=content,
        raw_response={},
        latency_ms=10.0,
    )


# ---------------------------------------------------------------------------
# system_prompt parameter
# ---------------------------------------------------------------------------


def test_run_react_uses_default_system_prompt(monkeypatch):
    """When system_prompt is not provided, REACT_SYSTEM_PROMPT is used."""
    agent = _agent()
    captured = []

    def fake_run_agent(ag, conversation, *args, **kwargs):
        captured.append(conversation)
        return _make_result(ag, "Thought: done\nFinal Answer: default")

    monkeypatch.setattr("ghostgrid.workflows.react.run_agent", fake_run_agent)

    result = run_react(
        agent,
        prompt="describe image",
        image_paths=[],
        detail="low",
        max_tokens=256,
        resize=False,
        target_size=(512, 512),
        enabled_tools=["describe"],
    )

    assert result["stop_reason"] == "final_answer"
    # The conversation passed to run_agent should contain text from REACT_SYSTEM_PROMPT
    assert "vision analysis agent" in captured[0]


def test_run_react_uses_custom_system_prompt(monkeypatch):
    """When system_prompt is provided it replaces the default."""
    agent = _agent()
    captured = []

    def fake_run_agent(ag, conversation, *args, **kwargs):
        captured.append(conversation)
        return _make_result(ag, "Thought: done\nFinal Answer: custom")

    monkeypatch.setattr("ghostgrid.workflows.react.run_agent", fake_run_agent)

    custom = "You are a CUSTOM agent.\n\nAvailable tools:\n{tool_descriptions}"
    run_react(
        agent,
        prompt="do something",
        image_paths=[],
        detail="low",
        max_tokens=256,
        resize=False,
        target_size=(512, 512),
        enabled_tools=["list_directory"],
        system_prompt=custom,
    )

    assert "CUSTOM agent" in captured[0]
    assert "vision analysis agent" not in captured[0]


def test_run_react_code_agent_prompt(monkeypatch):
    """Passing CODE_AGENT_SYSTEM_PROMPT produces a coding-focused conversation."""
    agent = _agent()
    captured = []

    def fake_run_agent(ag, conversation, *args, **kwargs):
        captured.append(conversation)
        return _make_result(ag, "Thought: done\nFinal Answer: code answer")

    monkeypatch.setattr("ghostgrid.workflows.react.run_agent", fake_run_agent)

    run_react(
        agent,
        prompt="fix the bug",
        image_paths=[],
        detail="low",
        max_tokens=256,
        resize=False,
        target_size=(512, 512),
        enabled_tools=["read_file"],
        system_prompt=CODE_AGENT_SYSTEM_PROMPT,
    )

    assert "coding agent" in captured[0]


# ---------------------------------------------------------------------------
# allow_shell parameter
# ---------------------------------------------------------------------------


def test_run_react_allow_shell_false_blocks_run_bash(monkeypatch):
    """Without allow_shell the run_bash tool should return the blocked message."""
    agent = _agent()
    call_count = iter(range(100))

    def fake_run_agent(ag, conversation, *args, **kwargs):
        n = next(call_count)
        if n == 0:
            # First step: agent tries to call run_bash
            return _make_result(ag, 'Thought: run shell\nAction: run_bash\nAction Input: {"command": "echo hi"}')
        # Second step: agent concludes
        return _make_result(ag, "Thought: done\nFinal Answer: shell blocked")

    monkeypatch.setattr("ghostgrid.workflows.react.run_agent", fake_run_agent)

    result = run_react(
        agent,
        prompt="run bash",
        image_paths=[],
        detail="low",
        max_tokens=256,
        resize=False,
        target_size=(512, 512),
        enabled_tools=["run_bash"],
        allow_shell=False,
    )

    # The observation from step 1 should contain the blocked message
    step1_obs = result["steps"][0]["observation"]
    assert "--allow-shell" in step1_obs


def test_run_react_allow_shell_true_executes_run_bash(monkeypatch):
    """With allow_shell=True the run_bash tool should actually execute."""
    agent = _agent()
    call_count = iter(range(100))

    def fake_run_agent(ag, conversation, *args, **kwargs):
        n = next(call_count)
        if n == 0:
            return _make_result(
                ag, 'Thought: run shell\nAction: run_bash\nAction Input: {"command": "echo hello-test"}'
            )
        return _make_result(ag, "Thought: done\nFinal Answer: ran")

    monkeypatch.setattr("ghostgrid.workflows.react.run_agent", fake_run_agent)

    result = run_react(
        agent,
        prompt="run bash",
        image_paths=[],
        detail="low",
        max_tokens=256,
        resize=False,
        target_size=(512, 512),
        enabled_tools=["run_bash"],
        allow_shell=True,
    )

    step1_obs = result["steps"][0]["observation"]
    assert "hello-test" in step1_obs


# ---------------------------------------------------------------------------
# Backward compatibility: old callers without new params still work
# ---------------------------------------------------------------------------


def test_run_react_backward_compatible_no_new_params(monkeypatch):
    """run_react works without system_prompt / allow_shell (backward compat)."""
    agent = _agent()

    def fake_run_agent(ag, conversation, *args, **kwargs):
        return _make_result(ag, "Thought: done\nFinal Answer: ok")

    monkeypatch.setattr("ghostgrid.workflows.react.run_agent", fake_run_agent)

    result = run_react(
        agent,
        prompt="hello",
        image_paths=[],
        detail="low",
        max_tokens=256,
        resize=False,
        target_size=(512, 512),
        enabled_tools=["describe"],
    )

    assert result["content"] == "ok"
    assert result["workflow"] == "react"
