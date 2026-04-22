"""Tests for built-in ReAct tools (filesystem, shell, and vision stubs)."""

from ghostgrid.models import Agent
from ghostgrid.tools import BUILTIN_TOOLS
from ghostgrid.tools.builtin import (
    _tool_list_directory,
    _tool_read_file,
    _tool_run_bash,
    _tool_search_files,
    _tool_write_file,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent() -> Agent:
    return Agent(
        model="test-model",
        endpoint="http://localhost/v1/chat/completions",
        api_key="EMPTY",
        provider="openai",
    )


_COMMON = {"image_paths": [], "detail": "low", "max_tokens": 256, "resize": False, "target_size": (512, 512)}


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


def test_builtin_tools_contains_filesystem_tools():
    for name in ("read_file", "write_file", "list_directory", "run_bash", "search_files"):
        assert name in BUILTIN_TOOLS, f"'{name}' missing from BUILTIN_TOOLS"


def test_builtin_tools_contains_vision_tools():
    for name in ("describe", "detect_objects", "read_text", "analyze_region", "count_objects"):
        assert name in BUILTIN_TOOLS, f"'{name}' missing from BUILTIN_TOOLS"


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


def test_read_file_returns_content(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("hello world")
    result = _tool_read_file(_agent(), **_COMMON, path=str(f))
    assert result == "hello world"


def test_read_file_missing_path_param():
    result = _tool_read_file(_agent(), **_COMMON)
    assert result.startswith("ERROR:")


def test_read_file_nonexistent_file():
    result = _tool_read_file(_agent(), **_COMMON, path="/no/such/file.txt")
    assert result.startswith("ERROR:")


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


def test_write_file_creates_file(tmp_path):
    dest = tmp_path / "out.txt"
    result = _tool_write_file(_agent(), **_COMMON, path=str(dest), content="new content")
    assert "Written" in result
    assert dest.read_text() == "new content"


def test_write_file_creates_parent_directories(tmp_path):
    dest = tmp_path / "a" / "b" / "c.txt"
    _tool_write_file(_agent(), **_COMMON, path=str(dest), content="deep")
    assert dest.read_text() == "deep"


def test_write_file_overwrites_existing(tmp_path):
    dest = tmp_path / "f.txt"
    dest.write_text("old")
    _tool_write_file(_agent(), **_COMMON, path=str(dest), content="new")
    assert dest.read_text() == "new"


def test_write_file_missing_path_param():
    result = _tool_write_file(_agent(), **_COMMON)
    assert result.startswith("ERROR:")


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------


def test_list_directory_shows_files_and_dirs(tmp_path):
    (tmp_path / "file.txt").write_text("x")
    (tmp_path / "subdir").mkdir()
    result = _tool_list_directory(_agent(), **_COMMON, path=str(tmp_path))
    assert "file.txt" in result
    assert "subdir/" in result


def test_list_directory_empty_dir(tmp_path):
    result = _tool_list_directory(_agent(), **_COMMON, path=str(tmp_path))
    assert result == "(empty directory)"


def test_list_directory_nonexistent_path():
    result = _tool_list_directory(_agent(), **_COMMON, path="/no/such/dir")
    assert result.startswith("ERROR:")


def test_list_directory_defaults_to_cwd():
    result = _tool_list_directory(_agent(), **_COMMON)
    # Just verify it returns something without crashing
    assert isinstance(result, str)
    assert not result.startswith("ERROR:")


# ---------------------------------------------------------------------------
# run_bash
# ---------------------------------------------------------------------------


def test_run_bash_blocked_without_allow_shell():
    result = _tool_run_bash(_agent(), **_COMMON, command="echo hi", allow_shell=False)
    assert "disabled" in result.lower() or "--allow-shell" in result


def test_run_bash_blocked_by_default():
    # allow_shell kwarg absent → defaults to False inside the tool
    result = _tool_run_bash(_agent(), **_COMMON, command="echo hi")
    assert "--allow-shell" in result


def test_run_bash_executes_with_allow_shell():
    result = _tool_run_bash(_agent(), **_COMMON, command="echo hello-world", allow_shell=True)
    assert "hello-world" in result


def test_run_bash_captures_stderr_with_allow_shell():
    result = _tool_run_bash(_agent(), **_COMMON, command="echo err >&2", allow_shell=True)
    assert "err" in result


def test_run_bash_missing_command_param():
    result = _tool_run_bash(_agent(), **_COMMON, allow_shell=True)
    assert result.startswith("ERROR:")


# ---------------------------------------------------------------------------
# search_files
# ---------------------------------------------------------------------------


def test_search_files_finds_pattern(tmp_path):
    (tmp_path / "a.txt").write_text("hello world\nfoo bar\n")
    (tmp_path / "b.txt").write_text("nothing here\n")
    result = _tool_search_files(_agent(), **_COMMON, pattern="hello", path=str(tmp_path))
    assert "hello" in result
    assert "a.txt" in result


def test_search_files_no_match(tmp_path):
    (tmp_path / "x.txt").write_text("irrelevant")
    result = _tool_search_files(_agent(), **_COMMON, pattern="zzznotfound", path=str(tmp_path))
    assert "No matches" in result


def test_search_files_missing_pattern_param():
    result = _tool_search_files(_agent(), **_COMMON, path=".")
    assert result.startswith("ERROR:")


def test_search_files_truncates_large_output(tmp_path, monkeypatch):
    """Output exceeding 50 lines should be truncated with a notice."""
    import subprocess

    fake_output = "\n".join(f"file.txt:{i}:match" for i in range(100))

    def fake_run(cmd, **kwargs):
        class R:
            stdout = fake_output
            stderr = ""

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = _tool_search_files(_agent(), **_COMMON, pattern="match", path=str(tmp_path))
    assert "truncated" in result
