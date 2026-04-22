"""
Built-in ReAct tools for vision analysis and code/filesystem operations.
"""

import os
import subprocess

from ghostgrid.models import Agent, Tool
from ghostgrid.providers import run_agent

_SHELL_BLOCKED = "Shell execution is disabled. Re-run with --allow-shell to enable run_bash."


def _tool_describe(
    agent: Agent,
    image_paths: list[str],
    detail: str,
    max_tokens: int,
    resize: bool,
    target_size: tuple[int, int],
    **kwargs,
) -> str:
    """Ask the VLM to describe the image(s)."""
    prompt = kwargs.get("prompt", "Describe this image in detail.")
    result = run_agent(agent, prompt, image_paths, detail, max_tokens, resize, target_size)
    return result.content if result.success else f"ERROR: {result.error}"


def _tool_detect_objects(
    agent: Agent,
    image_paths: list[str],
    detail: str,
    max_tokens: int,
    resize: bool,
    target_size: tuple[int, int],
    **kwargs,
) -> str:
    """Ask the VLM to list all objects visible in the image(s)."""
    result = run_agent(
        agent,
        "List every distinct object you can see in this image. Return as a JSON array of strings.",
        image_paths,
        detail,
        max_tokens,
        resize,
        target_size,
    )
    return result.content if result.success else f"ERROR: {result.error}"


def _tool_read_text(
    agent: Agent,
    image_paths: list[str],
    detail: str,
    max_tokens: int,
    resize: bool,
    target_size: tuple[int, int],
    **kwargs,
) -> str:
    """Ask the VLM to extract all visible text (OCR) from the image(s)."""
    result = run_agent(
        agent,
        "Extract and return all text visible in this image, preserving the reading order.",
        image_paths,
        detail,
        max_tokens,
        resize,
        target_size,
    )
    return result.content if result.success else f"ERROR: {result.error}"


def _tool_analyze_region(
    agent: Agent,
    image_paths: list[str],
    detail: str,
    max_tokens: int,
    resize: bool,
    target_size: tuple[int, int],
    **kwargs,
) -> str:
    """Ask the VLM to focus analysis on a described region of the image."""
    region = kwargs.get("region", "the center of the image")
    question = kwargs.get("question", "What do you see?")
    result = run_agent(
        agent,
        f"Focus only on {region}. {question}",
        image_paths,
        detail,
        max_tokens,
        resize,
        target_size,
    )
    return result.content if result.success else f"ERROR: {result.error}"


def _tool_count_objects(
    agent: Agent,
    image_paths: list[str],
    detail: str,
    max_tokens: int,
    resize: bool,
    target_size: tuple[int, int],
    **kwargs,
) -> str:
    """Ask the VLM to count occurrences of a specific object."""
    object_name = kwargs.get("object", "objects")
    result = run_agent(
        agent,
        f"Count exactly how many '{object_name}' are visible in this image. Return only an integer.",
        image_paths,
        detail,
        max_tokens,
        resize,
        target_size,
    )
    return result.content if result.success else f"ERROR: {result.error}"


# ---------------------------------------------------------------------------
# File system and shell tools (code-agent mode)
# ---------------------------------------------------------------------------


def _tool_read_file(
    agent: Agent,
    image_paths: list[str],
    detail: str,
    max_tokens: int,
    resize: bool,
    target_size: tuple[int, int],
    **kwargs,
) -> str:
    """Read a file from the local filesystem and return its contents."""
    path = kwargs.get("path")
    if not path:
        return "ERROR: 'path' parameter is required."
    try:
        with open(path) as f:
            return f.read()
    except Exception as exc:
        return f"ERROR: {exc}"


def _tool_write_file(
    agent: Agent,
    image_paths: list[str],
    detail: str,
    max_tokens: int,
    resize: bool,
    target_size: tuple[int, int],
    **kwargs,
) -> str:
    """Write content to a file, creating parent directories as needed."""
    path = kwargs.get("path")
    content = kwargs.get("content", "")
    if not path:
        return "ERROR: 'path' parameter is required."
    try:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"Written {len(content)} bytes to {path}"
    except Exception as exc:
        return f"ERROR: {exc}"


def _tool_list_directory(
    agent: Agent,
    image_paths: list[str],
    detail: str,
    max_tokens: int,
    resize: bool,
    target_size: tuple[int, int],
    **kwargs,
) -> str:
    """List the contents of a directory."""
    path = kwargs.get("path", ".")
    try:
        entries = sorted(os.listdir(path))
        lines = []
        for entry in entries:
            tag = "/" if os.path.isdir(os.path.join(path, entry)) else ""
            lines.append(f"{entry}{tag}")
        return "\n".join(lines) if lines else "(empty directory)"
    except Exception as exc:
        return f"ERROR: {exc}"


def _tool_run_bash(
    agent: Agent,
    image_paths: list[str],
    detail: str,
    max_tokens: int,
    resize: bool,
    target_size: tuple[int, int],
    **kwargs,
) -> str:
    """Execute a shell command and return stdout + stderr. Requires --allow-shell."""
    if not kwargs.get("allow_shell", False):
        return _SHELL_BLOCKED
    command = kwargs.get("command")
    if not command:
        return "ERROR: 'command' parameter is required."
    try:
        proc = subprocess.run(
            command,
            shell=True,  # noqa: S602 — intentional; user must opt in via --allow-shell
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = proc.stdout
        if proc.stderr:
            output += f"\n[stderr]\n{proc.stderr}"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: Command timed out after 30 seconds."
    except Exception as exc:
        return f"ERROR: {exc}"


def _tool_search_files(
    agent: Agent,
    image_paths: list[str],
    detail: str,
    max_tokens: int,
    resize: bool,
    target_size: tuple[int, int],
    **kwargs,
) -> str:
    """Search for a text pattern across files in a directory (grep -rn)."""
    pattern = kwargs.get("pattern")
    path = kwargs.get("path", ".")
    if not pattern:
        return "ERROR: 'pattern' parameter is required."
    try:
        proc = subprocess.run(
            ["grep", "-r", "-n", "--include=*", pattern, path],
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = proc.stdout.strip()
        if not output:
            return f"No matches for '{pattern}' in {path}"
        # Cap output to avoid flooding the context
        lines = output.splitlines()
        if len(lines) > 50:
            return "\n".join(lines[:50]) + f"\n... ({len(lines) - 50} more lines truncated)"
        return output
    except subprocess.TimeoutExpired:
        return "ERROR: Search timed out."
    except Exception as exc:
        return f"ERROR: {exc}"


BUILTIN_TOOLS: dict[str, Tool] = {
    "describe": Tool(
        name="describe",
        description="Generate a detailed description of the image(s).",
        parameters='{"prompt": "optional focus instruction (string)"}',
        fn=_tool_describe,
    ),
    "detect_objects": Tool(
        name="detect_objects",
        description="List all distinct objects visible in the image(s).",
        parameters="{}",
        fn=_tool_detect_objects,
    ),
    "read_text": Tool(
        name="read_text",
        description="Extract all visible text from the image(s) (OCR).",
        parameters="{}",
        fn=_tool_read_text,
    ),
    "analyze_region": Tool(
        name="analyze_region",
        description="Focus analysis on a specific region of the image.",
        parameters='{"region": "description of the region, e.g. top-left corner", "question": "what to answer about that region"}',
        fn=_tool_analyze_region,
    ),
    "count_objects": Tool(
        name="count_objects",
        description="Count occurrences of a specific object in the image(s).",
        parameters='{"object": "name of the object to count"}',
        fn=_tool_count_objects,
    ),
    "read_file": Tool(
        name="read_file",
        description="Read a file from the local filesystem.",
        parameters='{"path": "file path to read"}',
        fn=_tool_read_file,
    ),
    "write_file": Tool(
        name="write_file",
        description="Write or overwrite a file on the local filesystem.",
        parameters='{"path": "file path to write", "content": "full file content as a string"}',
        fn=_tool_write_file,
    ),
    "list_directory": Tool(
        name="list_directory",
        description="List the contents of a directory.",
        parameters='{"path": "directory path (default: current directory)"}',
        fn=_tool_list_directory,
    ),
    "run_bash": Tool(
        name="run_bash",
        description="Execute a shell command and return its output. Only available when --allow-shell is set.",
        parameters='{"command": "shell command to execute"}',
        fn=_tool_run_bash,
    ),
    "search_files": Tool(
        name="search_files",
        description="Search for a text pattern across files in a directory (grep -rn).",
        parameters='{"pattern": "text or regex to search for", "path": "directory to search (default: .)"}',
        fn=_tool_search_files,
    ),
}
