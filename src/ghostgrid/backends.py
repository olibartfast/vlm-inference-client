"""Agent backend dispatch — opens interactive sessions with external coding-agent CLIs."""

import logging
import os
import subprocess
from collections.abc import Callable

from ghostgrid.config import CREDENTIAL_ENV_VARS

logger = logging.getLogger(__name__)

_BACKEND_CMDS: dict[str, Callable[[str | None], list[str]]] = {
    "claude-code": lambda p: ["claude", p] if p else ["claude"],
    "codex": lambda p: ["codex", p] if p else ["codex"],
    "opencode": lambda p: ["opencode", p] if p else ["opencode"],
    "pi": lambda p: ["pi", p] if p else ["pi"],
}

BACKEND_CHOICES: list[str] = list(_BACKEND_CMDS)


def sanitize_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return a copy of os.environ with credential env vars redacted.

    Extra env vars from *extra_env* are layered on top (and are NOT redacted).
    """
    env = {k: v for k, v in os.environ.items() if k not in CREDENTIAL_ENV_VARS}
    if extra_env:
        env.update(extra_env)
    return env


def open_backend_session(
    backend: str,
    prompt: str | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    sanitize: bool = True,
) -> int:
    """Launch an interactive coding-agent session and return its exit code.

    When *sanitize* is True (default), credential environment variables are
    stripped from the inherited environment. Pass explicit credentials via *env*.
    """
    if backend not in _BACKEND_CMDS:
        raise ValueError(f"Unknown agent backend: {backend!r}")
    merged_env = sanitize_env(env) if sanitize else ({**os.environ, **env} if env else None)
    logger.info("Launching %s backend session", backend)
    result = subprocess.run(_BACKEND_CMDS[backend](prompt), cwd=cwd, env=merged_env, check=False)
    return result.returncode
