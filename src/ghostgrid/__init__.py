"""
ghostgrid - Multi-provider LLM and VLM inference with workflow support.

A framework for building agentic applications with text, vision, and multimodal models.
Supports multiple providers (OpenAI, Anthropic, Google, Together, Azure, Groq, Mistral)
and 6 workflow patterns (sequential, parallel, conditional, iterative, MoA, ReAct).
"""

from pathlib import Path

# Agent backends
from ghostgrid.backends import BACKEND_CHOICES, open_backend_session, sanitize_env

# Core models
# Configuration
from ghostgrid.config import (
    CREDENTIAL_ENV_VARS,
    DEFAULT_ENDPOINT,
    PROVIDER_ENV_MAP,
    WORKFLOW_CHOICES,
    get_api_key,
    get_default_endpoint,
    resolve_endpoint,
)

# Image utilities
from ghostgrid.image import encode_image, is_url, resize_with_padding
from ghostgrid.models import Agent, AgentResult, InferenceConfig, Tool

# Provider functions
from ghostgrid.providers import (
    create_payload,
    normalize_response,
    run_agent,
    send_request,
    stream_request,
)

# Tools
from ghostgrid.tools import BUILTIN_TOOLS, register_tool, unregister_tool

# Workflows
from ghostgrid.workflows import (
    WORKFLOW_REGISTRY,
    run_conditional,
    run_iterative,
    run_moa,
    run_parallel,
    run_react,
    run_sequential,
)


def _read_version() -> str:
    """Read version from VERSION file."""
    version_file = Path(__file__).parent.parent.parent / "VERSION"
    if version_file.exists():
        return version_file.read_text().strip()
    return "0.0.0"


__version__ = _read_version()

__all__ = [
    # Version
    "__version__",
    # Backends
    "BACKEND_CHOICES",
    "open_backend_session",
    "sanitize_env",
    # Models
    "Agent",
    "AgentResult",
    "InferenceConfig",
    "Tool",
    # Config
    "CREDENTIAL_ENV_VARS",
    "DEFAULT_ENDPOINT",
    "PROVIDER_ENV_MAP",
    "WORKFLOW_CHOICES",
    "get_api_key",
    "get_default_endpoint",
    "resolve_endpoint",
    # Image
    "encode_image",
    "is_url",
    "resize_with_padding",
    # Providers
    "create_payload",
    "normalize_response",
    "run_agent",
    "send_request",
    "stream_request",
    # Tools
    "BUILTIN_TOOLS",
    "register_tool",
    "unregister_tool",
    # Workflows
    "WORKFLOW_REGISTRY",
    "run_conditional",
    "run_iterative",
    "run_moa",
    "run_parallel",
    "run_react",
    "run_sequential",
]
