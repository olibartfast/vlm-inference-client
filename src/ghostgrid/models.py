"""
Data models (dataclasses) for the ghostgrid.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class Agent:
    """Configuration for a single LLM or VLM agent."""

    model: str
    endpoint: str
    api_key: str
    provider: str = "openai"
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class AgentResult:
    """Result from executing a single agent call."""

    agent_id: str
    model: str
    provider: str
    content: str
    raw_response: dict
    latency_ms: float
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class InferenceConfig:
    """Shared inference parameters passed through the call stack."""

    image_paths: list[str] | None
    detail: str
    max_tokens: int
    resize: bool
    target_size: tuple[int, int]


@dataclass
class Tool:
    """Definition of a ReAct tool."""

    name: str
    description: str
    parameters: str  # JSON schema hint shown to the agent
    fn: Callable  # fn(agent, config: InferenceConfig, **kwargs) -> str


