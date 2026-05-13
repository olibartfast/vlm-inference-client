"""Command-line interface for ghostgrid."""

import argparse
import json
import logging
import os
import sys
import uuid

from ghostgrid.backends import BACKEND_CHOICES, open_backend_session
from ghostgrid.config import (
    CODE_AGENT_SYSTEM_PROMPT,
    CODE_AGENT_TOOLS,
    DEFAULT_ENDPOINT,
    PROVIDER_ENV_MAP,
    resolve_endpoint,
)
from ghostgrid.models import Agent, InferenceConfig
from ghostgrid.tools import BUILTIN_TOOLS
from ghostgrid.workflows import (
    WORKFLOW_REGISTRY,
    run_conditional,
    run_iterative,
    run_moa,
    run_react,
)


def make_agent(model: str, provider: str, endpoint: str) -> Agent:
    """Create a single Agent from model/provider/endpoint."""
    env_var = PROVIDER_ENV_MAP.get(provider.lower(), "OPENAI_API_KEY")
    api_key = os.getenv(env_var, "")

    # Allow empty API key for local endpoints
    if not api_key and "localhost" not in endpoint and "127.0.0.1" not in endpoint:
        raise RuntimeError(f"{env_var} not set (required for provider '{provider}')")
    if not api_key:
        api_key = "EMPTY"  # vLLM / SGLang local server convention

    return Agent(model=model, endpoint=endpoint, api_key=api_key, provider=provider)


def build_agents(models: list[str], providers: list[str], endpoints: list[str]) -> list[Agent]:
    """Create list of Agents from parallel model/provider/endpoint lists."""
    n = len(models)
    providers = providers or ["openai"] * n
    endpoints = endpoints or [resolve_endpoint(provider) for provider in providers]

    if len(providers) != n:
        raise ValueError(f"--providers length ({len(providers)}) must match --models length ({n})")
    if len(endpoints) != n:
        raise ValueError(f"--endpoints length ({len(endpoints)}) must match --models length ({n})")

    return [make_agent(m, p, e) for m, p, e in zip(models, providers, endpoints, strict=True)]


def _make_role_agent(model, provider, endpoint, agents: list[Agent]) -> Agent:
    """Create an agent falling back to agents[0] attrs when args are unset."""
    return make_agent(
        model or agents[0].model,
        provider or agents[0].provider,
        endpoint or agents[0].endpoint,
    )


logger = logging.getLogger(__name__)


def _setup_logging(args) -> None:
    """Configure logging based on CLI flags."""
    log_level = getattr(args, "log_level", "INFO").upper()
    log_file = getattr(args, "log_file", None)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def cmd_run(args) -> None:  # pylint: disable=too-many-locals
    """Handle standard workflow commands."""
    _setup_logging(args)
    correlation_id = str(uuid.uuid4())[:12]

    try:
        if args.agent_backend:
            sys.exit(open_backend_session(args.agent_backend, args.prompt))

        models = args.models if args.models else [args.model]
        providers = args.providers or [args.provider] * len(models)
        endpoints = args.endpoints or [args.url] * len(models)
        agents = build_agents(models, providers, endpoints)
        prompt = args.prompt
        config = InferenceConfig(
            image_paths=args.images or [],
            detail=args.detail,
            max_tokens=args.tokens,
            resize=args.resize,
            target_size=tuple(args.size),
            stream=getattr(args, "stream", False),
        )

        if args.workflow == "sequential":
            output = WORKFLOW_REGISTRY["sequential"](agents, prompt, config)

        elif args.workflow == "parallel":
            output = WORKFLOW_REGISTRY["parallel"](agents, prompt, config)

        elif args.workflow == "conditional":
            router = _make_role_agent(args.router_model, args.router_provider, args.router_endpoint, agents)
            if len(args.categories) != len(agents):
                raise ValueError(f"--categories ({len(args.categories)}) must match --models ({len(agents)})")
            output = run_conditional(router, agents, args.categories, prompt, config)

        elif args.workflow == "iterative":
            evaluator = None
            if args.evaluator_model:
                evaluator = _make_role_agent(
                    args.evaluator_model, args.evaluator_provider, args.evaluator_endpoint, agents
                )
            output = run_iterative(
                agents[0], prompt, config, evaluator_agent=evaluator, max_iterations=args.max_iterations
            )

        elif args.workflow == "moa":
            aggregator = _make_role_agent(
                args.aggregator_model, args.aggregator_provider, args.aggregator_endpoint, agents
            )
            output = run_moa(agents, aggregator, prompt, config)

        elif args.workflow == "react":
            code_agent = getattr(args, "code_agent", False)
            output = run_react(
                agents[0],
                prompt,
                config,
                enabled_tools=args.tools if args.tools else (CODE_AGENT_TOOLS if code_agent else None),
                max_steps=args.max_steps,
                system_prompt=CODE_AGENT_SYSTEM_PROMPT if code_agent else None,
                allow_shell=getattr(args, "allow_shell", False),
            )

        else:
            raise ValueError(f"Unknown workflow: {args.workflow}")

        output["correlation_id"] = correlation_id
        print(json.dumps(output, indent=2))

    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("Workflow failed")
        print(json.dumps({"error": str(exc), "correlation_id": correlation_id}, indent=2))
        sys.exit(1)


def _build_run_parser(subparsers) -> None:
    """Register the 'run' subcommand."""
    run_parser = subparsers.add_parser("run", help="Run an LLM or VLM workflow")

    run_parser.add_argument("--prompt", "-p", type=str, default="What's in this input?")
    run_parser.add_argument(
        "--images",
        "-i",
        type=str,
        nargs="*",
        default=[],
        help="Image paths or URLs (optional; not required for code-agent mode or text-only runs)",
    )
    run_parser.add_argument("--detail", "-d", type=str, default="low", choices=["auto", "low", "high"])
    run_parser.add_argument("--tokens", "-t", type=int, default=300, help="Max tokens per response")
    run_parser.add_argument("--resize", "-r", action="store_true", help="Resize images with padding")
    run_parser.add_argument("--size", "-s", type=int, nargs=2, default=[512, 512], metavar=("W", "H"))
    run_parser.add_argument(
        "--workflow",
        "-w",
        type=str,
        default="sequential",
        choices=["sequential", "parallel", "conditional", "iterative", "moa", "react"],
        help="Workflow type (default: sequential)",
    )
    run_parser.add_argument("--models", type=str, nargs="+", metavar="MODEL", help="One model per agent")
    run_parser.add_argument(
        "--providers",
        type=str,
        nargs="+",
        metavar="PROVIDER",
        help=f"Provider per agent. Known: {list(PROVIDER_ENV_MAP.keys())}",
    )
    run_parser.add_argument("--endpoints", type=str, nargs="+", metavar="URL", help="API endpoint per agent")
    run_parser.add_argument("--model", "-m", type=str, default="gpt-5.2")
    run_parser.add_argument("--url", "-u", type=str, default=DEFAULT_ENDPOINT)
    run_parser.add_argument("--provider", type=str, default="openai")
    run_parser.add_argument("--aggregator-model", type=str, default=None)
    run_parser.add_argument("--aggregator-provider", type=str, default=None)
    run_parser.add_argument("--aggregator-endpoint", type=str, default=None)
    run_parser.add_argument("--router-model", type=str, default=None)
    run_parser.add_argument("--router-provider", type=str, default=None)
    run_parser.add_argument("--router-endpoint", type=str, default=None)
    run_parser.add_argument("--categories", type=str, nargs="+", default=["general"])
    run_parser.add_argument("--evaluator-model", type=str, default=None)
    run_parser.add_argument("--evaluator-provider", type=str, default=None)
    run_parser.add_argument("--evaluator-endpoint", type=str, default=None)
    run_parser.add_argument("--max-iterations", type=int, default=3)
    run_parser.add_argument(
        "--tools",
        type=str,
        nargs="+",
        default=None,
        metavar="TOOL",
        help=f"ReAct tools to enable. Available: {list(BUILTIN_TOOLS.keys())}",
    )
    run_parser.add_argument("--max-steps", type=int, default=5)
    run_parser.add_argument(
        "--code-agent",
        action="store_true",
        help=(
            "Enable code-agent mode: uses filesystem/shell tools and a coding-focused system prompt. "
            f"Default tools: {CODE_AGENT_TOOLS}"
        ),
    )
    run_parser.add_argument(
        "--allow-shell",
        action="store_true",
        help="Allow run_bash tool to execute shell commands (opt-in for safety).",
    )
    run_parser.add_argument(
        "--agent-backend",
        type=str,
        default=None,
        choices=BACKEND_CHOICES,
        help="Delegate to an external coding-agent CLI instead of an LLM API. Choices: %(choices)s",
    )
    run_parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream tokens as they are generated (SSE). Content is accumulated and returned in the final JSON.",
    )
    run_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level for stderr output (default: INFO)",
    )
    run_parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Write logs to a file in addition to stderr",
    )
    run_parser.set_defaults(func=cmd_run)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="ghostgrid",
        description=(
            "ghostgrid — multi-provider LLM and VLM inference with "
            "sequential, parallel, conditional, iterative, MoA, and ReAct workflows"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    _build_run_parser(subparsers)

    args = parser.parse_args()

    if args.command is None:
        if "--images" in sys.argv or "-i" in sys.argv or "--prompt" in sys.argv or "-p" in sys.argv:
            sys.argv.insert(1, "run")
            args = parser.parse_args()
        else:
            parser.print_help()
            sys.exit(0)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
