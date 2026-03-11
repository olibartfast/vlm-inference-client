#!/usr/bin/env python3
"""
Together AI Nemotron Reasoning Example

Demonstrates a text-only reasoning workflow using NVIDIA Nemotron 3 Super
served through Together AI.

Usage:
    TOGETHER_API_KEY=... python examples/together_nemotron_reasoning.py \
        --prompt "Design an agent architecture for triaging IT support tickets"

    TOGETHER_API_KEY=... python examples/together_nemotron_reasoning.py \
        --prompt "Classify these incidents and propose a response plan" \
        --context-file ./incident_log.txt \
        --workflow iterative
"""

import argparse
import json
import sys
from pathlib import Path

from multimodal_agent_gateway import run_iterative, run_sequential
from multimodal_agent_gateway.cli import make_agent
from multimodal_agent_gateway.config import resolve_endpoint


DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"


def load_context(path: str | None) -> str:
    """Load optional extra context from a text file."""
    if not path:
        return ""
    return Path(path).read_text(encoding="utf-8").strip()


def build_reasoning_prompt(task: str, extra_context: str) -> str:
    """Compose a prompt that asks for structured reasoning output."""
    prompt = (
        "You are a senior AI systems engineer. Solve the task carefully and produce a response with these sections:\n"
        "1. Goal\n"
        "2. Key assumptions\n"
        "3. Step-by-step plan\n"
        "4. Risks\n"
        "5. Final recommendation\n\n"
        f"Task:\n{task.strip()}"
    )
    if extra_context:
        prompt += f"\n\nAdditional context:\n{extra_context}"
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NVIDIA Nemotron 3 Super on Together AI")
    parser.add_argument("--prompt", "-p", required=True, help="Task or question for Nemotron")
    parser.add_argument(
        "--workflow",
        choices=["sequential", "iterative"],
        default="sequential",
        help="Reasoning workflow to run",
    )
    parser.add_argument(
        "--context-file",
        help="Optional UTF-8 text file to include in the prompt, useful for long-context reasoning",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider", default="together")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--tokens", type=int, default=900)
    parser.add_argument("--max-iterations", type=int, default=3)
    args = parser.parse_args()

    try:
        agent = make_agent(args.model, args.provider, resolve_endpoint(args.provider, args.endpoint))
    except RuntimeError as exc:
        print(f"Error: {exc}")
        print("Make sure TOGETHER_API_KEY is set.")
        sys.exit(1)

    prompt = build_reasoning_prompt(args.prompt, load_context(args.context_file))

    if args.workflow == "iterative":
        result = run_iterative(
            agent=agent,
            prompt=prompt,
            image_paths=[],
            detail="low",
            max_tokens=args.tokens,
            resize=False,
            target_size=(512, 512),
            evaluator_agent=None,
            max_iterations=args.max_iterations,
        )
    else:
        result = run_sequential(
            agents=[agent],
            prompt=prompt,
            image_paths=[],
            detail="low",
            max_tokens=args.tokens,
            resize=False,
            target_size=(512, 512),
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
