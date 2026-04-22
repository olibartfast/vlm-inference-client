#!/usr/bin/env python3
"""
Gemma 4 Inference Example

Tests text inference with the latest Gemma 4 model via OpenRouter or Together AI.
Both providers expose an OpenAI-compatible chat/completions endpoint.

Usage (OpenRouter):
    OPENROUTER_API_KEY=... python examples/gemma4_inference.py \
        --prompt "Explain the difference between transformers and state-space models"

Usage (Together AI):
    TOGETHER_API_KEY=... python examples/gemma4_inference.py \
        --provider together \
        --prompt "Explain the difference between transformers and state-space models"

Usage (custom model / workflow):
    OPENROUTER_API_KEY=... python examples/gemma4_inference.py \
        --prompt "Write a haiku about neural networks" \
        --workflow iterative \
        --max-iterations 2
"""

import argparse
import json
import sys

from ghostgrid import run_iterative, run_sequential
from ghostgrid.cli import make_agent
from ghostgrid.config import resolve_endpoint

PROVIDER_MODELS = {
    "openrouter": "google/gemma-4-31b-it",
    "together": "google/gemma-4-27b-it",  # verify ID at api.together.xyz/v1/models
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gemma 4 inference via OpenRouter or Together AI")
    parser.add_argument("--prompt", "-p", required=True, help="Text prompt for the model")
    parser.add_argument(
        "--provider",
        choices=["openrouter", "together"],
        default="openrouter",
        help="Inference provider (default: openrouter)",
    )
    parser.add_argument(
        "--workflow",
        choices=["sequential", "iterative"],
        default="sequential",
        help="Workflow type (default: sequential)",
    )
    parser.add_argument("--model", default=None, help="Override model ID (default: provider-appropriate Gemma 4 ID)")
    parser.add_argument("--endpoint", default=None, help="Override API endpoint URL")
    parser.add_argument("--tokens", type=int, default=512, help="Max tokens to generate (default: 512)")
    parser.add_argument("--max-iterations", type=int, default=2, help="Max iterations for iterative workflow")
    args = parser.parse_args()

    model = args.model or PROVIDER_MODELS[args.provider]
    endpoint = resolve_endpoint(args.provider, args.endpoint)

    try:
        agent = make_agent(model, args.provider, endpoint)
    except RuntimeError as exc:
        env_var = "OPENROUTER_API_KEY" if args.provider == "openrouter" else "TOGETHER_API_KEY"
        print(f"Error: {exc}")
        print(f"Make sure {env_var} is set.")
        sys.exit(1)

    common = {
        "prompt": args.prompt,
        "image_paths": [],
        "detail": "low",
        "max_tokens": args.tokens,
        "resize": False,
        "target_size": (512, 512),
    }

    if args.workflow == "iterative":
        result = run_iterative(
            agent=agent,
            **common,
            evaluator_agent=None,
            max_iterations=args.max_iterations,
        )
    else:
        result = run_sequential(agents=[agent], **common)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
