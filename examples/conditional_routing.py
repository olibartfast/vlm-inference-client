#!/usr/bin/env python3
"""
Conditional Routing Example

Routes an image task to one of several specialists: OCR, scene understanding,
or safety monitoring. This demonstrates the `conditional` workflow.

Usage:
    OPENAI_API_KEY=... GOOGLE_API_KEY=... TOGETHER_API_KEY=... \
        python examples/conditional_routing.py image.jpg
"""

import argparse
import json
import sys

from ghostgrid import run_conditional
from ghostgrid.cli import make_agent
from ghostgrid.config import resolve_endpoint


def _build_agent(model: str, provider: str, endpoint: str | None):
    return make_agent(model, provider, resolve_endpoint(provider, endpoint))


def main() -> None:
    parser = argparse.ArgumentParser(description="Route an image task to the best-fit specialist")
    parser.add_argument("image", help="Image file path or URL")
    parser.add_argument(
        "--prompt",
        "-p",
        default="Classify this input and choose the best specialist to analyze it.",
    )
    parser.add_argument("--router-model", default="gpt-5.2")
    parser.add_argument("--router-provider", default="openai")
    args = parser.parse_args()

    categories = ["ocr", "scene", "safety"]

    try:
        router = _build_agent(args.router_model, args.router_provider, None)
        specialists = [
            _build_agent("gpt-5.2", "openai", None),
            _build_agent("gemini-2.5-flash", "google", None),
            _build_agent("meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "together", None),
        ]
    except RuntimeError as exc:
        print(f"Error: {exc}")
        print("Set the API keys required by the selected providers before running this example.")
        sys.exit(1)

    result = run_conditional(
        router_agent=router,
        specialist_agents=specialists,
        categories=categories,
        prompt=args.prompt,
        image_paths=[args.image],
        detail="low",
        max_tokens=500,
        resize=False,
        target_size=(512, 512),
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
