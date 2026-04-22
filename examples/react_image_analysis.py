#!/usr/bin/env python3
"""
ReAct Image Analysis Example

Demonstrates the tool-using `react` workflow for image analysis tasks that
benefit from intermediate OCR, object counting, and region-focused passes.

Usage:
    OPENAI_API_KEY=... python examples/react_image_analysis.py image.jpg

    GOOGLE_API_KEY=... python examples/react_image_analysis.py receipt.jpg \
        --provider google \
        --model gemini-2.5-flash \
        --prompt "Read the total amount and summarize the receipt"
"""

import argparse
import json
import sys

from ghostgrid import run_react
from ghostgrid.cli import make_agent
from ghostgrid.config import resolve_endpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Tool-using image analysis with the ReAct workflow")
    parser.add_argument("image", help="Image file path or URL")
    parser.add_argument(
        "--prompt",
        "-p",
        default="Count the people, read any visible text, and explain what is happening in the scene.",
    )
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--detail", default="low", choices=["auto", "low", "high"])
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument(
        "--tools",
        nargs="+",
        default=["describe", "read_text", "count_objects", "analyze_region"],
        help="Subset of built-in tools to enable",
    )
    args = parser.parse_args()

    try:
        agent = make_agent(args.model, args.provider, resolve_endpoint(args.provider, args.endpoint))
    except RuntimeError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    result = run_react(
        agent=agent,
        prompt=args.prompt,
        image_paths=[args.image],
        detail=args.detail,
        max_tokens=500,
        resize=False,
        target_size=(512, 512),
        enabled_tools=args.tools,
        max_steps=args.max_steps,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
