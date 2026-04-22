#!/usr/bin/env python3
"""
Local Open-Model Example

Demonstrates a fully local workflow against an OpenAI-compatible server such as
vLLM or SGLang serving an open multimodal model.

Usage:
    python examples/local_open_model.py image.jpg

    python examples/local_open_model.py image.jpg \
        --endpoint http://localhost:8000/v1/chat/completions \
        --model Qwen/Qwen3-VL-8B-Instruct
"""

import argparse
import json

from ghostgrid import run_sequential
from ghostgrid.models import Agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local open multimodal model through the sequential workflow")
    parser.add_argument("image", help="Image file path or URL")
    parser.add_argument("--prompt", "-p", default="Describe the image and list the most important visual details.")
    parser.add_argument("--endpoint", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--provider", default="openai", help="Provider label used for response metadata")
    args = parser.parse_args()

    agent = Agent(
        model=args.model,
        endpoint=args.endpoint,
        api_key="EMPTY",
        provider=args.provider,
    )

    result = run_sequential(
        agents=[agent],
        prompt=args.prompt,
        image_paths=[args.image],
        detail="low",
        max_tokens=400,
        resize=False,
        target_size=(512, 512),
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
