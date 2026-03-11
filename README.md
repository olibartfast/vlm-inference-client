# Multimodal Agent Gateway

Multi-provider LLM and VLM inference framework with 7 workflow patterns: sequential, parallel, conditional, iterative, Mixture-of-Agents (MoA), ReAct (Reasoning + Acting), and video monitoring.

## Installation

```bash
# Install from source
pip install -e .

# With video monitoring support (requires OpenCV)
pip install -e ".[video]"

# With development tools
pip install -e ".[dev,video]"
```

## Quick Start

```bash
# Install the local package and CLI
pip install -e ".[dev,video]"

# Sequential workflow (default) - each stage builds on previous
agent-gateway run --workflow sequential \
    --prompt "Summarize the tradeoffs of MoE vs dense models"

# Sequential multimodal workflow
agent-gateway run --workflow sequential \
    --prompt "Describe this image" \
    --images image.jpg \
    --models gpt-5.2 gpt-5.2 \
    --providers openai openai

# Parallel workflow - same input to multiple agents
agent-gateway run --workflow parallel \
    --prompt "What objects are in this image?" \
    --images image.jpg \
    --models gpt-5.2 meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 \
    --providers openai together

# ReAct workflow - agent reasons and uses tools
agent-gateway run --workflow react \
    --prompt "Count the people and describe what they're doing" \
    --images image.jpg \
    --model gpt-5.2 \
    --tools describe detect_objects count_objects

# Together AI text-only reasoning with NVIDIA Nemotron 3 Super
TOGETHER_API_KEY=... python3 examples/together_nemotron_reasoning.py \
    --prompt "Design an agent architecture for triaging IT support tickets"

# Video monitoring - continuous or single-shot
agent-gateway monitor \
    --video ./sample.mp4 \
    --alert-prompt "Is anyone falling or in distress?" \
    --provider google \
    --model gemini-2.5-flash

# Continuous webcam monitoring
agent-gateway monitor \
    --video 0 \
    --alert-prompt "Has anyone entered the restricted area?" \
    --continuous --interval 10
```

## Features

- **7 workflow patterns** — `sequential`, `parallel`, `conditional`, `iterative`, `moa`, `react`, `monitor`
- **Multi-provider support** — OpenAI, Anthropic, Google, Together, Azure, Groq, Mistral
- **Text-only or multimodal runs** — use `run` with just `--prompt`, or add `--images`
- **Video monitoring** — Fall detection, security monitoring, safety compliance
- **ReAct tools** — describe, detect_objects, read_text, analyze_region, count_objects
- **Observability** — per-agent latency, correlation IDs, structured JSON output

## Workflow Modes

| Mode | Agents | Data Flow |
|------|--------|-----------|
| `sequential` | ≥ 1 | Agent-1 → output-1 → Agent-2 (with context) → … → final |
| `parallel` | ≥ 2 | All agents receive same input concurrently → best answer |
| `conditional` | ≥ 2 | Router classifies input → matching specialist handles |
| `iterative` | 1 + evaluator | Agent loops, feeding output back until convergence |
| `moa` | ≥ 2 + aggregator | Parallel proposers → aggregator synthesizes |
| `react` | 1 | Thought → Action (tool) → Observation loop |
| `monitor` | 1 | Video frames → VLM analysis → structured alerts |

## Video Monitoring

The `monitor` command remains vision-specific and supports video-capable VLMs for real-time monitoring:

```bash
# Fall detection
agent-gateway monitor \
    --video ./elderly_room.mp4 \
    --alert-prompt "Is anyone falling, lying on the floor, or in distress?" \
    --fps 1 --max-frames 30

# Continuous security monitoring
agent-gateway monitor \
    --video rtsp://camera.local:554/stream \
    --alert-prompt "Has anyone entered the restricted zone?" \
    --continuous --interval 10 --window-frames 8

# Self-hosted with vLLM
agent-gateway monitor \
    --video 0 \
    --endpoint http://localhost:8000/v1/chat/completions \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --alert-prompt "Detect any hazard" \
    --continuous
```

See [docs/video-vlm-agents.md](docs/video-vlm-agents.md) for model recommendations and deployment guides.

## Supported Providers

| Provider | `--provider` | API Key Env Var |
|----------|--------------|-----------------|
| OpenAI | `openai` | `OPENAI_API_KEY` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` |
| Google | `google` | `GOOGLE_API_KEY` |
| Together AI | `together` | `TOGETHER_API_KEY` |
| Azure OpenAI | `azure` | `AZURE_OPENAI_API_KEY` |
| Groq | `groq` | `GROQ_API_KEY` |
| Mistral | `mistral` | `MISTRAL_API_KEY` |
| Cerebras | `cerebras` | `CEREBRAS_API_KEY` |

Together-hosted text model example:
- `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8` for long-context reasoning and structured text generation

## Python API

```python
from multimodal_agent_gateway import run_sequential, run_react, run_monitoring
from multimodal_agent_gateway.cli import make_agent

# Create a text-only Together AI agent
agent = make_agent(
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
    "together",
    "https://api.together.xyz/v1/chat/completions",
)

# Run a text-only workflow
result = run_sequential(
    agents=[agent],
    prompt="Summarize the tradeoffs between MoE and dense decoder-only models.",
    image_paths=[],
    detail="low",
    max_tokens=500,
    resize=False,
    target_size=(512, 512),
)
print(result["content"])
```

## Additional Examples

- `examples/react_image_analysis.py` shows a tool-using ReAct loop for OCR, counting, and scene analysis.
- `examples/conditional_routing.py` routes an image task between OCR, scene, and safety specialists.
- `examples/local_open_model.py` runs a self-hosted open model through a local OpenAI-compatible endpoint.
- `examples/together_nemotron_reasoning.py` runs NVIDIA Nemotron 3 Super on Together AI for text-only reasoning and long-context tasks.
- `examples/multi_model_analysis.py`, `examples/fall_detection.py`, and `examples/security_monitoring.py` cover MoA and monitoring flows.

## C++ Client

A lightweight C++ client for single-shot OpenAI-compatible inference.

```bash
cd vlm-inference-client/cpp
mkdir build && cd build
cmake .. && make

./vlm-inference-client \
    --prompt "Describe this image" \
    --images image.jpg \
    --model gpt-5.2 \
    --api_endpoint https://api.openai.com/v1/chat/completions \
    --api_key_env OPENAI_API_KEY
```

See [vlm-inference-client/cpp/Readme.md](vlm-inference-client/cpp/Readme.md) for details.

## Documentation

- [Video VLM Agents Guide](docs/video-vlm-agents.md) - Video-capable VLMs, vLLM deployment, hardware sizing
- [API Services](docs/api-services.md) - LLM and multimodal API providers
- [Benchmarks](docs/benchmarks.md) - LLM, VLM, and video evaluation references
- [Inference](docs/inference.md) - Inference frameworks and tools

## License

MIT
