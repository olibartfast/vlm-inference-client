# ghostgrid

![Alt text](data/ghostgrid.png)

> A grid for multimodal agents — route vision, text, and code across providers through one gateway.

Multi-provider LLM and VLM inference framework with 7 workflow patterns: sequential, parallel, conditional, iterative, Mixture-of-Agents (MoA), ReAct (Reasoning + Acting), and video monitoring — including **code-agent mode** (filesystem + shell tools, similar to Claude Code / OpenAI Codex).

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
ghostgrid run --workflow sequential \
    --prompt "Summarize the tradeoffs of MoE vs dense models"

# Sequential multimodal workflow
ghostgrid run --workflow sequential \
    --prompt "Describe this image" \
    --images image.jpg \
    --models gpt-5.2 gpt-5.2 \
    --providers openai openai

# Parallel workflow - same input to multiple agents
ghostgrid run --workflow parallel \
    --prompt "What objects are in this image?" \
    --images image.jpg \
    --models gpt-5.2 meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 \
    --providers openai together

# ReAct workflow - vision tools (images required)
ghostgrid run --workflow react \
    --prompt "Count the people and describe what they're doing" \
    --images image.jpg \
    --model gpt-5.2 \
    --tools describe detect_objects count_objects

# Code-agent mode - filesystem tools, no image needed
ghostgrid run --workflow react \
    --prompt "Find all TODO comments in the src/ directory and summarise them" \
    --model gpt-5.2 \
    --code-agent

# Code-agent with shell execution (opt-in)
ghostgrid run --workflow react \
    --prompt "Run the test suite and report any failures" \
    --model gpt-5.2 \
    --code-agent --allow-shell

# Together AI text-only reasoning with NVIDIA Nemotron 3 Super
TOGETHER_API_KEY=... python3 examples/together_nemotron_reasoning.py \
    --prompt "Design an agent architecture for triaging IT support tickets"

# Video monitoring - continuous or single-shot
ghostgrid monitor \
    --video ./sample.mp4 \
    --alert-prompt "Is anyone falling or in distress?" \
    --provider google \
    --model gemini-2.5-flash

# Continuous webcam monitoring
ghostgrid monitor \
    --video 0 \
    --alert-prompt "Has anyone entered the restricted area?" \
    --continuous --interval 10
```

## Features

- **7 workflow patterns** — `sequential`, `parallel`, `conditional`, `iterative`, `moa`, `react`, `monitor`
- **Multi-provider support** — OpenAI, Anthropic (native), Google, Together, Azure, Groq, Mistral, Cerebras
- **Text-only or multimodal runs** — use `run` with just `--prompt`, or add `--images`
- **Video monitoring** — Fall detection, security monitoring, safety compliance
- **Vision ReAct tools** — `describe`, `detect_objects`, `read_text`, `analyze_region`, `count_objects`
- **Code-agent mode** — `read_file`, `write_file`, `list_directory`, `search_files`, `run_bash` (opt-in)
- **`--images` optional** — code-agent tasks work without any image input
- **Observability** — per-agent latency, correlation IDs, structured JSON output

## Workflow Modes

| Mode | Agents | Data Flow |
| --- | --- | --- |
| `sequential` | ≥ 1 | Agent-1 → output-1 → Agent-2 (with context) → … → final |
| `parallel` | ≥ 2 | All agents receive same input concurrently → best answer |
| `conditional` | ≥ 2 | Router classifies input → matching specialist handles |
| `iterative` | 1 + evaluator | Agent loops, feeding output back until convergence |
| `moa` | ≥ 2 + aggregator | Parallel proposers → aggregator synthesizes |
| `react` | 1 | Thought → Action (tool) → Observation loop |
| `monitor` | 1 | Video frames → VLM analysis → structured alerts |

## ReAct Tools

The `react` workflow exposes a registry of tools the agent can invoke at each step.

### Vision tools (require `--images`)

| Tool | Description |
| --- | --- |
| `describe` | Generate a detailed description of the image(s) |
| `detect_objects` | List all distinct objects visible in the image(s) |
| `read_text` | Extract all visible text (OCR) from the image(s) |
| `analyze_region` | Focus analysis on a described spatial region |
| `count_objects` | Count occurrences of a named object |

### Code-agent / filesystem tools

| Tool | Description |
| --- | --- |
| `read_file` | Read a file from the local filesystem |
| `write_file` | Write or overwrite a file (creates parent directories) |
| `list_directory` | List the contents of a directory |
| `search_files` | Search for a text pattern across files (`grep -rn`) |
| `run_bash` | Execute a shell command — **requires `--allow-shell`** |

## Code-Agent Mode

`--code-agent` activates a coding-focused subset of tools and replaces the vision system prompt with one tuned for filesystem/code tasks. This mimics how tools like Claude Code and OpenAI Codex operate — a ReAct loop with file and shell access.

```bash
# Explore a codebase and answer a question (safe, no shell)
ghostgrid run --workflow react \
    --prompt "What does the run_react function do? Read the relevant source files." \
    --model claude-3-5-sonnet-20241022 \
    --provider anthropic \
    --code-agent

# Write a new feature (reads existing files, writes new ones)
ghostgrid run --workflow react \
    --prompt "Add a hello_world() function to src/utils.py" \
    --model gpt-5.2 \
    --code-agent

# Run tests and fix failures (shell enabled)
ghostgrid run --workflow react \
    --prompt "Run pytest and fix any failing tests" \
    --model gpt-5.2 \
    --code-agent --allow-shell --max-steps 10

# Combine vision + code: analyse a screenshot then write code based on it
ghostgrid run --workflow react \
    --prompt "Read the UI shown in the screenshot and generate matching HTML" \
    --images screenshot.png \
    --model gpt-5.2 \
    --tools describe read_file write_file
```

> **Safety note:** `run_bash` is blocked by default. Pass `--allow-shell` explicitly to enable it. All file writes are permanent — review the agent's plan before using `--allow-shell` in sensitive directories.

## Video Monitoring

The `monitor` command remains vision-specific and supports video-capable VLMs for real-time monitoring:

```bash
# Fall detection
ghostgrid monitor \
    --video ./elderly_room.mp4 \
    --alert-prompt "Is anyone falling, lying on the floor, or in distress?" \
    --fps 1 --max-frames 30

# Continuous security monitoring
ghostgrid monitor \
    --video rtsp://camera.local:554/stream \
    --alert-prompt "Has anyone entered the restricted zone?" \
    --continuous --interval 10 --window-frames 8

# Self-hosted with vLLM
ghostgrid monitor \
    --video 0 \
    --endpoint http://localhost:8000/v1/chat/completions \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --alert-prompt "Detect any hazard" \
    --continuous
```

See [docs/video-vlm-agents.md](docs/video-vlm-agents.md) for model recommendations and deployment guides.

## Supported Providers

| Provider | `--provider` | API Key Env Var | Notes |
| --- | --- | --- | --- |
| OpenAI | `openai` | `OPENAI_API_KEY` |  |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` | Native Messages API |
| Google | `google` | `GOOGLE_API_KEY` | OpenAI-compat endpoint |
| Together AI | `together` | `TOGETHER_API_KEY` |  |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | Good default for Kimi multimodal access |
| Z.AI | `zai` | `ZAI_API_KEY` | `GLM-5.1` text-only, `GLM-4.6V` for vision |
| Azure OpenAI | `azure` | `AZURE_OPENAI_API_KEY` | Requires explicit `--url` |
| Groq | `groq` | `GROQ_API_KEY` |  |
| Mistral | `mistral` | `MISTRAL_API_KEY` |  |
| Cerebras | `cerebras` | `CEREBRAS_API_KEY` |  |

Together-hosted text model example:

- `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8` for long-context reasoning and structured text generation
- `zai-org/GLM-5.1` for text-only reasoning and coding

Multimodal provider examples:

- OpenRouter: `moonshotai/kimi-k2.6` for image-aware multimodal runs
- Z.AI: `glm-4.6v` for GLM-family vision runs

`GLM-5.1` is available on Together AI and Z.AI, but it is text-only. Use `GLM-4.6V` when you need image input.

## Python API

```python
from ghostgrid import run_sequential, run_react, run_monitoring
from ghostgrid.cli import make_agent
from ghostgrid.config import CODE_AGENT_SYSTEM_PROMPT, CODE_AGENT_TOOLS

# Create a text-only Together AI agent
agent = make_agent(
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
    "together",
    "https://api.together.xyz/v1/chat/completions",
)

# Create an OpenRouter multimodal agent for Kimi K2.6
kimi_agent = make_agent(
    "moonshotai/kimi-k2.6",
    "openrouter",
    "https://openrouter.ai/api/v1/chat/completions",
)

# Create a Z.AI multimodal GLM agent
glm_vision_agent = make_agent(
    "glm-4.6v",
    "zai",
    "https://api.z.ai/api/paas/v4/chat/completions",
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

# Code-agent — filesystem tools, no image
result = run_react(
    agent=agent,
    prompt="List all Python files and summarise what each one does",
    image_paths=[],
    detail="low",
    max_tokens=1024,
    resize=False,
    target_size=(512, 512),
    enabled_tools=CODE_AGENT_TOOLS,
    system_prompt=CODE_AGENT_SYSTEM_PROMPT,
    allow_shell=False,   # set True to enable run_bash
    max_steps=10,
)
print(result["content"])
```

## Additional Examples

- `examples/react_image_analysis.py` shows a tool-using ReAct loop for OCR, counting, and scene analysis.
- `examples/conditional_routing.py` routes an image task between OCR, scene, and safety specialists.
- `examples/local_open_model.py` runs a self-hosted open model through a local OpenAI-compatible endpoint.
- `examples/together_nemotron_reasoning.py` runs NVIDIA Nemotron 3 Super on Together AI for text-only reasoning and long-context tasks.
- `examples/multi_model_analysis.py`, `examples/fall_detection.py`, and `examples/security_monitoring.py` cover MoA and monitoring flows.

## Documentation

- [Video VLM Agents Guide](docs/video-vlm-agents.md) — Video-capable VLMs, vLLM deployment, hardware sizing
- [API Services](docs/api-services.md) — LLM and multimodal API providers
- [Benchmarks](docs/benchmarks.md) — LLM, VLM, and video evaluation references
- [Inference](docs/inference.md) — Inference frameworks and tools

## License

MIT
