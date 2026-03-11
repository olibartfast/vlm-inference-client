# VLMs for Video Understanding & Agentic Monitoring

A guide to video-capable Vision Language Models and how to build monitoring
applications driven by agentic AI — using cloud APIs and self-hosted inference
(vLLM / SGLang).

---

## Video-Capable VLM Landscape (2026)

### Tier 1 — Frontier Video VLMs (Reasoning-First)

| Model | Params | Video Support | Key Strengths | Access |
|-------|--------|--------------|---------------|--------|
| **Gemini 3 Pro / Flash** | Closed | Native video (1M+ context) | **SOTA 2026**; "Deep Think" visual reasoning, 1hr+ high-res video | Google API |
| **GPT-5 / 5.2** | Closed | Native video / Frames | **SOTA 2026**; Agentic UI execution, unified multimodal reasoning | OpenAI API |
| **Qwen 3.5-VL** | 8B / 250B+ (MoE) | Unified Pixels-as-Tokens | **Open SOTA**; Single-stream architecture (no separate encoder) | HF, vLLM, DashScope |
| **Claude 4.5 Opus** | Closed | Frames as images | Best-in-class OCR and dense document/interface reasoning | Anthropic API |
| **Qwen3-VL** | 8B / 235B (MoE) | Native video + dynamic FPS | 3D grounding, visual agent, stable production choice | HF, vLLM, DashScope API |
| **Kimi K2.5** | Closed / open-weight variants | Native multimodal | Strong agentic multimodal system with image and video support | Kimi, Azure AI Foundry |
| **GLM-4.6V / 4.5V** | Closed / mixed release model family | Native multimodal | Current GLM vision line for multimodal reasoning and video-adjacent workflows | Z.AI |
| **GLM-4.1V-Thinking** | 9B | Native video, time-index tokens | Reasoning-first (RLCS), extreme efficiency for edge | HF, vLLM, Zhipu API |
| **Llama 3.2 Vision (Cerebras)** | 11B / 90B | Frames as images | **Extreme Speed**; Hundreds of tokens/sec for real-time analysis | Cerebras API |

### Tier 2 — Stable & Legacy Video VLMs

| Model | Params | Notes |
|-------|--------|-------|
| **Llama 4 Maverick** | 400B (MoE) | Meta's 2025 flagship; Native multimodal, rivaling GPT-4.5 levels |
| **Gemini 2.x Flash/Pro** | Closed | Stable legacy; Great for long video upload and audio analysis |
| **GPT-4o / 4o-mini** | Closed | Stable legacy; Broad capability, widely available |
| **Qwen 2.5-VL** | 3B / 72B | Stable legacy; Event pinpointing, absolute time encoding |
| **LLaVA-Video** | 7B / 72B | Strong on VideoMME benchmarks |

### SOTA Open-Weight Models (March 11, 2026)

If you want the strongest self-hostable options rather than closed APIs, these are the current high-signal choices:

| Model | Class | Why it matters | Best fit |
|-------|-------|----------------|----------|
| **Qwen3-VL** | Apache-2.0 open weights | Qwen positions it as its strongest VLM line so far, with improved video dynamics, spatial reasoning, and agent interaction. | General-purpose open default for image + video |
| **MiniCPM-V 4.5** | Apache-2.0 open weights | OpenBMB positions it as a top sub-30B multimodal model with unusually strong high-FPS and long-video efficiency. | Edge and cost-sensitive video workloads |
| **GLM-4.1V-9B-Thinking** | Open weights | Z.ai reports state-of-the-art performance among 10B-class VLMs, with reasoning-heavy behavior that competes well above its size. | Small-model reasoning and OCR-heavy pipelines |
| **Kimi K2.5 open variants** | Open-weight release track | Moonshot positions K2.5 as a native multimodal model family with image and video support plus strong agent tooling. | Frontier-style multimodal experimentation |
| **Llama 4 Maverick** | Source-available weights | Native multimodal MoE with large context and strong cloud/self-hosted ecosystem support. | High-throughput deployments when license terms are acceptable |

Notes:
- `Qwen/Qwen3-VL-8B-Instruct` is the safest default open model to recommend in this repo today.
- `openbmb/MiniCPM-V-4_5` is especially attractive when video token efficiency matters more than absolute frontier accuracy.
- `Kimi K2.5` belongs in the VLM landscape because Moonshot explicitly ships it for text, images, and video; use it when you want an agentic multimodal stack rather than a small self-hosted edge model.
- `GLM-5` is not listed here as a primary video VLM because Z.AI currently separates it from the vision-specific `GLM-4.6V` / `GLM-4.5V` line.
- `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` is powerful, but it is not OSI-open; treat it as source-available under Meta's community license.

### How Video Input Works

All these models process video through one of two mechanisms:

**Native video path** — The model's preprocessor handles frame extraction internally.
Qwen2.5-VL, Qwen3-VL, and GLM-4.1V-Thinking support this natively through
transformers or vLLM, with configurable FPS and resolution constraints.

**Frames-as-images** — You extract frames yourself and send them as multiple
`image_url` content blocks in a single chat-completions request. This is the
universal approach that works with any OpenAI-compatible API (GPT-4o, vLLM,
Together AI, Groq, etc.) and is what our example script uses.

```
Video → [Frame Extractor] → JPEG frames → [base64 encode] → API request
         (OpenCV/ffmpeg)                    (multiple image_url blocks)
```

---

## Self-Hosted Deployment with vLLM

vLLM is the recommended serving engine. It exposes an OpenAI-compatible API
and supports all the models listed above.

### Quick Start: Qwen 3.5-VL-8B

```bash
pip install vllm>=0.8.0 qwen-vl-utils>=0.1.0

vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --limit-mm-per-prompt video=1 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.95
```

### Quick Start: GLM-4.1V-9B-Thinking

```bash
vllm serve zai-org/GLM-4.1V-9B-Thinking \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code
```

### Hardware Guidelines (2026)

| Model | Min GPU | Recommended | Notes |
|-------|---------|-------------|-------|
| Qwen 3-VL-8B | RTX 4070 16GB | RTX 5090 / A100 | Strong current edge default |
| GLM-4.1V-9B-Thinking | RTX 3090 24GB | A100 40GB | Best sub-10B reasoning |
| Llama 4 Maverick (400B) | 8× A100 80GB | 8× H200 | Open-weights frontier |
| Qwen 3.5-VL-250B+ (MoE) | 8× H100 80GB | 8× H200/B200 | Enterprise SOTA |

> **Tip for your RTX 3060:** Qwen2.5-VL-3B with `--max-model-len 4096` and
> AWQ quantization fits comfortably. Use `--limit-mm-per-prompt image=4 video=0`
> to keep memory under control.

---

## Building a Monitoring Application

The `examples/video_monitoring_agent.py` script implements a complete
monitoring pipeline:

```
┌─────────────┐     ┌───────────────┐     ┌──────────────┐     ┌─────────────┐
│ Video Source │────▶│ Frame Extract │────▶│  VLM Agent   │────▶│   Alerting  │
│ file/webcam/ │     │ OpenCV @ FPS  │     │ ReAct-style  │     │ console/log │
│ RTSP stream  │     │ → base64 JPEG │     │ analysis     │     │ webhook/etc │
└─────────────┘     └───────────────┘     └──────────────┘     └─────────────┘
```

### Architecture

1. **Frame Extraction** — OpenCV captures frames at a configurable FPS rate.
   For a 30fps source with `--fps 1`, we sample 1 frame per second. Frames
   are JPEG-compressed and base64-encoded.

2. **VLM Analysis** — Frames are sent as a batch to the VLM with a structured
   system prompt. The agent must output a specific format:
   `Thought → Alert (YES/NO) → Summary → Confidence → Recommended Action`.

3. **Alert Routing** — The parsed response is dispatched to alert handlers
   (console, JSONL file). Extend with webhooks, MQTT, email, etc.

4. **Continuous Mode** — With `--continuous`, the script loops: capture a
   window of N frames → analyze → wait → repeat. Designed for live sources.

### Example: Fall Detection

```bash
# Fall detection
vlm-agent-gateway monitor \
    --video ./elderly_room.mp4 \
    --alert-prompt "Is anyone falling, lying on the floor, or in distress?" \
    --provider google \
    --model gemini-2.5-flash \
    --fps 1 --max-frames 30

# Continuous webcam monitoring with local vLLM
vlm-agent-gateway monitor \
    --video 0 \
    --provider openai \
    --endpoint http://localhost:8000/v1/chat/completions \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --alert-prompt "Is anyone falling or showing signs of distress?" \
    --fps 0.5 --continuous --interval 15 --window-frames 8 \
    --output-jsonl ./fall_alerts.jsonl
```

### Example: Security Monitoring

```bash
vlm-agent-gateway monitor \
    --video rtsp://camera.local:554/stream \
    --provider google \
    --model gemini-2.5-flash \
    --alert-prompt "Has anyone entered the restricted zone marked by yellow tape?" \
    --fps 0.5 --continuous --interval 10
```

### Example: Industrial Safety

```bash
vlm-agent-gateway monitor \
    --video ./factory_floor.mp4 \
    --provider openai \
    --model gpt-5.2 \
    --alert-prompt "Is any worker not wearing a hard hat or safety vest?" \
    --fps 1 --max-frames 20
```

---

## Integration with vlm-agent-gateway Workflows

The monitoring agent is a standalone script, but it's designed to compose
with the gateway's workflow patterns:

### MoA (Mixture-of-Agents) for Higher Confidence

Run the same frames through multiple VLMs and aggregate:

```bash
# Step 1: Extract frames once, run through multiple models
# Step 2: Use the gateway's MoA mode with an aggregator

python vlm-agent-gateway/main.py \
    --workflow moa \
    --prompt "Is anyone in this scene falling or in distress?" \
    --images frame_001.jpg frame_002.jpg frame_003.jpg \
    --models meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 gpt-5.2 \
    --providers together openai \
    --endpoints https://api.together.xyz/v1/chat/completions https://api.openai.com/v1/chat/completions \
    --aggregator-model gpt-5.2 \
    --aggregator-provider openai
```

### Sequential for Multi-Stage Analysis

First detect → then classify → then assess severity:

```bash
python vlm-agent-gateway/main.py \
    --workflow sequential \
    --prompt "Analyze these surveillance frames for safety incidents" \
    --images frame_*.jpg \
    --models gpt-5.2 gpt-5.2 \
    --providers openai openai
```

---

## Model Selection Guide

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| **Quick prototyping** | Gemini 2.5 Flash via Google API | Stable multimodal default with native video input |
| **Cost-effective cloud** | Llama 4 Maverick via Together AI | Current Together recommendation for vision workloads |
| **Best reasoning (self-hosted)** | GLM-4.1V-9B-Thinking on vLLM | 9B model that competes with 72B, chain-of-thought |
| **Edge / low-resource** | Qwen3-VL-8B + AWQ on vLLM | Modern open-weight option for single-GPU deployments |
| **Long video (1hr+)** | Gemini 2.5 Flash or Qwen3-VL | Large context windows with video-aware input handling |
| **Maximum accuracy** | GPT-5.2 or Gemini 3 Pro | Frontier closed-model performance |
| **Lowest latency / Real-time** | Llama 3.2 Vision on Cerebras | Extreme inference speed (100s of tokens/sec) |


---

## Key Considerations for Production

### Frame Sampling Strategy

- **Higher FPS** = better temporal resolution but more tokens and cost.
- **Typical monitoring**: 0.5–1 FPS is sufficient for fall detection, intrusion.
- **Fast-moving events** (traffic, sports): 2–5 FPS may be needed.
- Use `--detail low` to minimize token usage per frame (~85 tokens vs ~765 for high).

### Latency Budget

| Backend | Typical Latency (8 frames) | Notes |
|---------|---------------------------|-------|
| GPT-4o API | 3–8s | Consistent, no setup |
| Together AI (72B) | 5–15s | Variable under load |
| vLLM local (7B, A100) | 2–5s | Best for continuous monitoring |
| vLLM local (3B, RTX 3060) | 3–8s | Viable for edge |

### Cost Estimation

For continuous monitoring at 1 cycle/10s with 8 frames/cycle:
- **GPT-4o**: ~$0.30–0.60/hour (low detail)
- **Together AI (72B)**: ~$0.10–0.20/hour
- **Self-hosted**: GPU cost only (amortized)

### Extending the Alert Pipeline

The `AlertEvent` dataclass and handler pattern are designed for extension:

```python
# Webhook handler example
def alert_handler_webhook(event: AlertEvent, url: str) -> None:
    if event.alert:
        requests.post(url, json=event.__dict__, timeout=5)

# MQTT handler for IoT integration
def alert_handler_mqtt(event: AlertEvent, topic: str, client) -> None:
    if event.alert:
        client.publish(topic, json.dumps(event.__dict__))

# Home Assistant integration
def alert_handler_ha(event: AlertEvent, ha_url: str, token: str) -> None:
    if event.alert:
        requests.post(
            f"{ha_url}/api/services/notify/notify",
            headers={"Authorization": f"Bearer {token}"},
            json={"message": f"🚨 {event.summary}", "title": "Video Alert"},
        )
```

---

## Further Reading

- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)
- [Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL)
- [GLM-4.1V-Thinking Paper](https://arxiv.org/abs/2507.01006)
- [GLM-V GitHub](https://github.com/zai-org/GLM-V)
- [LLaVA-Video Blog](https://llava-vl.github.io/blog/2024-09-30-llava-video/)
- [LLaVA-NeXT GitHub](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [vLLM VLM Serving Guide](https://docs.vllm.ai/en/stable/models/vlm.html)
- [vLLM Qwen3-VL Recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html)
- [NVIDIA Jetson VLM Agent (edge deployment)](https://developer.nvidia.com/blog/develop-generative-ai-powered-visual-ai-agents-for-the-edge/)
