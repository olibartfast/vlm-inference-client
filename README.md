# VLM Inference Client

A C++ command-line interface for Vision Language Model (VLM) inference using OpenAI-compatible APIs.

## Main Project

### C++ VLM Inference Client
A production-ready multimodal inference client supporting text and image prompts with multiple API providers.

**[📖 Full Documentation](OpenAI-completion-client/cpp/Readme.md)**

#### Key Features
- 🔌 Multiple API provider support (OpenAI, Together, vLLM, and more)
- 🖼️ Multimodal capabilities (text + multiple images)
- 🔄 Automatic image preprocessing and resizing
- ⚙️ Configurable detail levels and token limits
- 🌐 Support for local files and image URLs

#### Quick Start
```bash
./vlm-inference-client \
    --prompt "Describe this image" \
    --images image.jpg \
    --model gpt-4o \
    --api_endpoint https://api.openai.com/v1/chat/completions \
    --api_key_env OPENAI_API_KEY
```

## Additional Resources

### Python Reference Implementation
  * [Python client](OpenAI-completion-client/python/Readme.md) - Alternative implementation in Python

### Example Code
  * **Google AI Examples** - Gemini API and Vertex AI examples in [`google/`](google/)
  * **Llama Utilities** - Multimodal utilities in [`llama/`](llama/)

## Documentation

- **[Benchmarks](docs/benchmarks.md)** - VLM evaluation benchmarks and leaderboards
- **[Courses & Tutorials](docs/courses.md)** - Online courses and learning resources
- **[API Services](docs/api-services.md)** - Vision multimodal API providers
- **[Finetuning](docs/finetuning.md)** - Resources for finetuning VLMs
- **[RAG](docs/rag.md)** - Multimodal RAG resources
- **[Inference](docs/inference.md)** - Inference frameworks and tools
- **[Cloud GPU](docs/cloud-gpu.md)** - GPU rental services
- **[Google AI](docs/google.md)** - Google-specific resources (Gemini, Vertex AI)
- **[Llama](docs/llama.md)** - Llama-specific resources
