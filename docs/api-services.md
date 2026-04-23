# LLM and Multimodal API Services

A tentative list of API providers for text-only LLMs and multimodal models:

* [Anthropic Claude Vision](https://docs.anthropic.com/en/docs/build-with-claude/vision)
* [OpenAI Vision](https://platform.openai.com/docs/guides/vision)
  * [OpenAI Multimodal Cookbook](https://cookbook.openai.com/topic/multimodal)
* [Google Cloud Vision](https://cloud.google.com/vision?hl=en)
* [Azure AI Vision](https://azure.microsoft.com/en-us/products/ai-services/ai-vision)
* [AWS Rekognition](https://aws.amazon.com/rekognition/)
* [Stability AI](https://platform.stability.ai/docs/api-reference)
* [Together AI](https://api.together.ai)
* [Cerebras Cloud](https://cloud.cerebras.ai)
* [OpenRouter](https://openrouter.ai/docs/quickstart)
* [Z.AI Open Platform](https://docs.z.ai/)

Text-first model example used in this repo:

* [Together AI: NVIDIA Nemotron 3 Super 120B A12B FP8](https://api.together.ai/models/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8)
* [Together AI: Z.AI GLM-5.1](https://api.together.ai/models/zai-org/GLM-5.1)

Multimodal examples relevant to this repo:

* OpenRouter for `moonshotai/kimi-k2.6`
* Z.AI for `glm-4.6v`

Important model split:

* `GLM-5.1` is text-only on Z.AI and Together AI
* `GLM-4.6V` is the GLM-family multimodal model to use for images
