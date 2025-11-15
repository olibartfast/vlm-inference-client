# VLM Inference Client

A production-ready C++ command-line interface for Vision Language Model (VLM) inference using OpenAI-compatible APIs. Send multimodal prompts (text and images) to various LLM providers with support for local images and URLs.

## Key Features

- **Multiple API Provider Support**:  Compatible with various providers (generally should work with any provider compatible with the OpenAI Chat API) including:
  - OpenAI
  - Together
  - vLLM
- **Multimodal Capabilities**: 
  - Submit text alongside multiple images
  - Support for both local image files and image URLs
  - Automatic image preprocessing (resizing while maintaining aspect ratio)
- **Image Processing Options**:
  - Customizable target image size
  - Multiple detail levels for image analysis (low, auto, high)
  - Automatic square padding for non-square images
- **API Configuration**:
  - Environment variable integration for API keys and endpoints
  - Custom endpoint URL support
  - Configurable maximum token limit for responses

## Building

### Dependencies

- C++17 or later
- CMake 3.10+
- CURL (for HTTP requests)
- OpenSSL
- OpenCV (image processing)

**Automatically fetched via CMake:**
- nlohmann/json (JSON handling)
- cxxopts (command-line argument parsing)
- cpp-base64 (Base64 encoding)

### Build Instructions

```bash
mkdir build && cd build
cmake ..
make
```

The binary will be created as `vlm-inference-client`.

## Environment Variables

Set up your API credentials using environment variables and pass it as cli input.

## Usage

Basic command structure:

```bash
./vlm-inference-client \
    --prompt <text_prompt> \
    --images <image_paths...> \
    --model <model_name> \
    --api_endpoint <api_provider_endpoint> \
    --api_key_env <api_provider_key_env_var> \
    [optional parameters]
```

### Command Line Options

- `-p, --prompt`: Text prompt for image analysis (required)
- `-i, --images`: One or more image file paths or URLs (required)
- `-m, --model`: Model name to use (e.g., `gpt-4o`, `llama-3.2-90b-vision-instruct`)
- `-e, --api_endpoint`: API endpoint URL (required)
- `-a, --api_key_env`: Environment variable name containing API key (required)
- `-r, --provider`: API provider name (optional)
- `-d, --detail`: Image detail level: `low`, `auto`, or `high` [default: `low`]
- `-t, --tokens`: Maximum tokens for response [default: `300`]
- `-s, --size`: Target image size for encoding in pixels [default: `512`]
- `-h, --help`: Print usage information

### Example Commands

1. **OpenAI GPT-4o with local images:**
```bash
export OPENAI_API_KEY="your-api-key-here"

./vlm-inference-client \
    --prompt "Compare these images" \
    --images image1.jpg image2.jpg \
    --model gpt-4o \
    --api_endpoint https://api.openai.com/v1/chat/completions \
    --api_key_env OPENAI_API_KEY \
    --detail low \
    --tokens 100
```

2. **Together AI with Llama Vision:**
```bash
export TOGETHER_API_KEY="your-api-key-here"

./vlm-inference-client \
    --prompt "Describe what you see" \
    --images photo.jpg \
    --model meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo \
    --api_endpoint https://api.together.xyz/v1/chat/completions \
    --api_key_env TOGETHER_API_KEY
```

3. **Using image URLs:**
```bash
./vlm-inference-client \
    --prompt "What is in this image?" \
    --images https://example.com/image.jpg \
    --model gpt-4o-mini \
    --api_endpoint https://api.openai.com/v1/chat/completions \
    --api_key_env OPENAI_API_KEY
```

4. **vLLM local server:**
```bash
./vlm-inference-client \
    --prompt "Analyze this" \
    --images input.jpg \
    --model llava-v1.5-7b \
    --api_endpoint http://localhost:8000/v1/chat/completions \
    --api_key_env VLLM_API_KEY
```

## Image Processing Details

- Images are automatically resized to maintain aspect ratio
- Non-square images are padded with black borders to create square output
- Final images are encoded as base64 JPEG before API submission
- URLs are passed directly to the API without modification

