# OpenAI API-compliant Python multimodal inference client

This project enables multimodal prompts (text and images) using OpenAI's API, allowing users to submit both text inputs and image files directly from the command line. It supports image inputs via URLs or local files, which are automatically encoded as base64 before being sent to the API for processing. Additionally, it is compatible with any LLM framework that allows serving models via an OpenAI-compatible server, such as vLLM.

## Features

- Supports multiple image inputs (local files or url)
- Configurable text prompts
- Model and API endpoint specification
- CLI parameters for detailed image analysis (detail levels)
- Automatically reads OpenAI API key from the environment

## Prerequisites

- Python 3.x
- OpenAI API key (set as an environment variable)
- `requests` library

You can install dependencies using:

```bash
pip install requests
```

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/olibartfast/vision-language-models-playground.git
cd vision-language-models-playground/OpenAI-client/python
```


### Run the Script

You can run the script using the following command:

```bash
python main.py --prompt "your prompt" --images image1.jpg ... imageN.jp  --model <your_service_model_name> --tokens <max_token_output> --resize --size

 <your_desired_W> <your_desired_H>
```

**Arguments:**

- `--prompt` (optional): Text prompt for image analysis (default: `"What's in this image?"`).
- `--images` (required): List of image file paths or URLs.
- `--model` (optional): GPT model to use (default: `gpt-4o-mini`).
- `--url` (optional): API endpoint URL (default: `https://api.openai.com/v1/chat/completions`).
- `--detail` (optional): Detail level for image analysis (default: `low`).
- `--tokens` (optional): Maximum number of tokens in the response (default: `300`).
- `--resize` (optional): Resize images with padding.
- `--size` (optional): Target size for resized images (width height), default: `512`.

### Example Usage

Analyze an image using the script:

```bash
python main.py -p "Describe this image" -i image.jpg -m gpt-4o-mini -t 300 --resize --size 800 600
```

### Environment Variables

- **OPENAI_API_KEY**: The API key for OpenAI must be set as an environment variable.

Example:
```bash
export OPENAI_API_KEY="your_openai_api_key"
```

### Notes

- The program automatically encodes local images as base64 before sending them in the request.
- Output will display the API's JSON response.


### Reference
* https://platform.openai.com/docs/guides/vision
