import os
import requests
import base64
import argparse
import json
from PIL import Image, ImageOps
import io

# Function to get the OpenAI API Key from environment variable
def get_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    return api_key

# Function to resize image with padding
def resize_with_padding(image, target_size=(512, 512)):
    # Open image from file-like object or path
    if isinstance(image, str) and not is_url(image):
        img = Image.open(image)
    elif isinstance(image, bytes):
        img = Image.open(io.BytesIO(image))
    else:
        raise ValueError("Unsupported image input")

    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Calculate padding
    img_ratio = img.width / img.height
    target_ratio = target_size[0] / target_size[1]
    
    if img_ratio > target_ratio:
        # Image is wider than target, pad height
        new_width = target_size[0]
        new_height = int(target_size[0] / img_ratio)
    else:
        # Image is taller than target, pad width
        new_height = target_size[1]
        new_width = int(target_size[1] * img_ratio)

    # Resize image maintaining aspect ratio
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Add padding to reach target size
    padding_color = (0, 0, 0)  # Black padding, change as needed
    img_padded = ImageOps.pad(img, target_size, color=padding_color, centering=(0.5, 0.5))
    
    # Convert to bytes for encoding
    img_byte_arr = io.BytesIO()
    img_padded.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

# Function to encode an image in Base64
def encode_image(image_path, resize=False, target_size=(512, 512)):
    if resize:
        if is_url(image_path):
            # Download URL image first
            response = requests.get(image_path)
            img_bytes = resize_with_padding(response.content, target_size)
        else:
            img_bytes = resize_with_padding(image_path, target_size)
        return base64.b64encode(img_bytes).decode("utf-8")
    else:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

# Function to check if a path is a URL
def is_url(image_path):
    return image_path.startswith("http://") or image_path.startswith("https://")

# Function to create the payload for the OpenAI API
def create_payload(prompt, image_paths, model, detail, max_tokens, resize=False, target_size=(512, 512)):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        "max_tokens": max_tokens
    }

    # Add images to the payload
    for image_path in image_paths:
        if is_url(image_path) and not resize:
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": image_path,
                    "detail": detail
                }
            }
        else:
            base64_image = encode_image(image_path, resize, target_size)
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": detail
                }
            }
        payload["messages"][0]["content"].append(image_content)

    return payload

# Function to send the request to the OpenAI API
def send_request(api_key, url, payload):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        raise RuntimeError(f"API request failed: {response.status_code} {response.text}")

    return response.json()

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Analyze images using OpenAI's API with optional resizing")
    parser.add_argument("--prompt", "-p", type=str, default="What's in this image?", help="Text prompt for image analysis")
    parser.add_argument("--images", "-i", type=str, nargs="+", required=True, help="Image file paths or URLs")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o-mini", help="Model name")
    parser.add_argument("--url", "-u", type=str, default="https://api.openai.com/v1/chat/completions", help="API endpoint URL")
    parser.add_argument("--detail", "-d", type=str, default="low", help="Image detail level (auto, low, high)")
    parser.add_argument("--tokens", "-t", type=int, default=300, help="Max tokens for response")
    parser.add_argument("--resize", "-r", action="store_true", help="Resize images with padding")
    parser.add_argument("--size", "-s", type=int, nargs=2, default=[512, 512], help="Target size for resized images (width height)")

    args = parser.parse_args()

    try:
        # Get the API key from environment variable
        api_key = get_api_key()

        # Create the payload for the request
        payload = create_payload(args.prompt, args.images, args.model, args.detail, args.tokens, args.resize, tuple(args.size))

        # Send the request to OpenAI API
        response = send_request(api_key, args.url, payload)

        # Print the response
        print(json.dumps(response, indent=2))

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
