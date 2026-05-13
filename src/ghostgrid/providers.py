"""API request/response handling for LLM and VLM providers."""

import json
import logging
import time
from collections.abc import Generator, Iterable
from typing import Any, cast

import requests

from ghostgrid.image import encode_image, is_url
from ghostgrid.models import Agent, AgentResult, InferenceConfig

logger = logging.getLogger(__name__)

TEXT_ONLY_MULTIMODAL_GUARDS: dict[str, str] = {
    "glm-5.1": "GLM-5.1 is text-only on Z.AI. For image or video inputs, switch to GLM-4.6V.",
}

DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 1.0
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


def validate_multimodal_model(model: str, image_paths: Iterable[str] | None) -> None:
    """Reject known text-only models when image inputs are present."""
    if not image_paths:
        return

    reason = TEXT_ONLY_MULTIMODAL_GUARDS.get(model.strip().lower())
    if reason:
        raise ValueError(reason)


def create_payload(
    prompt: str,
    model: str,
    config: InferenceConfig,
) -> dict:
    """Build an OpenAI-compatible chat-completions payload."""
    validate_multimodal_model(model, config.image_paths)

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "max_tokens": config.max_tokens,
    }
    if config.stream:
        payload["stream"] = True
    for image_path in config.image_paths or []:
        if is_url(image_path) and not config.resize:
            img_block = {
                "type": "image_url",
                "image_url": {"url": image_path, "detail": config.detail},
            }
        else:
            b64 = encode_image(image_path, config.resize, config.target_size)
            img_block = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": config.detail},
            }
        payload["messages"][0]["content"].append(img_block)
    return payload


def create_anthropic_payload(
    prompt: str,
    model: str,
    config: InferenceConfig,
) -> dict:
    """Build an Anthropic Messages API payload with images."""
    validate_multimodal_model(model, config.image_paths)

    content: list[dict] = [{"type": "text", "text": prompt}]
    for image_path in config.image_paths or []:
        if is_url(image_path) and not config.resize:
            img_block = {
                "type": "image",
                "source": {"type": "url", "url": image_path},
            }
        else:
            b64 = encode_image(image_path, config.resize, config.target_size)
            img_block = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64,
                },
            }
        content.append(img_block)
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": config.max_tokens,
        "messages": [{"role": "user", "content": content}],
    }
    if config.stream:
        payload["stream"] = True
    return payload


def _build_retry_headers(api_key: str, provider: str) -> dict:
    """Build auth headers for a provider."""
    if provider == "anthropic":
        return {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


def send_request(  # pylint: disable=too-many-arguments
    api_key: str,
    url: str,
    payload: dict,
    *,
    timeout: int = 120,
    retries: int = DEFAULT_RETRIES,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
) -> dict:
    """POST to an OpenAI-compatible endpoint with retry/backoff. Returns JSON response."""
    headers = _build_retry_headers(api_key, "openai")
    return _request_with_retry(url, headers, payload, timeout, retries, backoff_factor, "API")


def send_anthropic_request(  # pylint: disable=too-many-arguments
    api_key: str,
    url: str,
    payload: dict,
    *,
    timeout: int = 120,
    retries: int = DEFAULT_RETRIES,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
) -> dict:
    """POST to the Anthropic Messages API with retry/backoff. Returns JSON response."""
    headers = _build_retry_headers(api_key, "anthropic")
    return _request_with_retry(url, headers, payload, timeout, retries, backoff_factor, "Anthropic API")


def _request_with_retry(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    url: str,
    headers: dict,
    payload: dict,
    timeout: int,
    retries: int,
    backoff_factor: float,
    label: str,
) -> dict:
    """Send an HTTP POST with exponential backoff on transient failures."""
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if response.status_code == 200:
                return cast(dict, response.json())
            if response.status_code in RETRYABLE_STATUSES and attempt < retries:
                wait = backoff_factor * (2**attempt)
                logger.warning(
                    "%s returned %d, retrying in %.1fs (attempt %d/%d)",
                    label,
                    response.status_code,
                    wait,
                    attempt + 1,
                    retries,
                )
                time.sleep(wait)
                continue
            raise RuntimeError(f"{label} request failed: {response.status_code} {response.text}")
        except requests.exceptions.Timeout as exc:
            last_exc = exc
            if attempt < retries:
                wait = backoff_factor * (2**attempt)
                logger.warning("%s timed out, retrying in %.1fs (attempt %d/%d)", label, wait, attempt + 1, retries)
                time.sleep(wait)
                continue
        except requests.exceptions.ConnectionError as exc:
            last_exc = exc
            if attempt < retries:
                wait = backoff_factor * (2**attempt)
                logger.warning(
                    "%s connection error, retrying in %.1fs (attempt %d/%d)", label, wait, attempt + 1, retries
                )
                time.sleep(wait)
                continue
    raise RuntimeError(f"{label} request failed after {retries + 1} attempts") from last_exc


def stream_request(
    api_key: str,
    url: str,
    payload: dict,
    timeout: int = 120,
) -> Generator[str, None, None]:
    """Stream SSE text chunks from an OpenAI-compatible endpoint."""
    headers = _build_retry_headers(api_key, "openai")
    response = requests.post(url, headers=headers, json=payload, timeout=timeout, stream=True)
    if response.status_code != 200:
        raise RuntimeError(f"API streaming request failed: {response.status_code} {response.text}")
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[6:]
        if data.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                yield content


def stream_anthropic_request(
    api_key: str,
    url: str,
    payload: dict,
    timeout: int = 120,
) -> Generator[str, None, None]:
    """Stream SSE text chunks from the Anthropic Messages API."""
    headers = _build_retry_headers(api_key, "anthropic")
    response = requests.post(url, headers=headers, json=payload, timeout=timeout, stream=True)
    if response.status_code != 200:
        raise RuntimeError(f"Anthropic API streaming request failed: {response.status_code} {response.text}")
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[6:]
        if data.strip() == "[DONE]":
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "content_block_delta":
            text = event.get("delta", {}).get("text", "")
            if text:
                yield text


def normalize_response(response: dict) -> str:
    """Extract text content from any supported provider response shape."""
    try:
        return cast(str, response["choices"][0]["message"]["content"])  # OpenAI-compatible
    except (KeyError, IndexError):
        pass
    try:
        return cast(str, response["content"][0]["text"])  # Anthropic
    except (KeyError, IndexError):
        pass
    try:
        return cast(str, response["candidates"][0]["content"]["parts"][0]["text"])  # Google
    except (KeyError, IndexError):
        pass
    return cast(str, json.dumps(response))


def run_agent(agent: Agent, prompt: str, config: InferenceConfig) -> AgentResult:
    """Execute a single agent call and return an AgentResult."""
    if agent.provider == "anthropic":
        payload = create_anthropic_payload(prompt, agent.model, config)
    else:
        payload = create_payload(prompt, agent.model, config)

    logger.debug("Calling %s/%s (%s)", agent.provider, agent.model, agent.agent_id)
    t0 = time.time()
    try:
        if config.stream:
            content, raw_response = _run_agent_streaming(agent, payload)
        elif agent.provider == "anthropic":
            content = normalize_response(send_anthropic_request(agent.api_key, agent.endpoint, payload))
            raw_response = {}
        else:
            response = send_request(agent.api_key, agent.endpoint, payload)
            content = normalize_response(response)
            raw_response = response
        latency_ms = (time.time() - t0) * 1000
        logger.info("%s/%s → %.0fms, %d chars", agent.provider, agent.model, latency_ms, len(content))
        return AgentResult(
            agent_id=agent.agent_id,
            model=agent.model,
            provider=agent.provider,
            content=content,
            raw_response=raw_response,
            latency_ms=latency_ms,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        latency_ms = (time.time() - t0) * 1000
        logger.error("%s/%s failed: %s", agent.provider, agent.model, exc)
        return AgentResult(
            agent_id=agent.agent_id,
            model=agent.model,
            provider=agent.provider,
            content="",
            raw_response={},
            latency_ms=latency_ms,
            error=str(exc),
        )


def _run_agent_streaming(agent: Agent, payload: dict) -> tuple[str, dict]:
    """Run agent with streaming and accumulate full content."""
    if agent.provider == "anthropic":
        chunks = stream_anthropic_request(agent.api_key, agent.endpoint, payload)
    else:
        chunks = stream_request(agent.api_key, agent.endpoint, payload)
    content_parts: list[str] = []
    for chunk in chunks:
        content_parts.append(chunk)
    content = "".join(content_parts)
    return content, {}
