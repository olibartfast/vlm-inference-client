"""Tests for provider payload construction."""

import pytest

from ghostgrid.models import InferenceConfig
from ghostgrid.providers import create_payload


def test_create_payload_supports_text_only_requests():
    """Text-only runs should emit a valid chat payload with no image blocks."""
    payload = create_payload(
        prompt="Explain mixture-of-experts routing.",
        model="nemotron",
        config=InferenceConfig(image_paths=None, detail="low", max_tokens=200, resize=False, target_size=(512, 512)),
    )

    assert payload["model"] == "nemotron"
    assert payload["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Explain mixture-of-experts routing."}]}
    ]


def test_create_payload_rejects_text_only_model_for_images():
    """Known text-only models should fail fast when image inputs are supplied."""
    with pytest.raises(ValueError, match="GLM-5.1 is text-only"):
        create_payload(
            prompt="Describe this image.",
            model="GLM-5.1",
            config=InferenceConfig(
                image_paths=["example.jpg"],
                detail="low",
                max_tokens=200,
                resize=False,
                target_size=(512, 512),
            ),
        )
