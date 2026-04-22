"""Tests for provider payload construction."""

from ghostgrid.providers import create_payload


def test_create_payload_supports_text_only_requests():
    """Text-only runs should emit a valid chat payload with no image blocks."""
    payload = create_payload(
        prompt="Explain mixture-of-experts routing.",
        image_paths=None,
        model="nemotron",
        detail="low",
        max_tokens=200,
    )

    assert payload["model"] == "nemotron"
    assert payload["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Explain mixture-of-experts routing."}]}
    ]
