"""Embedding helpers with optional deterministic fallback for tests."""

from __future__ import annotations

import hashlib
import time
from typing import List

from .config import get_settings

settings = get_settings()

if not settings.use_fake_embeddings:
    try:  # pragma: no cover - network clients are exercised via integration
        from openai import OpenAI

        _client = OpenAI(api_key=settings.openai_api_key or None)
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("openai dependency missing") from exc
else:
    _client = None


_FAKE_DIM = 64


def _fake_embed(text: str) -> List[float]:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    floats = []
    for i in range(0, len(digest), 8):
        chunk = digest[i : i + 8]
        val = int(chunk, 16)
        floats.append((val % 1000) / 1000.0)
    # Repeat / trim to requested dim
    while len(floats) < _FAKE_DIM:
        floats.extend(floats)
    return floats[:_FAKE_DIM]


def embed_text(text: str) -> List[float]:
    """Generate an embedding for the supplied text."""

    if not text:
        raise ValueError("Cannot embed empty text")

    if settings.use_fake_embeddings:
        return _fake_embed(text)

    def _call():
        response = _client.embeddings.create(model=settings.embedding_model_name, input=[text])
        return response.data[0].embedding

    # simple retry/backoff
    delay = 0.5
    for _ in range(3):
        try:
            return _call()
        except Exception as exc:  # pragma: no cover - network path
            last_exc = exc
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"Failed to embed after retries: {last_exc}")
