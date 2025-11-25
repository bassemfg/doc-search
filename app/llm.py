"""LLM helper with retry/backoff for RAG answers."""

from __future__ import annotations

import time
from typing import List, Dict, Any, Optional

from .config import get_settings

settings = get_settings()

try:  # pragma: no cover - optional
    from openai import OpenAI

    _client = OpenAI(api_key=settings.openai_api_key or None)
except Exception:  # pragma: no cover
    _client = None


def _retry(fn, attempts: int = 3, backoff: float = 0.5):
    last_exc = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - network path
            last_exc = exc
            time.sleep(backoff * (2 ** i))
    if last_exc:
        raise last_exc
    raise RuntimeError("retry failed")


def _build_prompt(query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    context_lines = []
    for doc in docs:
        doc_id = doc.get("_id")
        title = doc.get("title") or doc.get("metadata", {}).get("title", "(untitled)")
        body = doc.get("content") or doc.get("plot") or ""
        snippet = (body[:400] + "...") if len(body) > 400 else body
        context_lines.append(f"[{doc_id}] {title}: {snippet}")

    system = (
        "You are a concise assistant. Answer using ONLY the provided documents. "
        "Cite sources by including their IDs in square brackets, e.g., [<doc_id>]. "
        "If unsure, say you don't know."
    )
    user = (
        f"Query: {query}\n\n"
        "Documents:\n" + "\n\n".join(context_lines)
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def generate_answer(query: str, docs: List[Dict[str, Any]]) -> str:
    """Call the chat model to build a cited answer. Falls back to extractive summary."""

    if not docs:
        return "No documents found to answer this query."

    if _client is None or not settings.openai_api_key:
        # Fallback: simple extractive summary with citations
        parts = []
        for doc in docs[:3]:
            doc_id = doc.get("_id")
            title = doc.get("title") or doc.get("metadata", {}).get("title", "(untitled)")
            body = doc.get("content") or doc.get("plot") or ""
            snippet = (body[:300] + "...") if len(body) > 300 else body
            parts.append(f"[{doc_id}] {title}: {snippet}")
        return "\n".join(parts)

    messages = _build_prompt(query, docs[:5])

    def _call():
        resp = _client.chat.completions.create(
            model=settings.llm_model_name,
            messages=messages,
            temperature=0.2,
        )
        return resp.choices[0].message.content

    try:
        return _retry(_call)
    except Exception:
        # Fall back to extractive if LLM call fails
        parts = []
        for doc in docs[:3]:
            doc_id = doc.get("_id")
            title = doc.get("title") or doc.get("metadata", {}).get("title", "(untitled)")
            body = doc.get("content") or doc.get("plot") or ""
            snippet = (body[:300] + "...") if len(body) > 300 else body
            parts.append(f"[{doc_id}] {title}: {snippet}")
        return "\n".join(parts)

