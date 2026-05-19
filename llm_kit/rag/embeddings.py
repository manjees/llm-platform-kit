"""OpenAI embedding helpers — batched calls, sensible defaults.

Env:
    OPENAI_API_KEY     required
    EMBED_MODEL        default "text-embedding-3-small" (1536 dim, cheap)

Returns None on failure — callers handle gracefully (best-effort).
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

# Default embedding model — small is cheap ($0.02/1M tokens) and good enough
# for most retrieval tasks. Switch to text-embedding-3-large for higher
# recall on complex queries.
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_EMBED_DIM = 1536


def _model() -> str:
    return os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL).strip() or DEFAULT_EMBED_MODEL


async def embed_texts(texts: list[str]) -> list[list[float]] | None:
    """Batch embed a list of texts. Returns list of 1536-dim vectors or None.

    OpenAI's embedding endpoint accepts up to 2048 inputs per call. For
    safety, this function does not split — keep your batches below 1000.
    """
    if not texts:
        return []
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        logger.warning("[llm_kit.rag] OPENAI_API_KEY not set — embed skipped")
        return None
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {"model": _model(), "input": texts}
    try:
        async with httpx.AsyncClient(timeout=30, headers=headers) as client:
            r = await client.post(
                "https://api.openai.com/v1/embeddings", json=body
            )
            r.raise_for_status()
            data = r.json()
        return [row["embedding"] for row in data.get("data", [])]
    except Exception as e:
        logger.warning(f"[llm_kit.rag] OpenAI embed failed: {e}")
        return None


async def embed_one(text: str) -> list[float] | None:
    """Embed a single text. Returns None on failure."""
    vectors = await embed_texts([text])
    if not vectors:
        return None
    return vectors[0]
