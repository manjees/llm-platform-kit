"""Langfuse-based LLM observability — singleton client + trace helpers.

Setup (env):
    LANGFUSE_PUBLIC_KEY   = pk-lf-...
    LANGFUSE_SECRET_KEY   = sk-lf-...
    LANGFUSE_HOST         = https://cloud.langfuse.com  (or jp/eu/self-host)
                            (also accepts LANGFUSE_BASE_URL — legacy SDK env)

Usage patterns:

1. AsyncOpenAI SDK auto-trace (drop-in, OpenAI v1.x only):
       from langfuse.openai import AsyncOpenAI
       # all subsequent calls are traced automatically

2. Manual generation tracking (recommended — works with any SDK or httpx):
       from llm_kit.observability import trace_generation

       resp = await client.chat.completions.create(...)
       trace_generation(
           name="my_feature.draft",
           model="gpt-4o-mini",
           input=messages,
           output=resp.choices[0].message.content,
           usage=resp.usage.model_dump() if resp.usage else None,
       )

3. Long-lived process (cron / server): no manual flush needed.
   Short-lived script: call `flush_langfuse()` before exit.

If env is missing or the langfuse SDK is unavailable, all helpers no-op
silently — zero impact on production.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


def _env_present() -> bool:
    return bool(
        os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
        and os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    )


@lru_cache(maxsize=1)
def get_langfuse() -> Any | None:
    """Singleton Langfuse client. Returns None if env missing or SDK unavailable.

    Callers should always handle the None case (best-effort observability).
    """
    if not _env_present():
        logger.debug("[llm_kit.obs] LANGFUSE env not set — observability disabled")
        return None
    try:
        from langfuse import Langfuse
    except ImportError as e:
        logger.warning(
            f"[llm_kit.obs] langfuse SDK not installed ({e}). "
            "Install: pip install llm-platform-kit[observability]"
        )
        return None
    # Support both LANGFUSE_HOST (SDK standard) and LANGFUSE_BASE_URL (legacy).
    host = (
        os.getenv("LANGFUSE_HOST", "").strip()
        or os.getenv("LANGFUSE_BASE_URL", "").strip()
        or "https://cloud.langfuse.com"
    )
    try:
        client = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"].strip(),
            secret_key=os.environ["LANGFUSE_SECRET_KEY"].strip(),
            host=host,
        )
    except Exception as e:
        logger.warning(f"[llm_kit.obs] Langfuse client init failed: {e}")
        return None
    # Verify credentials early — catches typos / wrong project mapping.
    try:
        if hasattr(client, "auth_check") and not client.auth_check():
            logger.warning(
                f"[llm_kit.obs] Langfuse auth_check failed (host={host}). "
                "Check public/secret key + project mapping."
            )
    except Exception as e:
        logger.debug(f"[llm_kit.obs] auth_check skipped: {e}")
    logger.info(f"[llm_kit.obs] Langfuse client initialized (host={host})")
    return client


def is_enabled() -> bool:
    """Quick check: is Langfuse observability active?"""
    return get_langfuse() is not None


def trace_generation(
    *,
    name: str,
    model: str,
    input: Any,
    output: Any,
    usage: dict[str, int] | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    model_parameters: dict[str, Any] | None = None,
    level: str = "DEFAULT",
    status_message: str | None = None,
) -> None:
    """Record one LLM generation to Langfuse. No-op if disabled.

    Args:
        name: Logical name for this LLM call — used to group/filter in
              the dashboard. Examples: "agent.openai.step",
              "draft_tweet.attempt_0", "memory.extract_user_rules".
        model: Model identifier (e.g. "gpt-4o-mini",
               "claude-3-5-sonnet-20241022").
        input: Messages / prompt sent to the model (list[dict] or str).
        output: Model's text or structured response.
        usage: {"input": int, "output": int, "total": int} prompt/completion
               tokens. Most SDKs return `prompt_tokens` / `completion_tokens` —
               map yourself.
        metadata: Free-form key/value for downstream filtering/grouping.
        tags: list[str] — appears as colored chips in the dashboard.
        model_parameters: {"temperature": ..., "max_tokens": ...} (optional).
        level: "DEFAULT" | "DEBUG" | "WARNING" | "ERROR" — severity filter.
        status_message: Optional string for ERROR level (exception message).
    """
    lf = get_langfuse()
    if lf is None:
        return
    try:
        lf.generation(
            name=name,
            model=model,
            input=input,
            output=output,
            usage=usage,
            metadata=metadata,
            tags=tags,
            model_parameters=model_parameters,
            level=level,
            status_message=status_message,
        )
    except Exception as e:
        logger.debug(f"[llm_kit.obs] generation record failed: {e}")


def trace_span(
    *,
    name: str,
    input: Any | None = None,
    output: Any | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Any | None:
    """Create a top-level trace span — parent of multiple generations.

    Useful when one user request spans multiple LLM calls + tool calls.
    Returns the span object — call `.generation(...)` on it or finalize via
    `.update(output=...)`.

    Returns None if observability is disabled.
    """
    lf = get_langfuse()
    if lf is None:
        return None
    try:
        return lf.trace(
            name=name,
            input=input,
            output=output,
            metadata=metadata,
            tags=tags,
        )
    except Exception as e:
        logger.debug(f"[llm_kit.obs] trace creation failed: {e}")
        return None


def flush_langfuse() -> None:
    """Force-flush queued events to Langfuse cloud. No-op if disabled.

    Call this before short-lived scripts exit (e.g. a CLI eval runner).
    Long-running processes (servers, cron) don't need this — a background
    thread flushes every ~10 seconds.
    """
    lf = get_langfuse()
    if lf is None:
        return
    try:
        lf.flush()
    except Exception as e:
        logger.debug(f"[llm_kit.obs] flush failed: {e}")
