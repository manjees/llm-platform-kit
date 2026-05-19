"""Example — wrap an OpenAI call with Langfuse trace.

Run:
    export OPENAI_API_KEY=...
    export LANGFUSE_PUBLIC_KEY=pk-lf-...
    export LANGFUSE_SECRET_KEY=sk-lf-...
    python examples/01_observability_openai.py

If Langfuse env is missing, the trace is silently skipped and the OpenAI call
still works — that's the design: zero impact on production.
"""

from __future__ import annotations

import asyncio
import os

import httpx

from llm_kit.observability import (
    flush_langfuse,
    is_enabled,
    trace_generation,
)


async def main() -> None:
    api_key = os.environ["OPENAI_API_KEY"]
    model = "gpt-4o-mini"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'pong' in one word."},
    ]

    async with httpx.AsyncClient(
        timeout=20,
        headers={"Authorization": f"Bearer {api_key}"},
    ) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 10,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    answer = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})

    print(f"OpenAI answered: {answer!r}")
    print(f"Langfuse enabled: {is_enabled()}")

    trace_generation(
        name="example.ping",
        model=model,
        input=messages,
        output=answer,
        usage={
            "input": usage.get("prompt_tokens", 0),
            "output": usage.get("completion_tokens", 0),
            "total": usage.get("total_tokens", 0),
        },
        model_parameters={"temperature": 0.0, "max_tokens": 10},
        tags=["example", "ping"],
    )

    flush_langfuse()
    print("Done. Check Langfuse dashboard for 'example.ping' trace.")


if __name__ == "__main__":
    asyncio.run(main())
