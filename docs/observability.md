# Observability — Langfuse integration

Every LLM call should be traceable. This module wraps [Langfuse](https://langfuse.com)
so you can drop traces into any OpenAI / Anthropic / httpx call.

## Why

Plain LLM calls hide three things you need in production:

1. **Cost** — per call, per user, per feature
2. **Latency** — p50/p95/p99 across deployments
3. **What was sent / received** — when a response goes wrong, you need
   the exact prompt and tool outputs to debug

## Setup

```bash
pip install llm-platform-kit[observability]
```

Get keys from `https://cloud.langfuse.com` (or your self-hosted instance) and
set environment variables:

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=https://cloud.langfuse.com   # or .jp / .eu / self-host
```

## Pattern A — manual `trace_generation` (recommended)

Works with any LLM client (OpenAI SDK, Anthropic SDK, raw `httpx`, ...).

```python
from llm_kit.observability import trace_generation

resp = await client.chat.completions.create(model="gpt-4o-mini", messages=msgs)

trace_generation(
    name="feature.subtask",
    model="gpt-4o-mini",
    input=msgs,
    output=resp.choices[0].message.content,
    usage=resp.usage.model_dump(),     # {"prompt_tokens", "completion_tokens", ...}
    model_parameters={"temperature": 0.2, "max_tokens": 200},
    metadata={"user_id": user_id, "tenant": "acme"},
    tags=["beta", "v2"],
)
```

`usage` keys must be `input` / `output` / `total` for Langfuse to compute cost.
Helper for OpenAI:

```python
usage = resp.usage
trace_usage = {
    "input": usage.prompt_tokens,
    "output": usage.completion_tokens,
    "total": usage.total_tokens,
}
```

## Pattern B — AsyncOpenAI drop-in (OpenAI v1.x only)

```python
from langfuse.openai import AsyncOpenAI   # ← one line
client = AsyncOpenAI()
# all subsequent calls auto-trace
```

⚠️ **Known incompatibility**: the drop-in does not work with OpenAI SDK v2.x
(`No module named 'openai.resources.beta.chat'`). Use Pattern A.

## Spans (nesting calls under one trace)

When one user request triggers multiple LLM calls (planner → workers →
synthesizer), wrap them in a single span so the dashboard shows the tree.

```python
from llm_kit.observability import trace_span, trace_generation

span = trace_span(
    name="ticket.triage",
    input={"ticket_id": tid},
    tags=["support"],
)

await call_planner(...)        # trace_generation calls inside
await call_worker(...)
await call_writer(...)

span.update(output={"final": ...})
```

## Flush on exit (short scripts)

Background flush runs every ~10 seconds. CLI scripts that exit immediately
should call `flush_langfuse()`:

```python
from llm_kit.observability import flush_langfuse

async def main():
    ...
    flush_langfuse()
```

Long-lived processes (web servers, cron jobs) don't need this.

## What the dashboard gives you

- **Tracing**: trace → expand → see inputs/outputs/tokens/latency per call
- **Cost dashboard**: daily / weekly / monthly aggregate by model
- **Latency dashboard**: p50/p95/p99 by call name
- **Sessions**: group calls by user_id / session_id
- **Datasets**: connect to the eval module (see [eval.md](eval.md))

## Failure mode

If `LANGFUSE_PUBLIC_KEY` is unset, every helper in this module is a silent
no-op. No exceptions, no log spam, zero production impact. This is intentional
so you can ship code that calls `trace_generation` without forcing every
deployment to wire Langfuse first.

## Real-world numbers (from production agent)

| Metric | Value |
|---|---|
| Daily traces | ~N |
| Cost per call (gpt-4o-mini) | $0.00X |
| p95 latency (multi-step agent) | XXX ms |
| Setup time | < 30 minutes |
