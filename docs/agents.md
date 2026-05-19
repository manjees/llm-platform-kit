# Agents — composable multi-agent primitives

Three small classes you can mix and match for planner / worker / critic flows.
No framework lock-in: plug in whatever LLM call function you already use.

## When you need this

| Pattern | Why |
|---|---|
| Writer + Critic | Generate, validate, retry on fail — much more reliable than one-shot |
| Planner → Worker → Synthesizer | Break a big task into stages, each stage focused on one job |
| Parallel specialist agents | Concurrent work (use `asyncio.gather`, no wrapper needed) |

If a single Agent call works, just use that. Add Critic / Pipeline only when
the single-shot fails too often or the task naturally has stages.

## The primitives

### `Agent`

Wraps one LLM call. You provide an async `call_fn(messages) -> (text, usage)`.
The wrapper handles the message-list assembly, calls your function, and
records one trace to Langfuse.

```python
from llm_kit.agents import Agent

async def openai_call(messages: list[dict]) -> tuple[str, dict]:
    resp = await client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return resp.choices[0].message.content, {
        "input": resp.usage.prompt_tokens,
        "output": resp.usage.completion_tokens,
        "total": resp.usage.total_tokens,
    }

writer = Agent(
    name="tweet.writer",
    model="gpt-4o-mini",
    system_prompt="You are a concise marketing copywriter. Max 280 chars.",
    call_fn=openai_call,
    model_parameters={"temperature": 0.7},   # surfaces in Langfuse only
    tags=["marketing", "tweets"],
)

result = await writer.run("Announce our new dashboard.")
print(result.text, result.usage)
```

Why not auto-call OpenAI for you? Because every team has different needs:
streaming, retries, special headers, sandbox endpoints, etc. Letting you
own the HTTP call keeps the library tiny and provider-agnostic.

### `WriterCriticPair`

The most common pattern in practice: writer drafts, critic checks, retry
once or twice with critic feedback in the message history.

```python
from llm_kit.agents import WriterCriticPair, CriticVerdict

def critic(text: str) -> CriticVerdict:
    if len(text) > 280:
        return CriticVerdict(False, "Too long; keep under 280 chars.")
    if "TODO" in text:
        return CriticVerdict(False, "Remove TODO marker.")
    return CriticVerdict(True, "ok")

pair = WriterCriticPair(writer=writer, critic=critic, max_attempts=2)
text, verdict = await pair.run("Announce our new dashboard.")
if text is None:
    text = FALLBACK_COPY
    log.warning(f"writer rejected after all attempts: {verdict.feedback}")
```

The critic can be:
- a pure rule function (as above)
- another `Agent` whose output you parse into `CriticVerdict` (LLM-as-judge)
- a hybrid: cheap rule check first, fall back to LLM only on edge cases

On retry, the wrapper appends the critic's feedback to the writer's next
attempt — the model gets a chance to fix its own mistake.

### `Pipeline`

Sequential agents where each stage's text output is the next stage's input.

```python
from llm_kit.agents import Pipeline

planner = Agent(name="plan", system_prompt="Plan the work in 3 numbered steps.", ...)
worker  = Agent(name="execute", system_prompt="Carry out the plan; report findings.", ...)
formatter = Agent(name="format", system_prompt="Turn findings into a Slack post.", ...)

pipe = Pipeline(stages=[planner, worker, formatter])
result = await pipe.run("Summarize last week's reliability metrics.")
```

For branching (e.g. parallel workers + a synthesizer), use `asyncio.gather`
directly:

```python
data, news = await asyncio.gather(
    worker_a.run(query),
    worker_b.run(query),
)
final = await synthesizer.run(f"Data:\n{data.text}\n\nNews:\n{news.text}")
```

A `Parallel` wrapper isn't included because the asyncio idiom is already
concise and more flexible than a class.

## Observability for free

Every `Agent.run(...)` records a `trace_generation` to Langfuse (silent no-op
if env unset). The dashboard shows:

- input/output of every call
- tokens + cost per call
- latency
- tags + metadata for filtering

WriterCriticPair adds `attempt: N` and `writer_critic_pair: true` to the
metadata; Pipeline adds `pipeline_stage: N` and `pipeline_total: M`. You can
filter the dashboard by these to slice retries vs first-pass success rate.

## Failure modes & fallbacks

- `WriterCriticPair.run()` returning `text=None` is a soft fail. Treat it
  like a `429` or `500`: fall back to a canned response, queue for human
  review, etc. Don't crash the request.
- `Pipeline` propagates exceptions from any stage — failures bubble up.
  Wrap in try/except if you need partial-success behavior.
- A long-running pipeline that calls 5+ LLMs per request is expensive.
  Profile with the Langfuse cost dashboard before shipping.

## Common patterns

### LLM-as-judge critic

```python
from llm_kit.eval.scorers import score_llm_judge

async def llm_critic(text: str) -> CriticVerdict:
    score, reason = await score_llm_judge(
        query=original_user_query,
        response_text=text,
        actual_calls=[],
        judge_criteria="Must be friendly, under 280 chars, no placeholder URLs.",
    )
    return CriticVerdict(ok=score >= 0.8, feedback=reason)
```

### Combine guards (cheap) with critic (expensive)

```python
from llm_kit.guards import check_length, check_forbidden_phrases, combine_validators

cheap_check = combine_validators(
    lambda t: check_length(t, 280),
    lambda t: check_forbidden_phrases(t, ["TODO", "lorem"]),
)

def hybrid_critic(text):
    r = cheap_check(text)
    if not r.ok:
        return CriticVerdict(False, r.reason)   # cheap rule failure
    return CriticVerdict(True, "ok")            # skip LLM judge
```

### Multi-stage with intermediate validation

```python
pipe = Pipeline(stages=[planner, worker])
mid = await pipe.run(query)
# Inspect mid.text, branch on it
if "no data" in mid.text.lower():
    return "Sorry, no data available."
final = await formatter.run(mid.text)
```

## Anti-patterns

- ❌ Wrapping every single LLM call in WriterCriticPair "just in case" —
   the retry doubles cost and latency. Use only where validation actually catches things.
- ❌ Putting the critic inside the writer's system prompt as "self-check"
   instructions — models are bad at self-criticism. Separate agent is more reliable.
- ❌ Deep pipelines (5+ stages). Each stage compounds latency and error.
   Combine adjacent stages if their roles overlap.
- ❌ Critics that call the same model as the writer with the same context —
   often agrees with the writer. Use a different model or different framing.
