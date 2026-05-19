# Guards — composable hallucination & quality defense

LLM output is non-deterministic. These guards are small, pure functions
you stack to reject bad responses before they reach users.

## The 6 layers

| # | Layer | Lives in | Helper |
|---|---|---|---|
| 1 | System-prompt rules | your prompt | (`llm_kit.prompts`) |
| 2 | Tool-schema directives | your tool descriptions | (your function-calling JSON) |
| 3 | Forbidden-phrase regex | code | `check_forbidden_phrases` |
| 4 | Length cap | code | `check_length` |
| 5 | Slop / boilerplate regex | code | `check_slop` |
| 6 | Orchestrated retry | code | `with_retry` |

Layers 1–2 live in your prompts. 3–5 are pure helpers. 6 is a thin
"validate, regenerate, retry" harness.

## Layer 3 — forbidden phrases

```python
from llm_kit.guards import check_forbidden_phrases

result = check_forbidden_phrases(
    "We're so sorry but as an AI I can't help.",
    forbidden=["as an AI", "I apologize", "Choaedol"],  # last = brand typo
)
# result.ok = False, result.reason = "forbidden phrases present: ['as an AI']"
```

Use for: brand misspellings, banned apology patterns, leaked internal jargon.

## Layer 4 — length cap

```python
from llm_kit.guards import check_length

check_length(text, max_chars=280)   # tweet length
check_length(text, max_chars=None)  # no cap → always passes
```

## Layer 5 — slop / boilerplate

LLMs love filler ("In conclusion, ...", "I hope this helps!"). Strip with
regex:

```python
from llm_kit.guards import check_slop, DEFAULT_SLOP_PATTERNS

# Defaults cover common English LLM filler
check_slop(text)

# Or pass project-specific patterns
check_slop(text, patterns=(r"우리는 항상", r"저희가 도와드리겠습니다"))
```

The `DEFAULT_SLOP_PATTERNS` covers:
- "however, it's important to note..."
- "in conclusion, ..."
- "i hope this helps"
- "feel free to ask"
- naked superlatives ("definitely", "absolutely")

Extend per project — these are starting points, not law.

## Layer 6 — retry with regeneration

When validation fails, you usually want one retry with a hint, not just
a rejection.

```python
from llm_kit.guards import with_retry, combine_validators, check_length, check_forbidden_phrases

# Compose validators (AND — first failure short-circuits)
validator = combine_validators(
    lambda t: check_length(t, 280),
    lambda t: check_forbidden_phrases(t, ["Choaedol"]),
    lambda t: check_slop(t),
)

# Generator gets the attempt number — use it to tweak the prompt on retry
async def generate(attempt: int) -> str:
    msgs = [system_msg, user_msg]
    if attempt > 0:
        msgs.append({
            "role": "user",
            "content": "Your previous answer was too long or used a banned phrase. "
                       "Try again, under 280 chars, no filler."
        })
    resp = await openai_call(msgs)
    return resp.choices[0].message.content

text, result = await with_retry(generate, validator, max_attempts=2)
if text is None:
    # all attempts failed → fall back to canned response, alert ops, etc.
    text = FALLBACK
```

`result.reason` carries the last failure message — log it so you can mine
patterns ("most retries are 'too long' → tighten length in system prompt").

## Layer 1–2 — your prompts (not in this library)

The deepest layer is what you write into the system prompt and tool
descriptions. The library can't author those for you, but here are the
patterns that worked in production:

System prompt sketch:

```
You answer using ONLY the data the tools return.
- If a tool returns no result, say so honestly. Do NOT invent.
- Do NOT include URLs other than those a tool returned.
- Keep answers concise (target: < 280 chars).
- Banned: filler phrases ("hope this helps", "feel free to ask"),
  invented citations, speculation phrased as fact.
```

Tool description sketch:

```python
{
    "type": "function",
    "function": {
        "name": "search_kb",
        "description": (
            "Search the knowledge base. Returns up to 5 documents with "
            "title/snippet/source. If the result list is empty, the agent "
            "should answer 'I couldn't find that information' — do NOT "
            "fabricate."
        ),
        "parameters": {...}
    }
}
```

Treat those `description` strings as runtime documentation: the LLM reads
them at call time and follows them more reliably than rules buried in the
system prompt.

## Putting it all together

```python
from llm_kit.guards import (
    with_retry, combine_validators,
    check_length, check_forbidden_phrases, check_slop,
)
from llm_kit.observability import trace_generation

validator = combine_validators(
    lambda t: check_length(t, 280),
    lambda t: check_forbidden_phrases(t, BANNED),
    lambda t: check_slop(t),
)

async def generate(attempt: int) -> str:
    msgs = build_messages(query, retry_hint=(attempt > 0))
    text, usage = await call_llm(msgs)
    trace_generation(
        name="answer.attempt",
        model="gpt-4o-mini",
        input=msgs, output=text, usage=usage,
        metadata={"attempt": attempt},
    )
    return text

text, result = await with_retry(generate, validator, max_attempts=2)
```

Five lines of glue, six layers of defense, and full Langfuse trace per
attempt for post-mortem analysis.

## Failure rate (production)

Tracked over 7 iteration rounds on a real agent (`eval` framework):

| Layer | Catches |
|---|---|
| length | rare (~ 2 % retries) |
| forbidden phrases | very rare (Choaedol typo etc.) |
| slop | medium (~ 8 % retries) |
| retry success | most retries succeed in 1 extra attempt |

The point isn't *catching a lot* — it's *catching the things you can't tolerate*
(brand-damaging output, oversized tweets) before they hit users.
