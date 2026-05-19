# Eval — YAML test sets + 4 scorers

Unit tests for LLM features. Catch regressions before they hit production.

## Why

A prompt tweak that fixes one test case often breaks two others. Without
an eval framework you find out from users. With one:

- Every PR runs the test set
- Score drops trigger a CI fail (configurable threshold)
- The exact answer the model gave is stored — debug from the dashboard

## Setup

```bash
pip install llm-platform-kit         # core
# OPENAI_API_KEY required for the LLM judge scorer
```

## Test set format (YAML)

```yaml
dataset_name: "customer_support_v1"

items:
  - id: ev_refund_policy
    query: "Tell me about your refund policy."
    expected_tools: [search_kb]
    forbidden_phrases:
      - "I don't know"
      - "as an AI"
    judge_criteria: |
      Pass if the answer mentions:
      - refund window (30 days)
      - condition (original packaging)
      Fail if it invents policies not present in the tool result.
    max_chars: 500

  - id: ev_decline_unknown
    query: "What's my bank balance?"
    expected_tools: []        # empty = model should answer directly
    forbidden_phrases: ["$"]
    judge_criteria: "Pass if politely declines (no banking integration)."
    max_chars: 200
```

## Run

Your agent is a black box. Pass `agent_fn(query: str) -> dict`:

```python
import asyncio
from llm_kit.eval import run_eval

async def my_agent(query: str) -> dict:
    # ... your real agent code ...
    return {
        "text": answer,
        "tools_called": [
            {"tool": "search_kb", "input": {"q": query}, "output": [...]},
        ],
    }

summary = asyncio.run(
    run_eval("evals/customer_support.yaml", agent_fn=my_agent)
)
print(summary["scores"])
# {'length': 1.0, 'hallucination': 1.0, 'tools_used': 0.93, 'llm_judge': 0.88}
```

## The 4 scorers

### `score_length`
1.0 if `len(response) <= max_chars`, else 0.0. None disables.

### `score_forbidden_phrases`
0.0 if any forbidden substring appears, else 1.0. Use for typos / banned
phrases / brand misspellings.

### `score_tools_used`
Fraction of `expected_tools` that were actually dispatched. Special case:
`expected_tools=[]` means "no tools should be called" — 1.0 if `tools_called`
is empty, 0.0 otherwise. Useful for "agent should answer directly" tests.

### `score_llm_judge`
Sends the query + agent answer + tool outputs + criteria to a small LLM
(default `gpt-4o-mini`) which returns a JSON `{score, reasoning}`.

Key point: the judge sees **tool outputs**, not just tool names. This lets
it verify factual fidelity — "the agent said the refund window is 60 days
but the tool returned 30 days → score 0.3".

Tool outputs are truncated to 5000 chars each to keep the judge's prompt
manageable.

## CI integration

```yaml
# .github/workflows/eval.yml
on:
  pull_request:
    paths: ['agent/**', 'prompts/**']

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e . pyyaml
      - env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: python -m my_app.run_eval
      - run: python scripts/threshold.py --min 0.85
```

A simple threshold script:

```python
# scripts/threshold.py
import json, sys, argparse

p = argparse.ArgumentParser()
p.add_argument("--min", type=float, required=True)
args = p.parse_args()

summary = json.load(sys.stdin)
worst = min(summary["scores"].values())
if worst < args.min:
    print(f"❌ regression: lowest scorer={worst:.3f} below threshold {args.min}")
    sys.exit(1)
print(f"✓ all scorers >= {args.min} (lowest={worst:.3f})")
```

## Fast local iteration

LLM judge calls cost money + take time. For smoke tests:

```python
summary = await run_eval("...", agent_fn=my_agent, skip_llm_judge=True)
```

Rule-based scorers still run. Catches most regressions in < 1 second.

## Real-world trajectory

Production agent, 7 rounds of changes tracked via eval:

| Run | llm_judge | Change |
|---|---|---|
| 1 (baseline) | 0.458 | — |
| 2 | 0.517 | new tool added |
| 3 | 0.875 | judge given tool outputs (not just names) |
| 4 | 0.958 | judge criteria examples expanded |
| 6 | 0.900 | "no filler" guideline added |
| 7 | **0.883 ⚠️** | temperature 0.1 experiment → **caught, rolled back** |
| 8 | 0.94 | stable |

Run 7 is the eval framework earning its keep: a "small tuning" silently
hurt several cases; rolled back before merge.

## Writing good `judge_criteria`

❌ Too vague: `"Answer the question well."`
✅ Specific: `"Pass if the answer cites refund window AND condition. Fail if either is missing or invented."`

❌ Single trait: `"Should be friendly."`
✅ Trait + constraint: `"Friendly tone, but factual — must use tool result numbers, not estimates."`

Treat the criteria like a code review checklist for the LLM judge.

## Adding new scorers

Drop a function returning `tuple[float, str]` and call it from your own
wrapper around `run_eval`:

```python
from llm_kit.eval import run_eval

async def my_run(...):
    summary = await run_eval(...)
    for item in summary["items"]:
        # add custom scoring
        my_score, my_reason = await my_custom_scorer(item["response_text"])
        item["scores"]["custom"] = my_score
    return summary
```

The four built-ins are enough for ~80% of cases — only add domain-specific
scorers when you see a clear gap.
