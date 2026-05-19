# llm-platform-kit

Production-ready building blocks for LLM-powered features. Extracted from a
real production AI agent (a K-POP fandom SNS automation bot, running for 4–6
weeks of cumulative operation) — only the domain-neutral patterns are kept here.

> **Who is this for?** Teams shipping LLM features (chat, search, summarization,
> recommendation, auto-reply) inside a real product.

## What's inside

| Module | Pattern | One-liner |
|---|---|---|
| `llm_kit.observability` | LLM call tracing | Langfuse drop-in + manual `generation` helper |
| `llm_kit.prompts` | Externalized prompts + hot-swap | Decouple prompt files from code; edit in place |
| `llm_kit.eval` | Test set + auto-scoring | YAML eval set × 4 scorers (rule + LLM judge) |
| `llm_kit.rag` | Semantic search (Chroma + embedding) | OpenAI embedding + local Chroma file |
| `llm_kit.guards` | 6-layer hallucination defense | regex + retry + LLM judge integration |

Each module is **independently usable**. Dependencies are also split
(`pip install llm-platform-kit[observability]`).

## Why this exists

Anyone who has shipped an LLM feature to production has needed roughly the
same setup, every time:

1. **How much did that call cost / how slow was it / did the model hallucinate?** → observability
2. **Tweak a prompt without redeploying the whole service** → externalization + hot-swap
3. **Did this change break any of my previous answers?** → eval framework
4. **Search my own data by meaning, not exact keywords** → RAG
5. **Stop the model from making things up** → hallucination guards

Larger frameworks (LangChain / LangGraph) cover all of this, but bring heavy
opinions and a steep learning curve. This library keeps each pattern as a
**small (100–300 LOC) module** you can drop into your existing stack.

## Quick start

```bash
pip install llm-platform-kit[all]
```

### Observability — trace OpenAI calls in 3 lines

```python
import os
from openai import AsyncOpenAI
from llm_kit.observability import trace_generation

os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."

client = AsyncOpenAI()
resp = await client.chat.completions.create(...)

trace_generation(
    name="my_feature.draft",
    model="gpt-4o-mini",
    input=messages,
    output=resp.choices[0].message.content,
    usage=resp.usage.model_dump(),
)
```

### Prompts — externalize

```python
# brain/prompts/customer_support.md lives outside the code repo
from llm_kit.prompts import read_text

system = read_text("prompts", "customer_support.md")
# Edit the file in production → the next call picks up the change.
```

### Eval — assertion-style scoring

```yaml
# evals/customer_support.yaml
dataset_name: "cs_v1"
items:
  - id: ev_refund_policy
    query: "Tell me about your refund policy."
    expected_tools: [search_kb]
    forbidden_phrases: ["I'm sorry, I don't know"]
    judge_criteria: "Must mention refund-eligible conditions + time window."
    max_chars: 500
```

```python
from llm_kit.eval import run_eval

async def my_agent(query: str) -> dict:
    # ... your agent call ...
    return {"text": response, "tools_called": [...]}

summary = await run_eval("evals/customer_support.yaml", agent_fn=my_agent)
print(summary["scores"])  # {'length': 1.0, 'hallucination': 1.0, ...}
```

### RAG — semantic search

```python
from llm_kit.rag import RAGCollection

rag = RAGCollection("knowledge_base")
await rag.upsert(
    ids=["doc1", "doc2"],
    texts=["Refund policy ...", "Payment methods ..."],
    metadatas=[{"source": "policy"}, {"source": "policy"}],
)

hits = await rag.search("how do I get my money back?", top_k=3)
# → [{"text": "Refund policy ...", "similarity": 0.82, ...}]
```

## Real-world track record

| Metric | Value |
|---|---|
| Production runtime | 4–6 weeks cumulative (and counting) |
| Daily LLM calls | ~N |
| Avg cost per call | $0.00X (gpt-4o-mini) |
| Eval composite score | 0.94 / 1.0 (13 test cases × 4 scorers) |
| Iteration tracking | 7 rounds, 1 regression caught & rolled back |

## Design principles

1. **Each module stands alone** — adopt just one if that's what you need
2. **Minimal hard dependencies** — no framework lock-in
3. **No silent fallbacks** — missing config → fail fast (no debugging a happy
   path that's actually broken)
4. **Production-tested** — every design choice is the result of an actual
   incident in operation

## Per-module docs

- [Observability](docs/observability.md) — Langfuse integration + manual trace
- [Prompts](docs/prompts.md) — externalization + hot-swap + fail-fast
- [Eval](docs/eval.md) — 4 scorers + CI integration
- [RAG](docs/rag.md) — Chroma + embedding pipeline
- [Guards](docs/guards.md) — 6-layer hallucination defense

## License

MIT
