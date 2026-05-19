"""llm-platform-kit — production-ready building blocks for LLM features.

Modules:
    observability  — Langfuse trace + cost / latency
    prompts        — externalized prompt files + hot-swap
    eval           — YAML eval set + 4 scorers
    rag            — Chroma + OpenAI embedding
    guards         — composable hallucination / quality guards
    agents         — Agent + WriterCriticPair + Pipeline composables

Each module is independent — use what you need.
"""

__version__ = "0.3.0"
