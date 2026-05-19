"""Composable multi-agent building blocks.

Three small primitives, designed to compose:
    Agent              — single LLM call wrapper. Caller provides an
                         async `call_fn(messages) -> (text, usage)`.
    WriterCriticPair   — writer drafts → critic validates → retry on fail.
    Pipeline           — sequential chain of agents (output → next input).

No framework lock-in: the caller plugs in their own LLM call function, so
this works with OpenAI / Anthropic / Bedrock / Ollama / vLLM equally.

For parallel execution, use `asyncio.gather` directly — no wrapper needed.

Every call is auto-traced via `llm_kit.observability.trace_generation` if
Langfuse env is set; silent no-op otherwise.
"""

from llm_kit.agents.agent import Agent, AgentCallFn, AgentResult
from llm_kit.agents.critic import CriticVerdict, WriterCriticPair
from llm_kit.agents.pipeline import Pipeline

__all__ = [
    "Agent",
    "AgentCallFn",
    "AgentResult",
    "CriticVerdict",
    "Pipeline",
    "WriterCriticPair",
]
