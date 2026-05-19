"""Agent — thin wrapper around a single LLM call.

You provide:
    name              — logical identifier (used as trace name)
    model             — model identifier (used for trace + cost attribution)
    system_prompt     — string passed as the system message
    call_fn           — async (messages: list[dict]) -> (text, usage_dict)

The wrapper:
    - prepends the system message
    - calls your `call_fn`
    - records a `trace_generation` to Langfuse (no-op if disabled)
    - returns an `AgentResult` (text + raw usage)

This keeps the library LLM-provider-agnostic. Users plug in whatever HTTP
client they already use (httpx, OpenAI SDK, Anthropic SDK, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from llm_kit.observability import trace_generation


@dataclass
class AgentResult:
    """Outcome of one Agent call."""
    text: str
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None  # Optional: the underlying SDK response (debug aid)


# (messages: list[dict]) -> (text, usage_dict)
AgentCallFn = Callable[[list[dict]], Awaitable[tuple[str, dict[str, int]]]]


class Agent:
    """Single-LLM-call wrapper. Pure: same input → same outgoing message list.

    The whole purpose is to standardize:
      1. system + user message assembly
      2. one observability trace per call
      3. consistent `AgentResult` shape

    For multi-step tool use, build that loop in your own code and call
    `Agent.run(...)` per step. This library does not impose a loop.
    """

    def __init__(
        self,
        *,
        name: str,
        model: str,
        system_prompt: str,
        call_fn: AgentCallFn,
        model_parameters: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self._call_fn = call_fn
        self._model_parameters = model_parameters or {}
        self._tags = list(tags or [])

    async def run(
        self,
        user_message: str,
        *,
        extra_messages: list[dict] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Run one LLM call.

        Args:
            user_message: the user's question / instruction.
            extra_messages: appended after [system, user]. Use for turn-by-turn
                            history or retry hints.
            metadata: free-form key/value attached to the Langfuse trace.

        Returns: AgentResult with text + usage.
        """
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        if extra_messages:
            messages.extend(extra_messages)

        text, usage = await self._call_fn(messages)

        trace_generation(
            name=self.name,
            model=self.model,
            input=messages,
            output=text,
            usage=usage,
            model_parameters=self._model_parameters,
            metadata=metadata,
            tags=self._tags,
        )
        return AgentResult(text=text, usage=usage)
