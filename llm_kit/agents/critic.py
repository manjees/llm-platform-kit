"""WriterCriticPair — the most common "make + validate + retry" pattern.

The writer drafts a response. The critic checks it against criteria. If the
critic rejects, the writer tries again (with the critic's feedback appended
to its messages). Stops on first pass or after `max_attempts`.

The critic itself can be:
  - a pure function (rule-based check)
  - another `Agent` whose output the wrapper parses into a verdict
  - a hybrid (rule-based first, LLM-based fallback)

This wrapper handles the loop; callers pick their own validation strategy.

Returns the writer's accepted text or None plus the final verdict, so
upstream code can fall back gracefully when nothing passes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

from llm_kit.agents.agent import Agent

logger = logging.getLogger(__name__)


@dataclass
class CriticVerdict:
    """Critic decision for one draft.

    `ok=True` accepts. `ok=False` triggers a retry (if attempts remain).
    `feedback` is appended to the writer's next-attempt message list so it
    can correct course.
    """
    ok: bool
    feedback: str


# (draft_text) -> CriticVerdict
Critic = Callable[[str], Awaitable[CriticVerdict] | CriticVerdict]


class WriterCriticPair:
    """Drive a writer Agent + critic until the critic accepts or attempts run out.

    Example:
        writer = Agent(name="tweet.writer", model="gpt-4o-mini", ...)

        def critic(text: str) -> CriticVerdict:
            if len(text) > 280:
                return CriticVerdict(False, "Too long; keep under 280 chars.")
            return CriticVerdict(True, "ok")

        pair = WriterCriticPair(writer=writer, critic=critic, max_attempts=2)
        text, verdict = await pair.run("Draft a launch tweet for our new feature.")

    Note:
        Each retry adds a user message of the form:
            "Your previous draft was rejected. Feedback: {feedback}.
             Please revise and try again."
        Tune the wording by subclassing if your prompt style differs.
    """

    def __init__(
        self,
        *,
        writer: Agent,
        critic: Critic,
        max_attempts: int = 2,
    ):
        self.writer = writer
        self.critic = critic
        self.max_attempts = max(1, int(max_attempts))

    async def run(
        self,
        user_message: str,
        *,
        metadata: dict | None = None,
    ) -> tuple[str | None, CriticVerdict]:
        """Try until accepted. Return (final_text or None, last verdict).

        On total failure (no attempt passes), returns (None, verdict) so the
        caller can decide what to do — fall back to a canned response, escalate
        to a human, etc.
        """
        history: list[dict] = []
        verdict = CriticVerdict(ok=False, feedback="no attempts yet")
        for attempt in range(self.max_attempts):
            attempt_metadata = {
                **(metadata or {}),
                "attempt": attempt,
                "writer_critic_pair": True,
            }
            result = await self.writer.run(
                user_message,
                extra_messages=history,
                metadata=attempt_metadata,
            )
            draft = result.text
            v = self.critic(draft)
            if hasattr(v, "__await__"):
                v = await v
            verdict = v if isinstance(v, CriticVerdict) else CriticVerdict(
                ok=bool(v), feedback=""
            )
            if verdict.ok:
                return draft, verdict
            logger.info(
                f"[llm_kit.agents] writer attempt {attempt} rejected: {verdict.feedback}"
            )
            history.append({
                "role": "user",
                "content": (
                    f"Your previous draft was rejected. Feedback: "
                    f"{verdict.feedback}. Please revise and try again."
                ),
            })
        return None, verdict
