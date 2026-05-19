"""Example — guard a generation with retry.

The "generate" here is a mock that returns different text per attempt.
In your real code, plug in an OpenAI call.
"""

from __future__ import annotations

import asyncio

from llm_kit.guards import (
    check_forbidden_phrases,
    check_length,
    combine_validators,
    with_retry,
)


# Simulated generator — first attempt is too long, second is clean.
ATTEMPTS = [
    "this is a way too long answer that goes on and on and on " * 10,
    "Refunds within 30 days, original packaging.",
]


async def fake_generate(attempt: int) -> str:
    return ATTEMPTS[min(attempt, len(ATTEMPTS) - 1)]


async def main() -> None:
    validator = combine_validators(
        lambda t: check_length(t, 100),
        lambda t: check_forbidden_phrases(t, ["sorry, I don't know"]),
    )
    text, result = await with_retry(fake_generate, validator, max_attempts=2)
    print(f"Final text:  {text!r}")
    print(f"Result:      ok={result.ok}, reason={result.reason}")


if __name__ == "__main__":
    asyncio.run(main())
