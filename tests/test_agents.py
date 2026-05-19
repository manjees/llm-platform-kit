"""Unit tests for llm_kit.agents — pure async with mock call_fn."""

from __future__ import annotations

import pytest

from llm_kit.agents import (
    Agent,
    CriticVerdict,
    Pipeline,
    WriterCriticPair,
)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_run_returns_call_fn_output():
    captured = {}

    async def call_fn(messages):
        captured["messages"] = messages
        return "hello", {"input": 5, "output": 1, "total": 6}

    a = Agent(
        name="t.agent",
        model="mock",
        system_prompt="You say hello.",
        call_fn=call_fn,
    )
    result = await a.run("Greet me")
    assert result.text == "hello"
    assert result.usage["total"] == 6
    # System + user prepended.
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][0]["content"] == "You say hello."
    assert captured["messages"][1]["role"] == "user"
    assert captured["messages"][1]["content"] == "Greet me"


@pytest.mark.asyncio
async def test_agent_extra_messages_appended():
    captured = {}

    async def call_fn(messages):
        captured["messages"] = messages
        return "ok", {}

    a = Agent(name="t", model="m", system_prompt="sys", call_fn=call_fn)
    extra = [{"role": "user", "content": "follow-up"}]
    await a.run("first", extra_messages=extra)
    assert captured["messages"][-1] == {"role": "user", "content": "follow-up"}
    assert len(captured["messages"]) == 3


# ---------------------------------------------------------------------------
# WriterCriticPair
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_writer_critic_accepts_first():
    async def call_fn(messages):
        return "short", {}

    writer = Agent(name="w", model="m", system_prompt="sys", call_fn=call_fn)
    pair = WriterCriticPair(
        writer=writer,
        critic=lambda t: CriticVerdict(True, "ok"),
        max_attempts=2,
    )
    text, verdict = await pair.run("draft")
    assert text == "short"
    assert verdict.ok


@pytest.mark.asyncio
async def test_writer_critic_retries_then_accepts():
    drafts = iter(["too long indeed", "good"])

    async def call_fn(messages):
        return next(drafts), {}

    writer = Agent(name="w", model="m", system_prompt="sys", call_fn=call_fn)

    def critic(text):
        if len(text) > 5:
            return CriticVerdict(False, "shorter")
        return CriticVerdict(True, "ok")

    pair = WriterCriticPair(writer=writer, critic=critic, max_attempts=2)
    text, verdict = await pair.run("draft")
    assert text == "good"
    assert verdict.ok


@pytest.mark.asyncio
async def test_writer_critic_exhausted_returns_none():
    async def call_fn(messages):
        return "always too long", {}

    writer = Agent(name="w", model="m", system_prompt="sys", call_fn=call_fn)
    pair = WriterCriticPair(
        writer=writer,
        critic=lambda t: CriticVerdict(False, "no good"),
        max_attempts=2,
    )
    text, verdict = await pair.run("draft")
    assert text is None
    assert not verdict.ok


@pytest.mark.asyncio
async def test_writer_critic_supports_async_critic():
    async def acritic(text):
        return CriticVerdict(True, "async ok")

    async def call_fn(messages):
        return "any", {}

    writer = Agent(name="w", model="m", system_prompt="sys", call_fn=call_fn)
    pair = WriterCriticPair(writer=writer, critic=acritic, max_attempts=1)
    text, verdict = await pair.run("draft")
    assert text == "any"
    assert verdict.ok
    assert verdict.feedback == "async ok"


@pytest.mark.asyncio
async def test_writer_critic_passes_feedback_into_next_attempt():
    seen_messages: list[list[dict]] = []
    drafts = iter(["bad", "good"])

    async def call_fn(messages):
        seen_messages.append(list(messages))
        return next(drafts), {}

    writer = Agent(name="w", model="m", system_prompt="sys", call_fn=call_fn)

    def critic(text):
        if text == "bad":
            return CriticVerdict(False, "use different word")
        return CriticVerdict(True, "ok")

    pair = WriterCriticPair(writer=writer, critic=critic, max_attempts=2)
    await pair.run("draft")
    # Second attempt should have an extra user message echoing the feedback.
    assert len(seen_messages) == 2
    assert any(
        "use different word" in m.get("content", "")
        for m in seen_messages[1]
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_chains_outputs():
    async def stage1(messages):
        return "step1-out", {}

    async def stage2(messages):
        # stage2's user message should be stage1's output.
        return f"got:{messages[1]['content']}", {}

    a1 = Agent(name="s1", model="m", system_prompt="s", call_fn=stage1)
    a2 = Agent(name="s2", model="m", system_prompt="s", call_fn=stage2)
    pipe = Pipeline(stages=[a1, a2])
    result = await pipe.run("initial")
    assert result.text == "got:step1-out"


@pytest.mark.asyncio
async def test_pipeline_empty_raises():
    with pytest.raises(ValueError):
        Pipeline(stages=[])
