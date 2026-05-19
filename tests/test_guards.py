"""Unit tests for llm_kit.guards — pure functions, easy to lock down."""

from __future__ import annotations

import pytest

from llm_kit.guards import (
    GuardResult,
    check_forbidden_phrases,
    check_length,
    check_pattern,
    check_slop,
    combine_validators,
    with_retry,
)


def test_check_length_passes_under_cap():
    r = check_length("hello", 10)
    assert r.ok


def test_check_length_fails_over_cap():
    r = check_length("hello world", 5)
    assert not r.ok


def test_check_length_none_disables():
    r = check_length("anything", None)
    assert r.ok


def test_check_forbidden_phrases_clean():
    r = check_forbidden_phrases("good text", ["bad", "evil"])
    assert r.ok


def test_check_forbidden_phrases_hits():
    r = check_forbidden_phrases("contains bad word", ["bad", "evil"])
    assert not r.ok
    assert "bad" in r.reason


def test_check_slop_default_patterns():
    r = check_slop("In conclusion, this is great.")
    assert not r.ok


def test_check_slop_passes_clean_text():
    r = check_slop("Refunds within 30 days.")
    assert r.ok


def test_check_pattern_must_match():
    r = check_pattern("Order #12345 confirmed", r"#\d+", must_match=True)
    assert r.ok


def test_check_pattern_must_not_match():
    r = check_pattern("contains [URL]", r"\[URL\]", must_match=False)
    assert not r.ok


def test_combine_validators_short_circuit():
    v = combine_validators(
        lambda t: check_length(t, 5),
        lambda t: check_forbidden_phrases(t, ["bad"]),
    )
    # Fails on length first.
    r = v("hello world bad text")
    assert not r.ok
    assert "too long" in r.reason


@pytest.mark.asyncio
async def test_with_retry_succeeds_first_attempt():
    async def gen(attempt: int) -> str:
        return "ok"

    text, result = await with_retry(
        gen, lambda t: GuardResult(True, "always pass"), max_attempts=2
    )
    assert text == "ok"
    assert result.ok


@pytest.mark.asyncio
async def test_with_retry_recovers_on_second():
    attempts: list[int] = []

    async def gen(attempt: int) -> str:
        attempts.append(attempt)
        return "long" if attempt == 0 else "ok"

    text, result = await with_retry(
        gen,
        lambda t: check_length(t, 2),
        max_attempts=2,
    )
    assert text == "ok"
    assert result.ok
    assert attempts == [0, 1]


@pytest.mark.asyncio
async def test_with_retry_exhausted():
    async def gen(attempt: int) -> str:
        return "way too long"

    text, result = await with_retry(
        gen,
        lambda t: check_length(t, 3),
        max_attempts=2,
    )
    assert text is None
    assert not result.ok
