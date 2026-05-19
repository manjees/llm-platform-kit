"""Unit tests for llm_kit.eval.scorers — rule-based scorers (no LLM)."""

from __future__ import annotations

from llm_kit.eval.scorers import (
    score_forbidden_phrases,
    score_length,
    score_tools_used,
)


def test_score_length_under_cap():
    score, _ = score_length("hi", 10)
    assert score == 1.0


def test_score_length_over_cap():
    score, _ = score_length("hello world", 5)
    assert score == 0.0


def test_score_length_no_cap():
    score, _ = score_length("anything", None)
    assert score == 1.0


def test_score_forbidden_clean():
    score, _ = score_forbidden_phrases("good text", ["bad"])
    assert score == 1.0


def test_score_forbidden_hits():
    score, _ = score_forbidden_phrases("text contains bad", ["bad"])
    assert score == 0.0


def test_score_tools_all_called():
    score, _ = score_tools_used(["a", "b"], [{"tool": "a"}, {"tool": "b"}])
    assert score == 1.0


def test_score_tools_partial():
    score, _ = score_tools_used(["a", "b"], [{"tool": "a"}])
    assert score == 0.5


def test_score_tools_expected_none_actual_none():
    """Empty expected + no actual calls = pass (1.0)."""
    score, _ = score_tools_used([], [])
    assert score == 1.0


def test_score_tools_expected_none_but_called():
    """Empty expected + actual calls = fail (0.0). E.g. agent should have
    answered directly but dispatched tools instead."""
    score, _ = score_tools_used([], [{"tool": "unexpected"}])
    assert score == 0.0
