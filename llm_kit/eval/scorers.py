"""Built-in scorers for the eval framework.

Every scorer is a small async (or sync) function returning a float in [0.0, 1.0]
plus an optional reasoning string. New scorers are easy to write — see the
4 built-ins below as templates.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule-based scorers (no LLM)
# ---------------------------------------------------------------------------


def score_length(response_text: str, max_chars: int | None) -> tuple[float, str]:
    """1.0 if response fits within max_chars, else 0.0. None disables the check.

    Returns (score, reason).
    """
    if not max_chars:
        return 1.0, "max_chars not set"
    n = len(response_text or "")
    if n <= int(max_chars):
        return 1.0, f"{n}/{max_chars} chars"
    return 0.0, f"{n}/{max_chars} chars (over by {n - int(max_chars)})"


def score_forbidden_phrases(
    response_text: str, forbidden: list[str]
) -> tuple[float, str]:
    """0.0 if any forbidden substring appears, else 1.0.

    Useful for hard-coded typos, brand misspellings, or specific apology
    patterns you've decided to ban.
    """
    text = response_text or ""
    hits = [p for p in (forbidden or []) if p and p in text]
    if hits:
        return 0.0, f"contains forbidden: {hits}"
    return 1.0, "no forbidden phrases"


def score_tools_used(
    expected: list[str], actual_calls: list[dict]
) -> tuple[float, str]:
    """Fraction of expected tools that were actually called.

    Empty `expected` means "tools should NOT be called" — e.g. for queries
    that the agent should answer directly without dispatching tools. In that
    case: 1.0 if no tool was called, else 0.0.

    Args:
        expected: tool names that the agent SHOULD call.
        actual_calls: list of dicts, each with at least a "tool" key.
    """
    actual_names = {c.get("tool") for c in (actual_calls or []) if c.get("tool")}
    if not expected:
        if not actual_calls:
            return 1.0, "no tools called (as expected)"
        return 0.0, f"unexpected tool calls: {sorted(actual_names)}"
    matched = sum(1 for t in expected if t in actual_names)
    score = matched / len(expected)
    return score, f"{matched}/{len(expected)} expected tools called"


# ---------------------------------------------------------------------------
# LLM-as-judge scorer
# ---------------------------------------------------------------------------


_JUDGE_SYSTEM = (
    "You evaluate the quality of an AI agent's answer against a rubric. "
    "You are also given the actual tool-call results (ground truth) so you "
    "can verify factual fidelity. "
    "Principle: full credit (1.0) if the answer is faithful to the tool "
    "results AND satisfies the rubric. Deduct for content not present in "
    "the tool results, for missing key facts, or for tone violations. "
    "If tool results are empty and the agent honestly says so, that's 1.0. "
    "Return JSON only: {\"score\": <float 0..1>, \"reasoning\": \"<one sentence>\"}"
)


async def score_llm_judge(
    *,
    query: str,
    response_text: str,
    actual_calls: list[dict],
    judge_criteria: str,
    model: str | None = None,
    tool_output_char_cap: int = 5000,
) -> tuple[float, str]:
    """LLM-as-judge scorer using OpenAI's JSON mode.

    The judge receives: the user's query, the agent's answer, the tool calls
    INCLUDING their outputs, and the per-case judge_criteria. It returns a
    score in [0, 1] plus a one-sentence reasoning.

    Tool outputs are truncated to `tool_output_char_cap` chars each — large
    JSON blobs would otherwise blow past the judge's context. Default 5000
    is enough for typical search results.

    Env required: OPENAI_API_KEY.
    Optional: EVAL_JUDGE_MODEL (default "gpt-4o-mini").
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return 0.0, "OPENAI_API_KEY not set"
    model = model or os.getenv("EVAL_JUDGE_MODEL", "gpt-4o-mini")

    tool_summary: list[dict] = []
    for c in actual_calls or []:
        out = c.get("output", "")
        out_str = json.dumps(out, ensure_ascii=False, default=str)
        if len(out_str) > tool_output_char_cap:
            out_str = out_str[:tool_output_char_cap] + "...(truncated)"
        tool_summary.append({
            "tool": c.get("tool"),
            "input": c.get("input", {}),
            "output": out_str,
        })

    user_msg = (
        f"# User query\n{query}\n\n"
        f"# Agent answer\n{response_text}\n\n"
        f"# Tool calls (ground truth)\n"
        f"{json.dumps(tool_summary, ensure_ascii=False, indent=2)}\n\n"
        f"# Rubric\n{judge_criteria}\n"
    )
    body = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
        "max_tokens": 200,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=30, headers=headers) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions", json=body
            )
            r.raise_for_status()
            data = r.json()
        content = (
            data.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        parsed = json.loads(content)
        score = float(parsed.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        return score, str(parsed.get("reasoning", ""))[:250]
    except Exception as e:
        logger.warning(f"[llm_kit.eval] llm_judge call failed: {e}")
        return 0.0, f"judge failed: {type(e).__name__}"
