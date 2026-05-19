"""Hallucination & quality guards — composable layers, used in production.

The 6 layers (in the order they typically run):

  1. System-prompt rules        — operator-owned (you write the prompt)
  2. Tool-schema directives     — operator-owned (you write tool descriptions)
  3. Forbidden-phrase regex     — `check_forbidden_phrases(...)`
  4. Length cap                 — `check_length(...)`
  5. Slop / boilerplate regex   — `check_slop(...)`
  6. Orchestrated retry         — `with_retry(generate_fn, validator)`

Layers 1–2 live in your prompts (this library doesn't touch them — see the
`prompts` module for hot-swap). Layers 3–5 are pure functions you compose.
Layer 6 is a tiny retry harness for "validate; if bad, regenerate once".

For agent-style flows where you need an LLM-as-judge sanity check, plug
`llm_kit.eval.score_llm_judge` into the validator.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


@dataclass
class GuardResult:
    """Outcome of a single guard check.

    `ok=True` → response passes this layer. `ok=False` → reason explains why.
    """
    ok: bool
    reason: str


# ---------------------------------------------------------------------------
# Layer 3: forbidden phrases
# ---------------------------------------------------------------------------


def check_forbidden_phrases(text: str, forbidden: list[str]) -> GuardResult:
    """Fail if any forbidden substring appears in `text`.

    Use for brand misspellings, banned apologies, leaked internal terms, etc.
    Compare substring (not regex) — use check_pattern() for regex.
    """
    hits = [p for p in (forbidden or []) if p and p in (text or "")]
    if hits:
        return GuardResult(False, f"forbidden phrases present: {hits}")
    return GuardResult(True, "no forbidden phrases")


# ---------------------------------------------------------------------------
# Layer 4: length cap
# ---------------------------------------------------------------------------


def check_length(text: str, max_chars: int | None) -> GuardResult:
    """Fail if `text` exceeds `max_chars`. None disables the check."""
    if not max_chars:
        return GuardResult(True, "no length cap")
    n = len(text or "")
    if n <= int(max_chars):
        return GuardResult(True, f"{n}/{max_chars} chars")
    return GuardResult(False, f"too long: {n}/{max_chars} chars")


# ---------------------------------------------------------------------------
# Layer 5: slop / boilerplate patterns
# ---------------------------------------------------------------------------


# Common "AI-flavored boilerplate" patterns that signal the model is filling
# space rather than answering. Extend per project.
DEFAULT_SLOP_PATTERNS: tuple[str, ...] = (
    r"however,?\s+it'?s? (important|worth) to note",
    r"in conclusion,?",
    r"i hope (this helps|that helps)",
    r"feel free to ask",
    r"\b(definitely|absolutely)\b",
)


def check_slop(
    text: str,
    *,
    patterns: tuple[str, ...] = DEFAULT_SLOP_PATTERNS,
    case_insensitive: bool = True,
) -> GuardResult:
    """Fail if any slop regex matches. Use for tone / quality control.

    `patterns` is a tuple of regex strings — defaults cover common LLM
    boilerplate in English. Replace with your project-specific patterns
    (e.g. Korean slop, marketing fluff, etc.).
    """
    flags = re.IGNORECASE if case_insensitive else 0
    for pat in patterns:
        if re.search(pat, text or "", flags):
            return GuardResult(False, f"slop pattern matched: {pat!r}")
    return GuardResult(True, "no slop patterns")


def check_pattern(
    text: str, pattern: str, *, must_match: bool = True
) -> GuardResult:
    """Generic regex helper. must_match=False means "must NOT match".

    Useful when forbidden_phrases (substring) is too rigid — e.g. you want
    to forbid URL placeholders like `[link]` or `[INSERT URL]`.
    """
    m = re.search(pattern, text or "")
    if must_match and not m:
        return GuardResult(False, f"required pattern missing: {pattern!r}")
    if not must_match and m:
        return GuardResult(False, f"forbidden pattern matched: {pattern!r}")
    return GuardResult(True, "pattern check passed")


# ---------------------------------------------------------------------------
# Layer 6: retry orchestration
# ---------------------------------------------------------------------------


Validator = Callable[[str], GuardResult]
Generator = Callable[[int], Awaitable[str]]


async def with_retry(
    generate: Generator,
    validator: Validator,
    *,
    max_attempts: int = 2,
) -> tuple[str | None, GuardResult]:
    """Run `generate` up to `max_attempts` times, accepting the first that
    passes `validator`. Returns (final_text or None, last_GuardResult).

    `generate(attempt)` receives the 0-indexed attempt number — use it to
    tweak the prompt on retry (e.g. "your previous answer was too long;
    keep it under 280 characters").

    `validator(text)` is sync (chain multiple guard checks inside it).

    Returns (None, GuardResult(ok=False, ...)) if all attempts fail.
    """
    last: GuardResult = GuardResult(False, "no attempts")
    for attempt in range(max(1, int(max_attempts))):
        try:
            text = await generate(attempt)
        except Exception as e:
            logger.warning(f"[llm_kit.guards] generate failed (attempt={attempt}): {e}")
            last = GuardResult(False, f"generate failed: {type(e).__name__}")
            continue
        last = validator(text)
        if last.ok:
            return text, last
        logger.info(
            f"[llm_kit.guards] attempt {attempt} rejected: {last.reason}"
        )
    return None, last


def combine_validators(*validators: Validator) -> Validator:
    """AND-combine multiple validators. First failure stops the chain.

    Example:
        v = combine_validators(
            lambda t: check_length(t, 280),
            lambda t: check_forbidden_phrases(t, ["typo_brand"]),
            lambda t: check_slop(t),
        )
        text, result = await with_retry(generate_fn, v)
    """
    def _combined(text: str) -> GuardResult:
        for v in validators:
            r = v(text)
            if not r.ok:
                return r
        return GuardResult(True, "all checks passed")
    return _combined
