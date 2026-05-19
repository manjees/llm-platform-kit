"""Eval runner — read YAML test set, run each item through your agent, score.

Your agent is a black box from the runner's perspective. Pass an
`agent_fn(query: str) -> dict` callable; the dict must contain:

    {
        "text":          str,         # the agent's textual answer
        "tools_called":  list[dict],  # optional, each {"tool", "input", "output"}
    }

`tools_called` may be omitted if your agent doesn't use tools — scorers that
need it will simply receive an empty list.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # noqa: N816

from llm_kit.eval.scorers import (
    score_forbidden_phrases,
    score_length,
    score_llm_judge,
    score_tools_used,
)
from llm_kit.observability import flush_langfuse, trace_span

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EvalItem:
    id: str
    query: str
    expected_tools: list[str] = field(default_factory=list)
    forbidden_phrases: list[str] = field(default_factory=list)
    judge_criteria: str = ""
    max_chars: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSet:
    dataset_name: str
    items: list[EvalItem]


@dataclass
class EvalResult:
    item_id: str
    duration_ms: int
    response_text: str
    tools_called: list[dict]
    scores: dict[str, float]
    judge_reason: str
    error: str | None = None


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_eval_set(path: str | Path) -> EvalSet:
    """Read a YAML file into an EvalSet. See docs/eval.md for the schema."""
    if yaml is None:
        raise RuntimeError("PyYAML not installed. pip install pyyaml")
    raw = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(data).__name__}")
    items_raw = data.get("items") or []
    items = [
        EvalItem(
            id=str(item["id"]),
            query=str(item["query"]),
            expected_tools=list(item.get("expected_tools") or []),
            forbidden_phrases=list(item.get("forbidden_phrases") or []),
            judge_criteria=str(item.get("judge_criteria") or ""),
            max_chars=item.get("max_chars"),
            metadata=dict(item.get("metadata") or {}),
        )
        for item in items_raw
    ]
    return EvalSet(
        dataset_name=str(data.get("dataset_name", "eval_set")),
        items=items,
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


AgentFn = Callable[[str], Awaitable[dict]]


async def run_eval(
    eval_set: str | Path | EvalSet,
    *,
    agent_fn: AgentFn,
    skip_llm_judge: bool = False,
    judge_model: str | None = None,
) -> dict[str, Any]:
    """Run every item × every scorer, then return a summary.

    Args:
        eval_set: path to YAML or an already-loaded EvalSet.
        agent_fn: async (query) -> {"text", "tools_called"} for your agent.
        skip_llm_judge: set True for a fast local run that avoids LLM judge
                        API cost (useful for quick CI smoke tests).
        judge_model: override OpenAI model for the judge.

    Returns:
        {
            "dataset_name": str,
            "total": int,
            "scores": {scorer: avg_float, ...},
            "items": [EvalResult-as-dict, ...],
        }

    Side effects:
        Each item is wrapped in a Langfuse trace (no-op if observability
        is disabled). flush_langfuse() is called at the end.
    """
    if not isinstance(eval_set, EvalSet):
        eval_set = load_eval_set(eval_set)

    summary_scores: dict[str, float] = {
        "length": 0.0,
        "hallucination": 0.0,
        "tools_used": 0.0,
        "llm_judge": 0.0,
    }
    items_out: list[dict] = []

    for item in eval_set.items:
        logger.info(f"[llm_kit.eval] running {item.id}: {item.query!r}")

        span = trace_span(
            name=f"eval.{item.id}",
            input={"query": item.query},
            tags=["eval", eval_set.dataset_name, item.id],
        )

        t0 = time.monotonic()
        try:
            agent_out = await agent_fn(item.query)
            response_text = str(agent_out.get("text") or "")
            tools_called = list(agent_out.get("tools_called") or [])
            err: str | None = None
        except Exception as e:
            logger.warning(f"[llm_kit.eval] agent_fn failed for {item.id}: {e}")
            response_text = ""
            tools_called = []
            err = f"{type(e).__name__}: {e}"
        duration_ms = int((time.monotonic() - t0) * 1000)

        s_len, _ = score_length(response_text, item.max_chars)
        s_halu, _ = score_forbidden_phrases(response_text, item.forbidden_phrases)
        s_tools, _ = score_tools_used(item.expected_tools, tools_called)
        if skip_llm_judge:
            s_judge, judge_reason = 0.0, "skipped"
        else:
            s_judge, judge_reason = await score_llm_judge(
                query=item.query,
                response_text=response_text,
                actual_calls=tools_called,
                judge_criteria=item.judge_criteria,
                model=judge_model,
            )

        scores = {
            "length": s_len,
            "hallucination": s_halu,
            "tools_used": s_tools,
            "llm_judge": s_judge,
        }
        for k, v in scores.items():
            summary_scores[k] += v

        if span is not None:
            try:
                span.update(
                    output={"text": response_text,
                            "tools": [c.get("tool") for c in tools_called]},
                    metadata={
                        "duration_ms": duration_ms,
                        "error": err,
                    },
                )
                for name, value in scores.items():
                    span.score(name=name, value=value)
                if judge_reason:
                    span.score(
                        name="llm_judge_reason",
                        value=s_judge,
                        comment=judge_reason,
                    )
            except Exception as e:
                logger.debug(f"[llm_kit.eval] langfuse span update failed: {e}")

        items_out.append({
            "id": item.id,
            "duration_ms": duration_ms,
            "response_text": response_text,
            "tools_called": [c.get("tool") for c in tools_called],
            "scores": scores,
            "judge_reason": judge_reason,
            "error": err,
        })

    n = len(eval_set.items) or 1
    avg_scores = {k: round(v / n, 3) for k, v in summary_scores.items()}

    flush_langfuse()

    logger.info(
        f"[llm_kit.eval] done — dataset={eval_set.dataset_name} "
        f"n={len(eval_set.items)} avg={avg_scores}"
    )
    return {
        "dataset_name": eval_set.dataset_name,
        "total": len(eval_set.items),
        "scores": avg_scores,
        "items": items_out,
    }
