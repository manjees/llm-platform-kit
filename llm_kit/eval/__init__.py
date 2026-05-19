"""YAML-driven eval framework — give your LLM features a regression safety net.

Public API:
    EvalItem, EvalSet, EvalResult     — data classes
    load_eval_set(path)               — read YAML
    score_length / score_forbidden /
    score_tools_used / score_llm_judge — 4 built-in scorers
    run_eval(path, agent_fn, ...)     — run all items × all scorers
"""

from llm_kit.eval.runner import (
    EvalItem,
    EvalResult,
    EvalSet,
    load_eval_set,
    run_eval,
)
from llm_kit.eval.scorers import (
    score_forbidden_phrases,
    score_length,
    score_llm_judge,
    score_tools_used,
)

__all__ = [
    "EvalItem",
    "EvalResult",
    "EvalSet",
    "load_eval_set",
    "run_eval",
    "score_forbidden_phrases",
    "score_length",
    "score_llm_judge",
    "score_tools_used",
]
