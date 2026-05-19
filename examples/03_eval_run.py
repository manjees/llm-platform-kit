"""Example — run an eval set against a toy agent.

The "agent" here is a mock — replace with your real agent that returns
{"text": ..., "tools_called": [...]}.

Run:
    export OPENAI_API_KEY=sk-...   # for the LLM judge
    python examples/03_eval_run.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from llm_kit.eval import run_eval


async def toy_agent(query: str) -> dict:
    """Stand-in for your real agent. Replace this body."""
    if "2 + 2" in query:
        return {"text": "The answer is 4.", "tools_called": []}
    if "bank account balance" in query:
        return {
            "text": (
                "I don't have access to your banking data. "
                "Please check your bank's app or website."
            ),
            "tools_called": [],
        }
    return {"text": f"I don't know how to answer: {query}", "tools_called": []}


async def main() -> None:
    eval_yaml = Path(__file__).parent / "eval_set_example.yaml"
    summary = await run_eval(eval_yaml, agent_fn=toy_agent)
    print(json.dumps(summary["scores"], indent=2))
    print()
    for item in summary["items"]:
        print(
            f"[{item['id']:<30}] judge={item['scores']['llm_judge']:.2f} "
            f"| {item['response_text'][:60]}"
        )


if __name__ == "__main__":
    asyncio.run(main())
