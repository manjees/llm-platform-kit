"""Pipeline — chain agents where each stage receives the previous stage's text.

Use this for planner → worker → synthesizer flows, where every stage takes
a single string in and produces a single string out.

For branching / parallel work or stages that need structured outputs, use
`asyncio.gather` or write your own orchestrator — Pipeline is intentionally
the simplest case.
"""

from __future__ import annotations

import logging
from typing import Any

from llm_kit.agents.agent import Agent

logger = logging.getLogger(__name__)


class Pipeline:
    """Sequential agent pipeline.

    Each stage is an Agent. The pipeline calls them in order. The output of
    stage N is passed as the `user_message` of stage N+1.

    Example:
        planner = Agent(name="plan", ...)        # breaks task into steps
        worker  = Agent(name="execute", ...)     # carries out the plan
        writer  = Agent(name="format", ...)      # final user-facing message

        pipe = Pipeline(stages=[planner, worker, writer])
        result = await pipe.run("Summarize last week's metrics for the team")
        print(result.text)
    """

    def __init__(self, *, stages: list[Agent]):
        if not stages:
            raise ValueError("Pipeline needs at least one stage")
        self.stages = list(stages)

    async def run(
        self,
        initial_message: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Run all stages sequentially. Returns the last stage's AgentResult."""
        current = initial_message
        last_result = None
        for i, agent in enumerate(self.stages):
            stage_metadata = {
                **(metadata or {}),
                "pipeline_stage": i,
                "pipeline_total": len(self.stages),
            }
            last_result = await agent.run(current, metadata=stage_metadata)
            current = last_result.text
        return last_result
