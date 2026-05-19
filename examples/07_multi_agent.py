"""Example — Writer+Critic loop and a 3-stage Pipeline using mock LLMs.

No OpenAI key needed for this demo — `call_fn` is a stand-in. In real
code your call_fn calls OpenAI / Anthropic / your sandbox model.

Run:
    python examples/07_multi_agent.py
"""

from __future__ import annotations

import asyncio

from llm_kit.agents import (
    Agent,
    CriticVerdict,
    Pipeline,
    WriterCriticPair,
)


# --- Mock LLM call_fn -----------------------------------------------------


WRITER_DRAFTS = iter([
    "this is a way too long answer that goes on and on and on " * 10,
    "We've shipped real-time alerts — check Slack for details!",
])


async def fake_writer_call(messages):
    # Returns next canned draft each time it's called.
    return next(WRITER_DRAFTS, "fallback"), {"input": 100, "output": 30, "total": 130}


async def fake_planner_call(messages):
    return "Step 1: gather metrics. Step 2: spot anomalies. Step 3: write summary.", \
           {"input": 50, "output": 30, "total": 80}


async def fake_worker_call(messages):
    return (
        "Anomalies found: latency p95 +18%, error rate stable, no cost spike.",
        {"input": 80, "output": 30, "total": 110},
    )


async def fake_writer_stage_call(messages):
    return (
        "Weekly status: p95 latency rose 18% — investigate hot endpoints. "
        "Errors / cost unchanged.",
        {"input": 120, "output": 30, "total": 150},
    )


# --- Demo 1: WriterCriticPair ---------------------------------------------


def length_critic(text: str) -> CriticVerdict:
    if len(text) > 120:
        return CriticVerdict(
            ok=False,
            feedback=f"Draft is {len(text)} chars; keep under 120.",
        )
    return CriticVerdict(ok=True, feedback="ok")


async def demo_writer_critic():
    writer = Agent(
        name="demo.writer",
        model="mock-gpt",
        system_prompt="You are a concise product copywriter.",
        call_fn=fake_writer_call,
        tags=["demo", "writer_critic"],
    )
    pair = WriterCriticPair(writer=writer, critic=length_critic, max_attempts=2)

    text, verdict = await pair.run("Announce our real-time alerts feature.")
    print("=== WriterCriticPair ===")
    print(f"accepted: {verdict.ok}, feedback: {verdict.feedback}")
    print(f"final:    {text}")


# --- Demo 2: Pipeline -----------------------------------------------------


async def demo_pipeline():
    planner = Agent(
        name="demo.planner", model="mock-gpt",
        system_prompt="Plan the work in short numbered steps.",
        call_fn=fake_planner_call,
    )
    worker = Agent(
        name="demo.worker", model="mock-gpt",
        system_prompt="Execute the plan; return findings in one paragraph.",
        call_fn=fake_worker_call,
    )
    writer = Agent(
        name="demo.summarizer", model="mock-gpt",
        system_prompt="Turn findings into a Slack-ready status update.",
        call_fn=fake_writer_stage_call,
    )

    pipe = Pipeline(stages=[planner, worker, writer])
    result = await pipe.run("Summarize last week's reliability metrics.")
    print("\n=== Pipeline ===")
    print(f"final:    {result.text}")


async def main():
    await demo_writer_critic()
    await demo_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
