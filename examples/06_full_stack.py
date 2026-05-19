"""Example — wire all four modules together in one mini agent.

Flow per query:
    1. RAG semantic search  →  get the most relevant FAQ doc
    2. Build an LLM prompt with that doc as context
    3. Call OpenAI through Langfuse trace
    4. Guard the response (length cap + forbidden phrases + retry)

Run:
    export OPENAI_API_KEY=sk-...
    export LANGFUSE_PUBLIC_KEY=pk-lf-...        # optional
    export LANGFUSE_SECRET_KEY=sk-lf-...        # optional
    python examples/06_full_stack.py

The Langfuse pieces are no-ops if env is unset — see observability.py.
"""

from __future__ import annotations

import asyncio
import os

import httpx

from llm_kit.guards import (
    check_forbidden_phrases,
    check_length,
    combine_validators,
    with_retry,
)
from llm_kit.observability import flush_langfuse, trace_generation
from llm_kit.rag import RAGCollection

DOCS = [
    ("doc1", "Refunds: returns accepted within 30 days, original packaging."),
    ("doc2", "Shipping: standard 3-5 business days, expedited 1-2 days extra fee."),
    ("doc3", "Payment: credit card, PayPal, Apple Pay, Google Pay. No cryptocurrency."),
    ("doc4", "Support hours: Mon-Fri 9am-6pm KST. Email support@example.com."),
]


async def index_docs() -> RAGCollection:
    rag = RAGCollection("full_stack_demo")
    if rag.is_ready and rag.count() == 0:
        await rag.upsert(
            ids=[d[0] for d in DOCS],
            texts=[d[1] for d in DOCS],
            metadatas=[{"category": "faq"} for _ in DOCS],
        )
    return rag


async def call_openai(messages: list[dict], model: str = "gpt-4o-mini") -> tuple[str, dict]:
    api_key = os.environ["OPENAI_API_KEY"]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 200}
    async with httpx.AsyncClient(timeout=20, headers=headers) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions", json=body)
        r.raise_for_status()
        data = r.json()
    text = data["choices"][0]["message"]["content"].strip()
    usage = data.get("usage", {})
    return text, {
        "input": usage.get("prompt_tokens", 0),
        "output": usage.get("completion_tokens", 0),
        "total": usage.get("total_tokens", 0),
    }


async def answer_question(rag: RAGCollection, question: str) -> str | None:
    # 1. Retrieve
    hits = await rag.search(question, top_k=2)
    context = "\n".join(f"- {h['text']}" for h in hits) or "(no matching docs)"

    # 2. Build prompt
    system = (
        "You are a customer support assistant. Answer using ONLY the provided "
        "context. If the context doesn't cover the question, say so honestly. "
        "Keep the answer under 200 characters."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]

    # 3. Generate with retry + 4. guards
    async def generate(attempt: int) -> str:
        if attempt > 0:
            messages.append({
                "role": "user",
                "content": "Your previous answer was too long. Keep it under 200 chars.",
            })
        text, usage = await call_openai(messages)
        trace_generation(
            name="full_stack.answer",
            model="gpt-4o-mini",
            input=messages,
            output=text,
            usage=usage,
            metadata={"attempt": attempt, "n_hits": len(hits)},
            tags=["full_stack_demo"],
        )
        return text

    validator = combine_validators(
        lambda t: check_length(t, 200),
        lambda t: check_forbidden_phrases(t, ["I don't know"]),
    )
    text, result = await with_retry(generate, validator, max_attempts=2)
    print(f"  guard result: ok={result.ok}, reason={result.reason}")
    return text


async def main() -> None:
    rag = await index_docs()
    if not rag.is_ready:
        print("⚠️  RAG infra unavailable. pip install llm-platform-kit[rag]")
        return

    for q in [
        "How do I return a product?",
        "Can I pay with bitcoin?",
        "When do you ship internationally?",
    ]:
        print(f"\nQ: {q}")
        answer = await answer_question(rag, q)
        print(f"A: {answer}")

    flush_langfuse()


if __name__ == "__main__":
    asyncio.run(main())
