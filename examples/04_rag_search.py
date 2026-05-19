"""Example — index a few documents and run semantic search.

Run:
    export OPENAI_API_KEY=sk-...
    python examples/04_rag_search.py
"""

from __future__ import annotations

import asyncio

from llm_kit.rag import RAGCollection


DOCS = [
    ("doc1", "Refund policy: returns accepted within 30 days of purchase, original packaging required."),
    ("doc2", "Shipping: standard delivery 3-5 business days, expedited 1-2 days for an extra fee."),
    ("doc3", "Payment methods: credit card, PayPal, Apple Pay, Google Pay. No cryptocurrency."),
    ("doc4", "Contact: support@example.com, available Monday–Friday 9am–6pm KST."),
    ("doc5", "Loyalty program: 1 point per dollar spent, 100 points = $5 discount."),
]


async def main() -> None:
    rag = RAGCollection("examples_kb")
    if not rag.is_ready:
        print("⚠️  Chroma not available. Install: pip install llm-platform-kit[rag]")
        return

    print(f"Indexing {len(DOCS)} docs...")
    n = await rag.upsert(
        ids=[d[0] for d in DOCS],
        texts=[d[1] for d in DOCS],
        metadatas=[{"category": "faq"} for _ in DOCS],
    )
    print(f"Upserted: {n}. Total in collection: {rag.count()}")

    for query in [
        "how do I get my money back?",
        "when does my package arrive?",
        "can I pay with bitcoin?",
    ]:
        print(f"\nQ: {query}")
        hits = await rag.search(query, top_k=2)
        for h in hits:
            print(f"  sim={h['similarity']:.3f}  {h['text'][:70]}")


if __name__ == "__main__":
    asyncio.run(main())
