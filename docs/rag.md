# RAG — Chroma + OpenAI embedding

Semantic search for the LLM context window. Beats keyword search when users
phrase questions in their own words.

## Why

| Approach | Strength | Weakness |
|---|---|---|
| `WHERE name = '...'` (SQL) | Exact match, fast, indexed | Synonyms / paraphrase miss |
| FTS / trigram | Substring + fuzzy match | Misses semantic relationships |
| Vector / RAG | Captures meaning, not just words | Embedding cost, index size |

In practice: SQL/FTS for "find by ID/keyword", RAG for "what's similar to
this idea".

## Setup

```bash
pip install llm-platform-kit[rag]
export OPENAI_API_KEY=sk-...
# optional
export CHROMA_PATH=./my_chroma_data    # default: ./chroma_data
export EMBED_MODEL=text-embedding-3-small   # default
```

## Basic usage

```python
from llm_kit.rag import RAGCollection

rag = RAGCollection("knowledge_base")

await rag.upsert(
    ids=["doc1", "doc2", "doc3"],
    texts=["Refund policy...", "Shipping rates...", "Payment methods..."],
    metadatas=[
        {"category": "policy"},
        {"category": "logistics"},
        {"category": "billing"},
    ],
)

hits = await rag.search("how do I get my money back?", top_k=3)
# [
#   {"id": "doc1", "text": "Refund policy...", "similarity": 0.82, "metadata": {...}},
#   ...
# ]
```

`similarity` is in `[0, 1]` (higher = closer). It's `1 - cosine_distance`.

## Filtering

Chroma supports metadata filtering via the `where` arg:

```python
hits = await rag.search(
    "shipping question",
    top_k=3,
    where={"category": "logistics"},
)
```

## Delete (e.g. after source doc removed)

```python
rag.delete(ids=["doc3"])
```

## Indexing pipeline

For continuously growing data, run a cron that:

1. Selects new rows since `last_embedded_at`
2. Batches them (100 per call is safe)
3. Calls `rag.upsert(...)`
4. Updates `last_embedded_at`

```python
from llm_kit.rag import embed_texts, RAGCollection

async def index_new_docs(rows: list[dict]) -> int:
    rag = RAGCollection("news")
    return await rag.upsert(
        ids=[r["id"] for r in rows],
        texts=[r["title"] + "\n" + r["snippet"] for r in rows],
        metadatas=[{"source": r["source"], "published_at": r["published_at"]} for r in rows],
    )
```

## Cost

`text-embedding-3-small` (1536-dim) at $0.02 per 1M tokens. Typical doc =
~200 tokens.

| Volume | Cost / month |
|---|---|
| 100 docs/day | $0.012 |
| 10K docs/day | $1.20 |
| 100K docs/day | $12 |

Storage for Chroma local file is ~ 6 KB per doc (vector + metadata).

## When to graduate from Chroma local

| Scale | Recommended |
|---|---|
| < 100K docs | Chroma local file (this library) |
| 100K–10M | Qdrant (Rust, fast) or pgvector |
| > 10M, multi-tenant | Pinecone / Vertex AI / Vespa |

The `RAGCollection` API is small enough that swapping the backend later
means rewriting one file. Embedding contracts (1536-dim OpenAI) stay
identical, so existing vectors transfer.

## Chunking (when docs are huge)

This library doesn't chunk for you — you pass the text as-is. For typical
KB articles (under 1000 tokens) just pass the full text. For longer docs,
chunk before upsert:

```python
def chunk(text: str, max_chars: int = 800, overlap: int = 100) -> list[str]:
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + max_chars])
        i += max_chars - overlap
    return chunks
```

Each chunk gets its own `id` (e.g. `"doc1#0"`, `"doc1#1"`).

## Hybrid search (FTS + vector)

For best recall combine both — exact keyword match for named entities,
vector for paraphrase. Pattern:

```python
async def hybrid_search(query: str, k: int = 5):
    fts_hits = await my_fts_search(query, top_k=k)        # keyword
    rag_hits = await rag.search(query, top_k=k)           # semantic
    seen = set()
    out = []
    for h in fts_hits + rag_hits:
        if h["id"] in seen:
            continue
        seen.add(h["id"])
        out.append(h)
        if len(out) >= k:
            break
    return out
```

## Cleanup parity

If your source data deletes a row (TTL, GDPR, etc.) delete from the vector
store too — otherwise stale hits keep appearing:

```python
async def delete_doc(doc_id: str):
    await my_sql.delete(doc_id)
    rag.delete(ids=[doc_id])
```

## Common pitfalls

- ❌ Switching embedding models mid-stream → all stored vectors invalid;
  re-embed everything
- ❌ Chunks too large (> 1500 chars) → meaning dilutes, similarity drops
- ❌ Storing the embedding vector in metadata too — wastes 6 KB per doc
- ❌ Ignoring `metadata.published_at` for time-sensitive results
