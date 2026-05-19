"""RAG (Retrieval-Augmented Generation) — Chroma + OpenAI embedding.

Public API:
    embed_texts / embed_one   — batch / single OpenAI embedding calls
    RAGCollection             — Chroma collection wrapper (upsert/search/delete)
"""

from llm_kit.rag.collection import RAGCollection
from llm_kit.rag.embeddings import embed_one, embed_texts

__all__ = ["RAGCollection", "embed_one", "embed_texts"]
