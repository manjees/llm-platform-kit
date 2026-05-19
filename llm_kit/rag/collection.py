"""Chroma collection wrapper — upsert, search, delete with sane defaults.

Local file mode (no separate service). For 100K+ docs, migrate to pgvector
or Qdrant — the embedding pipeline stays the same.

Env:
    CHROMA_PATH   default "./chroma_data"

Distance: cosine (1 - similarity). Search results expose similarity in
[0, 1] — closer to 1 means more similar.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

from llm_kit.rag.embeddings import embed_one, embed_texts

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_PATH = "./chroma_data"


@lru_cache(maxsize=1)
def _client() -> Any | None:
    """Chroma PersistentClient singleton. None if SDK or path fails."""
    try:
        import chromadb
    except ImportError as e:
        logger.warning(
            f"[llm_kit.rag] chromadb not installed ({e}). "
            "Install: pip install llm-platform-kit[rag]"
        )
        return None
    path = os.getenv("CHROMA_PATH", DEFAULT_CHROMA_PATH).strip() or DEFAULT_CHROMA_PATH
    try:
        os.makedirs(path, exist_ok=True)
        return chromadb.PersistentClient(path=path)
    except Exception as e:
        logger.warning(f"[llm_kit.rag] Chroma client init failed: {e}")
        return None


class RAGCollection:
    """A named vector collection. Thin wrapper around a Chroma collection.

    Cosine distance, 1536-dim vectors (assumes default OpenAI embedder).
    """

    def __init__(self, name: str):
        self.name = name
        self._coll = self._get_or_create()

    def _get_or_create(self) -> Any | None:
        client = _client()
        if client is None:
            return None
        try:
            return client.get_or_create_collection(
                name=self.name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            logger.warning(
                f"[llm_kit.rag] get_or_create_collection failed ({self.name}): {e}"
            )
            return None

    @property
    def is_ready(self) -> bool:
        """True if Chroma client + collection are both up."""
        return self._coll is not None

    # ---- Write ----------------------------------------------------------

    async def upsert(
        self,
        *,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        """Embed `texts` and upsert into the collection. Returns count.

        ids, texts, metadatas must all be the same length (or metadatas None).
        Upsert = insert-or-replace, so calling twice with the same id is safe.

        Returns 0 if embedding or upsert failed.
        """
        if not ids or not texts:
            return 0
        if len(ids) != len(texts):
            raise ValueError("ids and texts must have the same length")
        if metadatas is None:
            metadatas = [{} for _ in ids]
        if len(metadatas) != len(ids):
            raise ValueError("metadatas length must match ids")
        if not self.is_ready:
            return 0
        vectors = await embed_texts(texts)
        if not vectors:
            return 0
        try:
            self._coll.upsert(
                ids=ids,
                embeddings=vectors,
                documents=texts,
                metadatas=metadatas,
            )
            return len(ids)
        except Exception as e:
            logger.warning(f"[llm_kit.rag] upsert failed: {e}")
            return 0

    # ---- Read -----------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search — embed query, return top_k by cosine similarity.

        Each hit:
            {"id", "text", "similarity" (0..1), "metadata"}

        Returns [] if RAG infra unavailable or query embedding fails.
        """
        if not self.is_ready:
            return []
        vector = await embed_one(query)
        if vector is None:
            return []
        try:
            raw = self._coll.query(
                query_embeddings=[vector],
                n_results=max(1, min(int(top_k), 100)),
                where=where,
            )
        except Exception as e:
            logger.warning(f"[llm_kit.rag] query failed: {e}")
            return []

        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]

        out: list[dict[str, Any]] = []
        for _id, doc, meta, dist in zip(ids, docs, metas, dists):
            out.append({
                "id": _id,
                "text": doc,
                "similarity": round(1.0 - float(dist), 4),
                "metadata": meta or {},
            })
        return out

    # ---- Delete ---------------------------------------------------------

    def delete(self, ids: list[str]) -> int:
        """Delete by id. Returns count attempted (Chroma doesn't report rowcount)."""
        if not ids or not self.is_ready:
            return 0
        try:
            self._coll.delete(ids=ids)
            return len(ids)
        except Exception as e:
            logger.warning(f"[llm_kit.rag] delete failed: {e}")
            return 0

    def count(self) -> int:
        """Number of documents currently in the collection."""
        if not self.is_ready:
            return 0
        try:
            return int(self._coll.count())
        except Exception:
            return 0
