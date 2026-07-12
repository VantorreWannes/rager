"""Embedding Index protocol definitions."""

from datetime import timedelta
from typing import TYPE_CHECKING, Protocol

import blake3
import concresce
import faiss
import numpy as np

if TYPE_CHECKING:
    from rager import DenseEmbedding


class Index[E, K](Protocol):
    """Protocol for an embedding index."""

    async def add(self, embedding: E) -> K:
        """Add an embedding to the index and return its key."""
        ...

    async def remove(self, key: K) -> None:
        """Remove an embedding from the index by its key."""
        ...

    async def similar(self, embedding: E) -> list[K]:
        """Retrieve the most similar embedding keys to the given embedding."""
        ...


class DenseIndex:
    """Dense embedding index backed by a flat FAISS inner-product index."""

    def __init__(self, dimensions: int, results: int = 10) -> None:
        """Initialize the FAISS index with the specified embedding dimension."""
        self.dimensions = dimensions
        self.results = results
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimensions))

    @staticmethod
    def _to_rows(embeddings: list[DenseEmbedding]) -> np.ndarray:
        """Convert embeddings into a float32 matrix for FAISS."""
        return np.asarray(embeddings, dtype=np.float32)

    @staticmethod
    def _key(row: np.ndarray) -> int:
        """Derive a deterministic int64 key from the embedding's content."""
        digest = blake3.blake3(row.tobytes()).digest()
        return int.from_bytes(digest[:8], "little", signed=True)

    @concresce.batch(window=timedelta(milliseconds=100))
    async def add(self, embedding: DenseEmbedding) -> int:
        """Add an embedding to the index and return its key."""
        embeddings = await concresce.collect(embedding)
        rows = self._to_rows(embeddings)
        keys = [self._key(row) for row in rows]
        unique = dict(zip(keys, rows, strict=True))
        ids = np.asarray(list(unique), dtype=np.int64)
        self._index.remove_ids(faiss.IDSelectorBatch(ids))
        self._index.add_with_ids(self._to_rows(list(unique.values())), ids)
        return concresce.scatter(keys)

    @concresce.batch(window=timedelta(milliseconds=100))
    async def remove(self, key: int) -> None:
        """Remove an embedding from the index by its key."""
        keys = await concresce.collect(key)
        ids = np.asarray(keys, dtype=np.int64)
        self._index.remove_ids(faiss.IDSelectorBatch(ids))
        return concresce.scatter([None] * len(keys))

    @concresce.batch(window=timedelta(milliseconds=100))
    async def similar(self, embedding: DenseEmbedding) -> list[int]:
        """Retrieve the most similar embedding keys to the given embedding."""
        embeddings = await concresce.collect(embedding)
        _, ids = self._index.search(self._to_rows(embeddings), self.results)
        results = [[int(i) for i in row if i != -1] for row in ids]
        return concresce.scatter(results)
