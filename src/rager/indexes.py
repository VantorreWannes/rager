"""Embedding Index protocol definitions."""

from typing import TYPE_CHECKING, Protocol

import blake3
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
    def _to_row(embedding: DenseEmbedding) -> np.ndarray:
        """Convert an embedding into a single-row float32 matrix for FAISS."""
        return np.asarray([embedding], dtype=np.float32)

    @staticmethod
    def _key(row: np.ndarray) -> int:
        """Derive a deterministic int64 key from the embedding's content."""
        digest = blake3.blake3(row.tobytes()).digest()
        return int.from_bytes(digest[:8], "little", signed=True)

    async def add(self, embedding: DenseEmbedding) -> int:
        """Add an embedding to the index and return its key."""
        row = self._to_row(embedding)
        key = self._key(row)
        ids = np.asarray([key], dtype=np.int64)
        self._index.remove_ids(faiss.IDSelectorBatch(ids))
        self._index.add_with_ids(row, ids)
        return key

    async def remove(self, key: int) -> None:
        """Remove an embedding from the index by its key."""
        ids = np.asarray([key], dtype=np.int64)
        self._index.remove_ids(faiss.IDSelectorBatch(ids))

    async def similar(self, embedding: DenseEmbedding) -> list[int]:
        """Retrieve the most similar embedding keys to the given embedding."""
        _, ids = self._index.search(self._to_row(embedding), self.results)
        return [int(i) for i in ids[0] if i != -1]
