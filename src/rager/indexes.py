"""Embedding Index protocol definitions."""

from datetime import timedelta
from functools import cached_property
from typing import TYPE_CHECKING, Protocol

import blake3
import concresce
import faiss
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from rager import DenseEmbedding, SparseEmbedding


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


_BATCH_WINDOW = timedelta(milliseconds=100)


class DenseIndex:
    """Dense embedding index backed by a flat FAISS inner-product index."""

    def __init__(self, dimensions: int, results: int = 10) -> None:
        """Initialize the FAISS index with the specified embedding dimension."""
        self.dimensions = dimensions
        self.results = results
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimensions))

    @cached_property
    def add(self) -> Callable[[DenseEmbedding], Awaitable[int]]:
        """Add an embedding to the index and return its key."""
        return concresce.batch(window=_BATCH_WINDOW)(self._add)

    @cached_property
    def remove(self) -> Callable[[int], Awaitable[None]]:
        """Remove an embedding from the index by its key."""
        return concresce.batch(window=_BATCH_WINDOW)(self._remove)

    @cached_property
    def similar(self) -> Callable[[DenseEmbedding], Awaitable[list[int]]]:
        """Retrieve the most similar embedding keys to the given embedding."""
        return concresce.batch(window=_BATCH_WINDOW)(self._similar)

    @staticmethod
    def _to_rows(embeddings: list[DenseEmbedding]) -> np.ndarray:
        """Convert embeddings into a float32 matrix for FAISS."""
        return np.asarray(embeddings, dtype=np.float32)

    @staticmethod
    def _key(row: np.ndarray) -> int:
        """Derive a deterministic int64 key from the embedding's content."""
        digest = blake3.blake3(row.tobytes()).digest()
        return int.from_bytes(digest[:8], "little", signed=True)

    async def _add(self, embedding: DenseEmbedding) -> int:
        """Add an embedding to the index and return its key."""
        embeddings = await concresce.collect(embedding)
        rows = self._to_rows(embeddings)
        keys = [self._key(row) for row in rows]
        unique = dict(zip(keys, rows, strict=True))
        ids = np.asarray(list(unique), dtype=np.int64)
        self._index.remove_ids(faiss.IDSelectorBatch(ids))
        self._index.add_with_ids(self._to_rows(list(unique.values())), ids)
        return concresce.scatter(keys)

    async def _remove(self, key: int) -> None:
        """Remove an embedding from the index by its key."""
        keys = await concresce.collect(key)
        ids = np.asarray(keys, dtype=np.int64)
        self._index.remove_ids(faiss.IDSelectorBatch(ids))
        return concresce.scatter([None] * len(keys))

    async def _similar(self, embedding: DenseEmbedding) -> list[int]:
        """Retrieve the most similar embedding keys to the given embedding."""
        embeddings = await concresce.collect(embedding)
        _, ids = self._index.search(self._to_rows(embeddings), self.results)
        results = [[int(i) for i in row if i != -1] for row in ids]
        return concresce.scatter(results)


class SparseIndex:
    """Sparse embedding index using inner-product similarity over weight maps."""

    def __init__(self, results: int = 10) -> None:
        """Initialize the sparse index."""
        self.results = results
        self._embeddings: dict[int, SparseEmbedding] = {}

    @cached_property
    def add(self) -> Callable[[SparseEmbedding], Awaitable[int]]:
        """Add an embedding to the index and return its key."""
        return concresce.batch(window=_BATCH_WINDOW)(self._add)

    @cached_property
    def remove(self) -> Callable[[int], Awaitable[None]]:
        """Remove an embedding from the index by its key."""
        return concresce.batch(window=_BATCH_WINDOW)(self._remove)

    @cached_property
    def similar(self) -> Callable[[SparseEmbedding], Awaitable[list[int]]]:
        """Retrieve the most similar embedding keys to the given embedding."""
        return concresce.batch(window=_BATCH_WINDOW)(self._similar)

    @staticmethod
    def _key(embedding: SparseEmbedding) -> int:
        """Derive a deterministic int64 key from the embedding's content."""
        tokens = sorted(embedding)
        weights = [embedding[token] for token in tokens]
        data = np.asarray(tokens, dtype=np.int64).tobytes()
        data += np.asarray(weights, dtype=np.float32).tobytes()
        digest = blake3.blake3(data).digest()
        return int.from_bytes(digest[:8], "little", signed=True)

    @staticmethod
    def _score(query: SparseEmbedding, stored: SparseEmbedding) -> float:
        """Compute the inner product between two weight maps."""
        if len(stored) < len(query):
            query, stored = stored, query
        return sum(weight * stored.get(token, 0.0) for token, weight in query.items())

    def _top_keys(self, query: SparseEmbedding) -> list[int]:
        """Rank stored keys by inner product with the query, dropping misses."""
        scores = {
            key: self._score(query, stored) for key, stored in self._embeddings.items()
        }
        ranked = sorted(scores, key=lambda key: scores[key], reverse=True)
        return [key for key in ranked[: self.results] if scores[key] > 0]

    async def _add(self, embedding: SparseEmbedding) -> int:
        """Add an embedding to the index and return its key."""
        embeddings = await concresce.collect(embedding)
        keys = [self._key(e) for e in embeddings]
        self._embeddings.update(zip(keys, embeddings, strict=True))
        return concresce.scatter(keys)

    async def _remove(self, key: int) -> None:
        """Remove an embedding from the index by its key."""
        keys = await concresce.collect(key)
        for k in keys:
            self._embeddings.pop(k, None)
        return concresce.scatter([None] * len(keys))

    async def _similar(self, embedding: SparseEmbedding) -> list[int]:
        """Retrieve the most similar embedding keys to the given embedding."""
        embeddings = await concresce.collect(embedding)
        results = [self._top_keys(query) for query in embeddings]
        return concresce.scatter(results)
