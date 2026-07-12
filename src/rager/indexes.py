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
    """Protocol for an embedding index.

    Similarity is measured by inner product, which equals cosine similarity
    only when the embeddings are L2-normalized. Normalize your embeddings
    before adding them if you want cosine ranking.
    """

    async def add(self, embedding: E) -> K:
        """Add an embedding to the index and return its key."""
        ...

    async def remove(self, key: K) -> None:
        """Remove an embedding from the index by its key."""
        ...

    async def similar(self, embedding: E, results: int = 100) -> list[K]:
        """Retrieve the keys of the ``results`` most similar embeddings."""
        ...


class DenseIndex:
    """Dense embedding index backed by a flat FAISS inner-product index.

    Ranking uses inner product, so it matches cosine similarity only for
    L2-normalized embeddings (as produced by the bundled dense embedder).

    The embedding dimension is inferred from the first embedding added. Pass
    ``dimensions`` to validate that every embedding matches an expected width.
    """

    def __init__(self, dimensions: int | None = None) -> None:
        """Initialize the index, optionally fixing the embedding dimension."""
        self.dimensions = dimensions
        self._index: faiss.IndexIDMap2 | None = None

    @cached_property
    def add(self) -> Callable[[DenseEmbedding], Awaitable[int]]:
        """Add an embedding to the index and return its key."""
        return concresce.batch(window=timedelta(milliseconds=1))(self._add)

    @cached_property
    def remove(self) -> Callable[[int], Awaitable[None]]:
        """Remove an embedding from the index by its key."""
        return concresce.batch(window=timedelta(milliseconds=1))(self._remove)

    async def similar(self, embedding: DenseEmbedding, results: int = 100) -> list[int]:
        """Retrieve the keys of the ``results`` most similar embeddings."""
        return await self._similar(embedding, results)

    @cached_property
    def _similar(self) -> Callable[[DenseEmbedding, int], Awaitable[list[int]]]:
        """Coalesce concurrent queries into one batched FAISS search."""
        return concresce.batch(window=timedelta(milliseconds=1))(self._similar_batch)

    def _ensure_index(self, dimensions: int) -> faiss.IndexIDMap2:
        """Return the index, building it to match the first embedding's width."""
        if self.dimensions is None:
            self.dimensions = dimensions
        if dimensions != self.dimensions:
            message = (
                f"Embedding has {dimensions} dimensions, "
                f"but the index expects {self.dimensions}."
            )
            raise ValueError(message)
        if self._index is None:
            self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dimensions))
        return self._index

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
        index = self._ensure_index(rows.shape[1])
        keys = [self._key(row) for row in rows]
        unique = dict(zip(keys, rows, strict=True))
        ids = np.asarray(list(unique), dtype=np.int64)
        index.remove_ids(faiss.IDSelectorBatch(ids))
        index.add_with_ids(self._to_rows(list(unique.values())), ids)
        return concresce.scatter(keys)

    async def _remove(self, key: int) -> None:
        """Remove an embedding from the index by its key."""
        keys = await concresce.collect(key)
        if self._index is not None:
            ids = np.asarray(keys, dtype=np.int64)
            self._index.remove_ids(faiss.IDSelectorBatch(ids))
        return concresce.scatter([None] * len(keys))

    async def _similar_batch(
        self, embedding: DenseEmbedding, results: int
    ) -> list[int]:
        """Retrieve the most similar embedding keys to the given embedding."""
        collected = await concresce.collect((embedding, results))
        embeddings = [query for query, _ in collected]
        counts = [count for _, count in collected]
        if self._index is None:
            return concresce.scatter([[] for _ in collected])
        _, ids = self._index.search(self._to_rows(embeddings), max(counts))
        rankings = [
            [int(i) for i in row[:count] if i != -1]
            for row, count in zip(ids, counts, strict=True)
        ]
        return concresce.scatter(rankings)


class SparseIndex:
    """Sparse embedding index using inner-product similarity over weight maps."""

    def __init__(self) -> None:
        """Initialize the sparse index."""
        self._embeddings: dict[int, SparseEmbedding] = {}

    @cached_property
    def add(self) -> Callable[[SparseEmbedding], Awaitable[int]]:
        """Add an embedding to the index and return its key."""
        return concresce.batch(window=timedelta(milliseconds=1))(self._add)

    @cached_property
    def remove(self) -> Callable[[int], Awaitable[None]]:
        """Remove an embedding from the index by its key."""
        return concresce.batch(window=timedelta(milliseconds=1))(self._remove)

    async def similar(
        self, embedding: SparseEmbedding, results: int = 100
    ) -> list[int]:
        """Retrieve the keys of the ``results`` most similar embeddings."""
        return await self._similar(embedding, results)

    @cached_property
    def _similar(self) -> Callable[[SparseEmbedding, int], Awaitable[list[int]]]:
        """Coalesce concurrent queries into one batched ranking pass."""
        return concresce.batch(window=timedelta(milliseconds=1))(self._similar_batch)

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

    def _top_keys(self, query: SparseEmbedding, results: int) -> list[int]:
        """Rank stored keys by inner product with the query, dropping misses."""
        scores = {
            key: self._score(query, stored) for key, stored in self._embeddings.items()
        }
        ranked = sorted(scores, key=lambda key: scores[key], reverse=True)
        return [key for key in ranked[:results] if scores[key] > 0]

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

    async def _similar_batch(
        self, embedding: SparseEmbedding, results: int
    ) -> list[int]:
        """Retrieve the most similar embedding keys to the given embedding."""
        collected = await concresce.collect((embedding, results))
        rankings = [self._top_keys(query, count) for query, count in collected]
        return concresce.scatter(rankings)
