"""Embedding Index protocol definitions."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import blake3
import concresce
import faiss
import numpy as np
from belljar import Jar

from rager.types import SparseEmbedding

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from rager import DenseEmbedding

logger = logging.getLogger(__name__)


class Index[E, K](Protocol):
    """Protocol for an embedding index."""

    def add(self, embedding: E) -> Awaitable[K]:
        """Add an embedding to the index and return its key."""
        ...

    def remove(self, key: K) -> Awaitable[None]:
        """Remove an embedding from the index by its key."""
        ...

    def similar(self, embedding: E, results: int = 100) -> Awaitable[list[K]]:
        """Retrieve the keys of the ``results`` most similar embeddings."""
        ...


class MemoryDenseIndex:
    """In-memory dense embedding index backed by a flat FAISS inner-product index."""

    def __init__(self, dimensions: int | None = None) -> None:
        """Initialize the index, optionally fixing the embedding dimension."""
        self.dimensions = dimensions
        self._index: faiss.IndexIDMap | None = None

    def _ensure_index(self, dimensions: int) -> faiss.IndexIDMap:
        """Return the index, building it to match the first embedding's width."""
        if self.dimensions is None:
            logger.debug("Inferred embedding dimension %d from first batch", dimensions)
            self.dimensions = dimensions
        if dimensions != self.dimensions:
            message = (
                f"Embedding has {dimensions} dimensions, "
                f"but the index expects {self.dimensions}."
            )
            logger.error(message)
            raise ValueError(message)
        if self._index is None:
            logger.info("Created dense FAISS index with %d dimensions", self.dimensions)
            self._index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dimensions))
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

    @concresce.batch
    async def add(self, embedding: DenseEmbedding) -> int:
        """Add an embedding to the index and return its key."""
        embeddings = await concresce.collect(embedding)
        rows = self._to_rows(embeddings)
        index = self._ensure_index(rows.shape[1])
        keys = [self._key(row) for row in rows]
        unique = dict(zip(keys, rows, strict=True))
        ids = np.asarray(list(unique), dtype=np.int64)
        index.remove_ids(faiss.IDSelectorBatch(ids))
        index.add_with_ids(self._to_rows(list(unique.values())), ids)
        logger.debug(
            "Added %d embeddings (%d unique) to the dense index; total is now %d",
            len(keys),
            len(unique),
            index.ntotal,
        )
        return cast("int", keys)

    @concresce.batch
    async def remove(self, key: int) -> None:
        """Remove an embedding from the index by its key."""
        keys = await concresce.collect(key)
        if self._index is None:
            logger.warning(
                "Ignoring removal of %d keys from an empty dense index", len(keys)
            )
        else:
            ids = np.asarray(keys, dtype=np.int64)
            removed = self._index.remove_ids(faiss.IDSelectorBatch(ids))
            logger.debug(
                "Removed %d of %d keys from the dense index; total is now %d",
                removed,
                len(keys),
                self._index.ntotal,
            )
        return cast("None", [None] * len(keys))

    @concresce.batch
    async def similar(self, embedding: DenseEmbedding, results: int = 100) -> list[int]:
        """Retrieve the keys of the ``results`` most similar embeddings."""
        collected = await concresce.collect((embedding, results))
        embeddings = [query for query, _ in collected]
        counts = [count for _, count in collected]
        if self._index is None:
            logger.warning(
                "Similarity search of %d queries on an empty dense index",
                len(collected),
            )
            return cast("list[int]", [[] for _ in collected])
        logger.debug(
            "Searching the dense index of %d embeddings with %d queries",
            self._index.ntotal,
            len(collected),
        )
        _, ids = self._index.search(self._to_rows(embeddings), max(counts))
        rankings = [
            [int(i) for i in row[:count] if i != -1]
            for row, count in zip(ids, counts, strict=True)
        ]
        return cast("list[int]", rankings)


def _sparse_key(embedding: SparseEmbedding) -> int:
    """Derive a deterministic int64 key from the embedding's content."""
    tokens = sorted(embedding)
    weights = [embedding[token] for token in tokens]
    data = np.asarray(tokens, dtype=np.int64).tobytes()
    data += np.asarray(weights, dtype=np.float32).tobytes()
    digest = blake3.blake3(data).digest()
    return int.from_bytes(digest[:8], "little", signed=True)


def _sparse_score(query: SparseEmbedding, stored: SparseEmbedding) -> float:
    """Compute the inner product between two weight maps."""
    if len(stored) < len(query):
        query, stored = stored, query
    return sum(weight * stored.get(token, 0.0) for token, weight in query.items())


def _top_sparse_keys(
    embeddings: dict[int, SparseEmbedding], query: SparseEmbedding, results: int
) -> list[int]:
    """Rank stored keys by inner product with the query, dropping misses."""
    scores = {key: _sparse_score(query, stored) for key, stored in embeddings.items()}
    ranked = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [key for key in ranked[:results] if scores[key] > 0]


class MemorySparseIndex:
    """In-memory sparse index using inner-product similarity over weight maps."""

    def __init__(self) -> None:
        """Initialize the sparse index."""
        self._embeddings: dict[int, SparseEmbedding] = {}

    @concresce.batch
    async def add(self, embedding: SparseEmbedding) -> int:
        """Add an embedding to the index and return its key."""
        embeddings = await concresce.collect(embedding)
        keys = [_sparse_key(e) for e in embeddings]
        self._embeddings.update(zip(keys, embeddings, strict=True))
        logger.debug(
            "Added %d embeddings to the memory sparse index; total is now %d",
            len(keys),
            len(self._embeddings),
        )
        return cast("int", keys)

    @concresce.batch
    async def remove(self, key: int) -> None:
        """Remove an embedding from the index by its key."""
        keys = await concresce.collect(key)
        removed = 0
        for k in keys:
            removed += self._embeddings.pop(k, None) is not None
        logger.debug(
            "Removed %d of %d keys from the memory sparse index; total is now %d",
            removed,
            len(keys),
            len(self._embeddings),
        )
        return cast("None", [None] * len(keys))

    @concresce.batch
    async def similar(
        self, embedding: SparseEmbedding, results: int = 100
    ) -> list[int]:
        """Retrieve the keys of the ``results`` most similar embeddings."""
        collected = await concresce.collect((embedding, results))
        if not self._embeddings:
            logger.warning(
                "Similarity search of %d queries on an empty memory sparse index",
                len(collected),
            )
        else:
            logger.debug(
                "Searching the memory sparse index of %d embeddings with %d queries",
                len(self._embeddings),
                len(collected),
            )
        rankings = [
            _top_sparse_keys(self._embeddings, query, count)
            for query, count in collected
        ]
        return cast("list[int]", rankings)


class FileSparseIndex:
    """JAR-based sparse index using inner-product similarity over weight maps.

    Embeddings are sealed on disk in a JAR, so only their keys stay in memory.
    """

    def __init__(self) -> None:
        """Initialize the index with no keys."""
        self._keys: set[int] = set()

    def _jar(self, key: int) -> Jar[SparseEmbedding]:
        """Open a jar positioned at the identity of the given key."""
        jar = Jar[SparseEmbedding](Path(".jar/indexes"))
        jar.include(_sparse_key.__code__)
        jar.include(key)
        return jar

    def _load(self) -> dict[int, SparseEmbedding]:
        """Load the sealed embeddings, dropping keys whose seal is gone."""
        embeddings: dict[int, SparseEmbedding] = {}
        for key in self._keys:
            embedding = self._jar(key).get()
            if embedding is None:
                logger.debug("No embedding found in JAR for key %r", key)
            else:
                embeddings[key] = embedding
        return embeddings

    @concresce.batch
    async def add(self, embedding: SparseEmbedding) -> int:
        """Add an embedding to the index and return its key."""
        embeddings = await concresce.collect(embedding)
        keys = [_sparse_key(e) for e in embeddings]
        for key, stored in zip(keys, embeddings, strict=True):
            self._jar(key).set(stored)
        self._keys.update(keys)
        logger.debug(
            "Added %d embeddings to the file sparse index; total is now %d",
            len(keys),
            len(self._keys),
        )
        return cast("int", keys)

    @concresce.batch
    async def remove(self, key: int) -> None:
        """Remove an embedding from the index by its key."""
        keys = await concresce.collect(key)
        removed = 0
        for k in keys:
            removed += k in self._keys
            self._keys.discard(k)
        logger.debug(
            "Removed %d of %d keys from the file sparse index; total is now %d",
            removed,
            len(keys),
            len(self._keys),
        )
        return cast("None", [None] * len(keys))

    @concresce.batch
    async def similar(
        self, embedding: SparseEmbedding, results: int = 100
    ) -> list[int]:
        """Retrieve the keys of the ``results`` most similar embeddings."""
        collected = await concresce.collect((embedding, results))
        if not self._keys:
            logger.warning(
                "Similarity search of %d queries on an empty file sparse index",
                len(collected),
            )
        else:
            logger.debug(
                "Searching the file sparse index of %d embeddings with %d queries",
                len(self._keys),
                len(collected),
            )
        embeddings = self._load()
        rankings = [
            _top_sparse_keys(embeddings, query, count) for query, count in collected
        ]
        return cast("list[int]", rankings)
