"""Embedding Index protocol definitions."""

import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Hashable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, override

import concresce
import dill
import faiss
import numpy as np
from belljar import Jar
from blake3 import blake3

from rager.types import SparseEmbedding

if TYPE_CHECKING:
    from rager.stores import Store


logger = logging.getLogger(__name__)


class Index[K, E](Protocol):
    """Protocol for an embedding index."""

    def keys(self) -> Awaitable[list[K]]:
        """Return the keys in the index."""
        ...

    def get(self, key: K) -> Awaitable[E | None]:
        """Retrieve an embedding by its key, or None if absent."""
        ...

    def set(self, key: K, value: E) -> Awaitable[None]:
        """Store an embedding with the given key."""
        ...

    def remove(self, key: K) -> Awaitable[None]:
        """Remove an embedding by its key."""
        ...

    def similar(self, embedding: E, embedding_results: int = 100) -> Awaitable[list[K]]:
        """Retrieve the keys of the most similar embeddings."""
        ...


class BaseIndex[K, E](ABC):
    """Abstract base class for embedding indexes."""

    @abstractmethod
    def _jar(self) -> Jar[list[K]]:
        """Open a jar positioned at the identity of the index."""

    @abstractmethod
    def keys(self) -> Awaitable[list[K]]:
        """Return the keys in the index."""
        ...

    @abstractmethod
    def _batched_get(self, keys: list[K]) -> Awaitable[list[E | None]]:
        """Retrieve embeddings for multiple keys."""
        ...

    @abstractmethod
    def _batched_set(self, keys: list[K], values: list[E]) -> Awaitable[None]:
        """Store embeddings with the given keys."""
        ...

    @abstractmethod
    def _batched_remove(self, keys: list[K]) -> Awaitable[None]:
        """Remove embeddings for multiple keys."""
        ...

    @abstractmethod
    def _batched_similar(
        self, embeddings: list[E], results: list[int]
    ) -> Awaitable[list[list[K]]]:
        """Retrieve similar keys for multiple embeddings."""
        ...

    @abstractmethod
    def get(self, key: K) -> Awaitable[E | None]:
        """Retrieve an embedding by its key, or None if absent."""

    @abstractmethod
    def set(self, key: K, value: E) -> Awaitable[None]:
        """Store an embedding with the given key."""

    @abstractmethod
    def remove(self, key: K) -> Awaitable[None]:
        """Remove an embedding by its key."""

    @abstractmethod
    def _similar(self, embedding: E, embedding_results: int) -> Awaitable[list[K]]:
        """Retrieve the keys of the most similar embeddings."""

    async def _collect_get(self, key: K) -> E | None:
        """Pool a get into the current batch and return this caller's result."""
        keys = await concresce.collect(key)
        results = await self._batched_get(list(keys))
        return concresce.scatter(results)

    async def _collect_set(self, key: K, value: E) -> None:
        """Pool a set into the current batch."""
        collected = await concresce.collect((key, value))
        keys = [k for k, _ in collected]
        values = [v for _, v in collected]
        await self._batched_set(keys, values)
        return concresce.scatter([None] * len(keys))

    async def _collect_remove(self, key: K) -> None:
        """Pool a remove into the current batch."""
        keys = await concresce.collect(key)
        await self._batched_remove(list(keys))
        return concresce.scatter([None] * len(keys))

    async def _collect_similar(self, embedding: E, embedding_results: int) -> list[K]:
        """Pool a similarity query into the current batch."""
        collected = await concresce.collect((embedding, embedding_results))
        embeddings = [emb for emb, _ in collected]
        results = [res for _, res in collected]
        rankings = await self._batched_similar(embeddings, results)
        return concresce.scatter(rankings)

    async def similar(self, embedding: E, embedding_results: int = 100) -> list[K]:
        """Retrieve the keys of the most similar embeddings."""
        jar = self._jar()
        jar.include(embedding)
        jar.include(embedding_results)
        if (cached := jar.get()) is not None:
            return cached
        result = await self._similar(embedding, embedding_results)
        return jar.set(result)


class FaissIndex[K: Hashable, E](BaseIndex[K, E]):
    """Index for embeddings using FAISS."""

    def __init__(self, dimensions: int, key_map: Store[K, int]) -> None:
        """Initialize the index with the given dimensions."""
        self._dimensions = dimensions
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self._dimensions))
        self._key_map = key_map
        self._vectors: dict[int, E] = {}
        self._instance = uuid.uuid4()

    @override
    @concresce.batch
    async def get(self, key: K) -> E | None:
        """Retrieve an embedding by its key, or None if absent."""
        return await self._collect_get(key)

    @override
    @concresce.batch
    async def set(self, key: K, value: E) -> None:
        """Store an embedding with the given key."""
        return await self._collect_set(key, value)

    @override
    @concresce.batch
    async def remove(self, key: K) -> None:
        """Remove an embedding by its key."""
        return await self._collect_remove(key)

    @override
    @concresce.batch
    async def _similar(self, embedding: E, embedding_results: int = 100) -> list[K]:
        """Retrieve the keys of the most similar embeddings."""
        return await self._collect_similar(embedding, embedding_results)

    @override
    def _jar(self) -> Jar[list[K]]:
        """Open a jar positioned at the identity of the index."""
        jar = Jar[list[K]](Path(".jar/indexes"))
        jar.include(self.keys.__code__)
        jar.include(self.get.__code__)
        jar.include(self.remove.__code__)
        jar.include(self._jar.__code__)
        jar.include(self._instance)
        jar.include(self._key_map.keys())
        return jar

    @staticmethod
    def _id(key: object) -> int:
        """Derive a deterministic int64 FAISS id from the key's content."""
        digest = blake3(dill.dumps(key)).digest()
        return int.from_bytes(digest[:8], "little", signed=True)

    def _to_rows(self, embeddings: list[E]) -> np.ndarray:
        """Convert embeddings into a float32 matrix, validating their width."""
        rows = np.asarray(embeddings, dtype=np.float32)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.shape[1] != self._dimensions:
            message = (
                f"Embedding has {rows.shape[1]} dimensions, "
                f"but the index expects {self._dimensions}."
            )
            logger.error(message)
            raise ValueError(message)
        return rows

    @override
    async def keys(self) -> list[K]:
        """Return the keys in the index."""
        return list(self._key_map.keys())

    @override
    async def _batched_get(self, keys: list[K]) -> list[E | None]:
        """Retrieve embeddings for multiple keys."""
        if not keys:
            return []
        results: list[E | None] = []
        for key in keys:
            if self._key_map.get(key) is None:
                results.append(None)
                continue
            results.append(self._vectors.get(self._id(key)))
        return results

    @override
    async def _batched_set(self, keys: list[K], values: list[E]) -> None:
        """Store embeddings with the given keys."""
        rows = self._to_rows(values)
        ids = np.array([self._id(key) for key in keys], dtype=np.int64)
        self.index.remove_ids(faiss.IDSelectorBatch(ids))
        self.index.add_with_ids(rows, ids)
        for key, value, faiss_id in zip(keys, values, ids, strict=True):
            self._key_map.set(key, int(faiss_id))
            self._vectors[int(faiss_id)] = value

    @override
    async def _batched_remove(self, keys: list[K]) -> None:
        """Remove embeddings for multiple keys."""
        ids = np.array([self._id(key) for key in keys], dtype=np.int64)
        self.index.remove_ids(faiss.IDSelectorBatch(ids))
        for key in keys:
            self._key_map.remove(key)
            self._vectors.pop(self._id(key), None)

    @override
    async def _batched_similar(
        self, embeddings: list[E], results: list[int]
    ) -> list[list[K]]:
        """Retrieve similar keys for multiple embeddings."""
        rows = self._to_rows(embeddings)
        top_k = max(results, default=0)
        if top_k == 0 or self.index.ntotal == 0:
            return [[] for _ in results]
        _, ids = self.index.search(rows, min(top_k, self.index.ntotal))
        reverse = {
            faiss_id: key
            for key in self._key_map
            if (faiss_id := self._key_map.get(key)) is not None
        }
        rankings: list[list[K]] = []
        for row_ids, k in zip(ids, results, strict=True):
            ranking: list[K] = []
            for faiss_id in row_ids[:k]:
                if faiss_id == -1:
                    continue
                if (key := reverse.get(int(faiss_id))) is not None:
                    ranking.append(key)
            rankings.append(ranking)
        return rankings


def _scores[K: Hashable](
    postings: dict[int, dict[K, float]], query: SparseEmbedding
) -> dict[K, float]:
    """Accumulate inner-product scores for a query."""
    scores: dict[K, float] = {}
    for token, weight in query.items():
        for key, posted in postings.get(token, {}).items():
            scores[key] = scores.get(key, 0.0) + weight * posted
    return scores


def _top_keys[K](scores: dict[K, float], count: int) -> list[K]:
    """Return the highest-scoring keys, best first, dropping non-positive scores."""
    positive = {key: score for key, score in scores.items() if score > 0}
    return sorted(positive, key=positive.__getitem__, reverse=True)[:count]


class SparseIndex[K: Hashable](BaseIndex[K, SparseEmbedding]):
    """Sparse index using inner-product similarity over weight maps."""

    def __init__(
        self,
        embedding_map: Store[K, SparseEmbedding],
        token_map: Store[int, dict[K, float]],
    ) -> None:
        """Initialize the index with the given embedding and token stores."""
        self._embedding_map = embedding_map
        self._token_map = token_map
        self._instance = uuid.uuid4()

    @override
    @concresce.batch
    async def get(self, key: K) -> SparseEmbedding | None:
        """Retrieve an embedding by its key, or None if absent."""
        return await self._collect_get(key)

    @override
    @concresce.batch
    async def set(self, key: K, value: SparseEmbedding) -> None:
        """Store an embedding with the given key."""
        return await self._collect_set(key, value)

    @override
    @concresce.batch
    async def remove(self, key: K) -> None:
        """Remove an embedding by its key."""
        return await self._collect_remove(key)

    @override
    @concresce.batch
    async def _similar(
        self, embedding: SparseEmbedding, embedding_results: int = 100
    ) -> list[K]:
        """Retrieve the keys of the most similar embeddings."""
        return await self._collect_similar(embedding, embedding_results)

    def _postings(self, token: int) -> dict[K, float]:
        """Return the token's posting list, or an empty one."""
        try:
            return self._token_map[token] or {}
        except KeyError:
            return {}

    def _drop(self, token: int, key: K) -> None:
        """Remove the key from the token's posting list."""
        try:
            postings = self._token_map[token] or {}
        except KeyError:
            return
        postings.pop(key, None)
        if postings:
            self._token_map[token] = postings
        else:
            del self._token_map[token]

    @override
    def _jar(self) -> Jar[list[K]]:
        """Open a jar positioned at the identity of the index and its contents."""
        jar = Jar[list[K]](Path(".jar/indexes"))
        jar.include(self.keys.__code__)
        jar.include(self.get.__code__)
        jar.include(self.remove.__code__)
        jar.include(self._jar.__code__)
        jar.include(self._instance)
        jar.include(self._embedding_map)
        jar.include(self._token_map)
        return jar

    @override
    async def keys(self) -> list[K]:
        """Return the keys in the store."""
        return self._embedding_map.keys()

    @override
    async def _batched_get(self, keys: list[K]) -> list[SparseEmbedding | None]:
        """Retrieve embeddings for multiple keys."""
        results: list[SparseEmbedding | None] = []
        for key in keys:
            try:
                results.append(self._embedding_map[key])
            except KeyError:
                results.append(None)
        return results

    @override
    async def _batched_set(self, keys: list[K], values: list[SparseEmbedding]) -> None:
        """Store embeddings with the given keys."""
        for key, value in zip(keys, values, strict=True):
            try:
                stale = self._embedding_map[key] or {}
            except KeyError:
                stale = {}
            for token in stale.keys() - value.keys():
                self._drop(token, key)
            for token, weight in value.items():
                postings = self._postings(token)
                postings[key] = weight
                self._token_map[token] = postings
            self._embedding_map[key] = value

    @override
    async def _batched_remove(self, keys: list[K]) -> None:
        """Remove embeddings for multiple keys."""
        for key in keys:
            try:
                embedding = self._embedding_map[key] or {}
            except KeyError:
                continue
            for token in embedding:
                self._drop(token, key)
            del self._embedding_map[key]

    @override
    async def _batched_similar(
        self, embeddings: list[SparseEmbedding], results: list[int]
    ) -> list[list[K]]:
        """Retrieve similar keys for multiple embeddings."""
        if not self._embedding_map.keys():
            logger.warning(
                "Similarity search of %d queries on an empty sparse index",
                len(embeddings),
            )
            return [[] for _ in embeddings]
        tokens = {token for query in embeddings for token in query}
        logger.debug(
            "Searching the sparse index with %d queries over %d tokens",
            len(embeddings),
            len(tokens),
        )
        postings = {token: self._postings(token) for token in tokens}
        return [
            _top_keys(_scores(postings, query), count)
            for query, count in zip(embeddings, results, strict=True)
        ]
