"""Embedding Index protocol definitions."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, override

import concresce
import dill
import faiss
import numpy as np
from belljar import Jar
from blake3 import blake3

from rager.stores import BaseStore, Store
from rager.types import SparseEmbedding

if TYPE_CHECKING:
    from collections.abc import Awaitable, Hashable


logger = logging.getLogger(__name__)


class Index[K, E](Store[K, E], Protocol):
    """Protocol for an embedding index."""

    def similar(self, embedding: E, embedding_results: int) -> Awaitable[list[K]]:
        """Retrieve the keys of the ``embedding_results`` most similar embeddings."""
        ...


class FaissIndex[K: Hashable, E](BaseStore[K, E]):
    """Index for embeddings using FAISS.

    Similarity search runs on an in-memory FAISS index, while the embeddings
    themselves are sealed on disk in a JAR, in the same way as ``FileStore``.
    """

    def __init__(self, dimensions: int, key_map: Store[K, int]) -> None:
        """Initialize the index with the given dimensions."""
        self._dimensions = dimensions
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self._dimensions))
        self._key_map = key_map

    def _jar(self, key: K, faiss_id: int) -> Jar[E]:
        """Open a jar positioned at the identity of the given key and FAISS id."""
        jar = Jar[E](Path(".jar/indexes"))
        jar.include(self.keys.__code__)
        jar.include(self.__setitem__.__code__)
        jar.include(self.__getitem__.__code__)
        jar.include(self.__delitem__.__code__)
        jar.include(self._jar.__code__)
        jar.include(key)
        jar.include(faiss_id)
        return jar

    @staticmethod
    def _id(key: object) -> int:
        """Derive a deterministic int64 FAISS id from the key's content."""
        digest = blake3(dill.dumps(key)).digest()
        return int.from_bytes(digest[:8], "little", signed=True)

    def _to_rows(self, embeddings: list[E]) -> np.ndarray:
        """Convert embeddings into a float32 matrix, validating their width."""
        rows = np.asarray(embeddings, dtype=np.float32)
        if rows.shape[1] != self._dimensions:
            message = (
                f"Embedding has {rows.shape[1]} dimensions, "
                f"but the index expects {self._dimensions}."
            )
            logger.error(message)
            raise ValueError(message)
        return rows

    @override
    def keys(self) -> list[K]:
        """Return the keys in the store."""
        return self._key_map.keys()

    @override
    def __setitem__(self, key: K, value: E) -> None:
        """Store an embedding with the given key."""
        logger.debug("Storing embedding for key %r", key)
        rows = self._to_rows([value])
        faiss_id = self._id(key)
        ids = np.asarray([faiss_id], dtype=np.int64)
        self.index.remove_ids(faiss.IDSelectorBatch(ids))
        self.index.add_with_ids(rows, ids)
        self._jar(key, faiss_id).set(value)
        self._key_map[key] = faiss_id

    @override
    def __getitem__(self, key: K) -> E:
        """Retrieve an embedding by its key."""
        logger.debug("Retrieving embedding for key %r", key)
        faiss_id = self._key_map[key]
        return cast("E", self._jar(key, faiss_id).get())

    @override
    def __delitem__(self, key: K) -> None:
        """Remove an embedding by its key."""
        logger.debug("Removing embedding for key %r", key)
        ids = np.asarray([self._key_map[key]], dtype=np.int64)
        self.index.remove_ids(faiss.IDSelectorBatch(ids))
        del self._key_map[key]

    @concresce.batch
    async def similar(self, embedding: E, embedding_results: int = 100) -> list[K]:
        """Retrieve the keys of the ``embedding_results`` most similar embeddings."""
        collected = await concresce.collect((embedding, embedding_results))
        if not self.index.ntotal:
            logger.warning(
                "Similarity search of %d queries on an empty FAISS index",
                len(collected),
            )
            return concresce.scatter([[] for _ in collected])
        logger.debug(
            "Searching the FAISS index of %d embeddings with %d queries",
            self.index.ntotal,
            len(collected),
        )
        queries = self._to_rows([query for query, _ in collected])
        counts = [count for _, count in collected]
        _, ids = self.index.search(queries, max(counts))
        stored = self._key_map.keys()
        key_by_id = {self._key_map[key]: key for key in stored}
        rankings = [
            [key_by_id[int(i)] for i in row[:count] if int(i) in key_by_id]
            for row, count in zip(ids, counts, strict=True)
        ]
        return concresce.scatter(rankings)


def _scores[K](
    postings: dict[int, dict[K, float]], query: SparseEmbedding
) -> dict[K, float]:
    """Accumulate inner-product scores for every stored key matching the query."""
    scores: dict[K, float] = {}
    for token, weight in query.items():
        for key, stored in postings[token].items():
            scores[key] = scores.get(key, 0.0) + weight * stored
    return scores


def _top_keys[K](scores: dict[K, float], results: int) -> list[K]:
    """Rank keys by score, dropping non-positive matches."""
    ranked = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [key for key in ranked[:results] if scores[key] > 0]


class SparseIndex[K: Hashable](BaseStore[K, SparseEmbedding]):
    """Sparse index using inner-product similarity over weight maps.

    The embeddings and the inverted token index are delegated to the injected
    stores, so the caller decides whether they live in memory or on disk.
    Similarity search only loads the posting lists of the query's tokens,
    never the full set of stored embeddings.
    """

    def __init__(
        self,
        embedding_map: Store[K, SparseEmbedding],
        token_map: Store[int, dict[K, float]],
    ) -> None:
        """Initialize the index with the given embedding and token stores."""
        self._embedding_map = embedding_map
        self._token_map = token_map

    def _postings(self, token: int) -> dict[K, float]:
        """Return the token's posting list, or an empty one if it has none."""
        try:
            return self._token_map[token] or {}
        except KeyError:
            return {}

    def _drop(self, token: int, key: K) -> None:
        """Remove the key from the token's posting list, dropping empty lists."""
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
    def keys(self) -> list[K]:
        """Return the keys in the store."""
        return self._embedding_map.keys()

    @override
    def __setitem__(self, key: K, value: SparseEmbedding) -> None:
        """Store an embedding with the given key."""
        logger.debug("Storing embedding for key %r", key)
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
    def __getitem__(self, key: K) -> SparseEmbedding:
        """Retrieve an embedding by its key."""
        logger.debug("Retrieving embedding for key %r", key)
        return self._embedding_map[key]

    @override
    def __delitem__(self, key: K) -> None:
        """Remove an embedding by its key."""
        logger.debug("Removing embedding for key %r", key)
        embedding = self._embedding_map[key] or {}
        for token in embedding:
            self._drop(token, key)
        del self._embedding_map[key]

    @concresce.batch
    async def similar(
        self, embedding: SparseEmbedding, embedding_results: int = 100
    ) -> list[K]:
        """Retrieve the keys of the ``embedding_results`` most similar embeddings."""
        collected = await concresce.collect((embedding, embedding_results))
        if not self._embedding_map.keys():
            logger.warning(
                "Similarity search of %d queries on an empty sparse index",
                len(collected),
            )
            return concresce.scatter([[] for _ in collected])
        tokens = {token for query, _ in collected for token in query}
        logger.debug(
            "Searching the sparse index with %d queries over %d tokens",
            len(collected),
            len(tokens),
        )
        postings = {token: self._postings(token) for token in tokens}
        rankings = [
            _top_keys(_scores(postings, query), count) for query, count in collected
        ]
        return concresce.scatter(rankings)
