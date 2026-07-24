"""Embedders for converting text chunks into vector representations."""

import logging
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, override

import concresce
from belljar import Jar
from sentence_transformers import SentenceTransformer, SparseEncoder

from rager.types import DenseEmbedding, SparseEmbedding

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterable, Iterator

    from torch import Tensor


logger = logging.getLogger(__name__)


class Embedder[E](Protocol):
    """Protocol for embedders."""

    def embed(self, chunk: str) -> Awaitable[E]:
        """Convert a text chunk into a vector representation."""
        ...


class BaseEmbedder[E](ABC):
    """Abstract helper base deriving the ``Embedder`` protocol from two operations.

    Subclasses implement ``_jar`` and ``_batched``; ``embed`` seals every
    encoded embedding in the jar and returns it from there on later calls.
    """

    @abstractmethod
    def _jar(self, chunk: str) -> Jar[E]:
        """Open a jar positioned at the identity of the given chunk."""

    @abstractmethod
    def _batched(self, chunks: list[str]) -> Awaitable[list[E]]:
        """Convert a batch of text chunks into vector representations."""

    @abstractmethod
    def _embed(self, chunk: str) -> Awaitable[E]:
        """Convert a single text chunk into a vector representation."""

    async def _collect(self, chunk: str) -> E:
        """Pool a chunk into the current batch and return its embedding."""
        chunks = await concresce.collect(chunk)
        embeddings = await self._batched(chunks)
        return concresce.scatter(embeddings)

    async def embed(self, chunk: str) -> E:
        """Convert a text chunk into a vector representation."""
        jar = self._jar(chunk)
        if (cached := jar.get()) is not None:
            return cached
        logger.debug("Cache miss; embedding chunk of %d characters", len(chunk))
        embedding = await self._embed(chunk)
        return jar.set(embedding)


class SentenceTransformerEmbedder(BaseEmbedder[DenseEmbedding]):
    """Dense embedding using SentenceTransformer."""

    def __init__(self, model_name: str) -> None:
        """Initialize the dense embedder with a specific model."""
        self.model_name = model_name

    @cached_property
    def model(self) -> SentenceTransformer:
        """Load the SentenceTransformer model."""
        logger.info("Loading SentenceTransformer model %r", self.model_name)
        return SentenceTransformer(self.model_name)

    @override
    @concresce.batch
    async def _embed(self, chunk: str) -> DenseEmbedding:
        """Convert a single text chunk into a vector representation."""
        return await self._collect(chunk)

    @override
    def _jar(self, chunk: str) -> Jar[DenseEmbedding]:
        """Open a jar positioned at the identity of the given chunk."""
        jar = Jar[DenseEmbedding](Path(".jar/embedders"))
        jar.include(self._jar.__code__)
        jar.include(self.model_name)
        jar.include(chunk)
        return jar

    @override
    async def _batched(self, chunks: list[str]) -> list[DenseEmbedding]:
        """Convert a batch of text chunks into vector representations."""
        logger.debug(
            "Encoding batch of %d chunks with %r", len(chunks), self.model_name
        )
        return self.model.encode(chunks, normalize_embeddings=True).tolist()


class SpladeEmbedder(BaseEmbedder[SparseEmbedding]):
    """Sparse embedding using a SPLADE SparseEncoder."""

    def __init__(self, model_name: str) -> None:
        """Initialize the sparse embedder with a specific model."""
        self.model_name = model_name

    @cached_property
    def model(self) -> SparseEncoder:
        """Load the SPLADE SparseEncoder model."""
        logger.info("Loading SPLADE SparseEncoder model %r", self.model_name)
        return SparseEncoder(self.model_name)

    @staticmethod
    def _decode_entries(coalesced: Tensor) -> Iterator[tuple[int, int, float]]:
        """Yield (row, token_id, weight) triples from a coalesced sparse tensor."""
        indices = coalesced.indices()
        values = coalesced.values()
        for row, token, weight in zip(indices[0], indices[1], values, strict=True):
            yield int(row), int(token), float(weight)

    @staticmethod
    def _group_by_row(
        entries: Iterable[tuple[int, int, float]], count: int
    ) -> list[SparseEmbedding]:
        """Bucket (row, token_id, weight) triples into one weight map per row."""
        embeddings: list[SparseEmbedding] = [{} for _ in range(count)]
        for row, token, weight in entries:
            embeddings[row][token] = weight
        return embeddings

    @classmethod
    def _coalesced_to_embeddings(
        cls, coalesced: Tensor, count: int
    ) -> list[SparseEmbedding]:
        """Split a coalesced sparse tensor into one weight map per input row."""
        return cls._group_by_row(cls._decode_entries(coalesced), count)

    @override
    @concresce.batch
    async def _embed(self, chunk: str) -> SparseEmbedding:
        """Convert a single text chunk into a sparse vector representation."""
        return await self._collect(chunk)

    @override
    def _jar(self, chunk: str) -> Jar[SparseEmbedding]:
        """Open a jar positioned at the identity of the given chunk."""
        jar = Jar[SparseEmbedding](Path(".jar/embedders"))
        jar.include(self._jar.__code__)
        jar.include(self.model_name)
        jar.include(chunk)
        return jar

    @override
    async def _batched(self, chunks: list[str]) -> list[SparseEmbedding]:
        """Convert a batch of text chunks into sparse vector representations."""
        logger.debug(
            "Encoding batch of %d chunks with %r", len(chunks), self.model_name
        )
        coalesced = self.model.encode(chunks).coalesce()
        return self._coalesced_to_embeddings(coalesced, len(chunks))
