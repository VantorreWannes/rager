"""Embedders for converting text chunks into vector representations."""

import logging
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import concresce
from belljar import Jar
from sentence_transformers import SentenceTransformer, SparseEncoder

from rager.types import DenseEmbedding, SparseEmbedding

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from torch import Tensor


logger = logging.getLogger(__name__)


class Embedder[E](Protocol):
    """Protocol for embedders."""

    async def embed(self, chunk: str) -> E:
        """Convert a text chunk into a vector representation."""
        ...


class SentenceTransformerDenseEmbedder:
    """Dense embedding using SentenceTransformer."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the dense embedder with a specific model."""
        self.model_name = model_name

    @cached_property
    def model(self) -> SentenceTransformer:
        """Load the SentenceTransformer model."""
        logger.info("Loading SentenceTransformer model %r", self.model_name)
        return SentenceTransformer(self.model_name)

    @concresce.batch
    async def _encode(self, chunk: str) -> DenseEmbedding:
        """Convert a text chunk into a vector representation."""
        chunks = await concresce.collect(chunk)
        logger.debug(
            "Encoding batch of %d chunks with %r", len(chunks), self.model_name
        )
        embeddings = self.model.encode(chunks, normalize_embeddings=True).tolist()
        return cast("DenseEmbedding", embeddings)

    async def embed(self, chunk: str) -> DenseEmbedding:
        """Convert a text chunk into a vector representation."""
        jar = Jar[DenseEmbedding](Path(".jar/embedders"))
        jar.include(self.embed.__code__)
        jar.include(self.model_name)
        jar.include(chunk)
        if (cached := jar.get()) is not None:
            return cached
        logger.debug("Cache miss; embedding chunk of %d characters", len(chunk))
        return jar.set(await self._encode(chunk))


class SpladeSparseEmbedder:
    """Sparse embedding using a SPLADE SparseEncoder."""

    def __init__(self, model_name: str = "prithivida/Splade_PP_en_v1") -> None:
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

    @concresce.batch
    async def _encode(self, chunk: str) -> SparseEmbedding:
        """Convert a text chunk into a sparse vector representation."""
        chunks = await concresce.collect(chunk)
        logger.debug(
            "Encoding batch of %d chunks with %r", len(chunks), self.model_name
        )
        coalesced = self.model.encode(chunks).coalesce()
        embeddings = self._coalesced_to_embeddings(coalesced, len(chunks))
        return cast("SparseEmbedding", embeddings)

    async def embed(self, chunk: str) -> SparseEmbedding:
        """Convert a text chunk into a sparse vector representation."""
        jar = Jar[SparseEmbedding](Path(".jar/embedders"))
        jar.include(self.embed.__code__)
        jar.include(self.model_name)
        jar.include(chunk)
        if (cached := jar.get()) is not None:
            return cached
        logger.debug("Cache miss; embedding chunk of %d characters", len(chunk))
        return jar.set(await self._encode(chunk))
