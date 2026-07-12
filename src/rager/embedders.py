"""Embedders for converting text chunks into vector representations."""

from datetime import timedelta
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import belljar
import concresce
from sentence_transformers import SentenceTransformer, SparseEncoder

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from torch import Tensor

    from rager import DenseEmbedding
    from rager.types import SparseEmbedding


class Embedder[E](Protocol):
    """Protocol for embedders."""

    async def embed(self, chunk: str) -> E:
        """Convert a text chunk into a vector representation."""
        ...


class SentenceTransformerDenseEmbedder:
    """Dense embedding using SentenceTransformer."""

    def __init__(self, model_name: str) -> None:
        """Initialize the dense embedder with a specific model."""
        self.model_name = model_name

    @cached_property
    def model(self) -> SentenceTransformer:
        """Load the SentenceTransformer model."""
        return SentenceTransformer(self.model_name)

    @concresce.batch(window=timedelta(milliseconds=100))
    async def _encode(self, chunk: str) -> DenseEmbedding:
        """Convert a text chunk into a vector representation."""
        chunks = await concresce.collect(chunk)
        embeddings = self.model.encode(chunks, normalize_embeddings=True).tolist()
        return concresce.scatter(embeddings)

    @belljar.store(Path(".jar/embedders"))
    async def embed(self, chunk: str) -> DenseEmbedding:
        """Convert a text chunk into a vector representation."""
        belljar.include(self.model_name)
        belljar.include(chunk)
        belljar.check()
        return await self._encode(chunk)


class SpladeSparseEmbedder:
    """Sparse embedding using a SPLADE SparseEncoder."""

    def __init__(self, model_name: str) -> None:
        """Initialize the sparse embedder with a specific model."""
        self.model_name = model_name

    @cached_property
    def model(self) -> SparseEncoder:
        """Load the SPLADE SparseEncoder model."""
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

    @concresce.batch(window=timedelta(milliseconds=100))
    async def _encode(self, chunk: str) -> SparseEmbedding:
        """Convert a text chunk into a sparse vector representation."""
        chunks = await concresce.collect(chunk)
        coalesced = self.model.encode(chunks).coalesce()
        embeddings = self._coalesced_to_embeddings(coalesced, len(chunks))
        return concresce.scatter(embeddings)

    @belljar.store(Path(".jar/embedders"))
    async def embed(self, chunk: str) -> SparseEmbedding:
        """Convert a text chunk into a sparse vector representation."""
        belljar.include(self.model_name)
        belljar.include(chunk)
        belljar.check()
        return await self._encode(chunk)
