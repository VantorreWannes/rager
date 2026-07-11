"""Embedders for converting text chunks into vector representations."""

from datetime import timedelta
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import belljar
import concresce
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from rager import Embedding


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
    async def _encode(self, chunk: str) -> Embedding:
        """Convert a text chunk into a vector representation."""
        chunks = await concresce.collect(chunk)
        return self.model.encode(chunks, normalize_embeddings=True).tolist()

    @belljar.store(Path(".jar/embedders"))
    async def embed(self, chunk: str) -> Embedding:
        """Convert a text chunk into a vector representation."""
        belljar.include(self.model_name)
        belljar.include(chunk)
        belljar.check()
        return await self._encode(chunk)
