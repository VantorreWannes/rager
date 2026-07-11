"""Embedders for converting text chunks into vector representations."""

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from rager.types import Embedding


class Embedder(Protocol):
    """Protocol for embedders."""

    async def embed(self, chunk: str) -> Embedding:
        """Convert a text chunk into a vector representation."""
        ...
