"""Embedders for converting text chunks into vector representations."""

from typing import Protocol


class Embedder[E](Protocol):
    """Protocol for embedders."""

    async def embed(self, chunk: str) -> E:
        """Convert a text chunk into a vector representation."""
        ...
