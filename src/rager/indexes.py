"""Embedding Index protocol definitions."""

from typing import Protocol


class Index[E, K](Protocol):
    """Protocol for an embedding index."""

    async def add(self, embedding: E) -> K:
        """Add an embedding to the index and return its key."""
        ...

    async def remove(self, key: K) -> None:
        """Remove an embedding from the index by its key."""
        ...

    async def similar(self, embedding: E) -> list[E]:
        """Retrieve the most similar embeddings to the given embedding."""
        ...
