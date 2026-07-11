"""Stores for persisting embeddings."""

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from rager.types import Embedding


class EmbeddingStore(Protocol):
    """Protocol for embedding stores."""

    async def add(self, embedding: Embedding) -> None:
        """Persist an embedding."""
        ...

    async def similar(
        self, embedding: Embedding, max_embeddings: int
    ) -> list[Embedding]:
        """Retrieve the most similar embeddings up to max_embeddings."""
        ...
