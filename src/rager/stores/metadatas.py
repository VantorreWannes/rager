"""Stores for persisting metadata."""

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from rager.types import Embedding, Metadata


class MetadataStore(Protocol):
    """Protocol for metadata stores."""

    async def add(self, metadata: Metadata) -> None:
        """Persist a piece of metadata."""
        ...

    async def similar(self, embedding: Embedding, max_embedding: int) -> list[Metadata]:
        """Retrieve the metadata most similar to embedding, up to max_embedding."""
        ...
