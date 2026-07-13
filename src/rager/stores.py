"""Key-value store protocol definitions."""

import logging
from typing import Protocol

from rager.types import DenseEmbedding, Metadata, SparseEmbedding

logger = logging.getLogger(__name__)


class Store[K, V](Protocol):
    """Protocol for a key-value store mapping keys of type K to values of type V."""

    def add(self, key: K, value: V) -> None:
        """Store a value with the given key."""
        ...

    def get(self, key: K) -> V | None:
        """Retrieve a value by its key."""
        ...

    def remove(self, key: K) -> None:
        """Remove a value by its key."""
        ...


class EmbeddingStore[E]:
    """In-memory store mapping index keys to embeddings."""

    def __init__(self) -> None:
        """Initialize the store with no embeddings."""
        self._embeddings: dict[int, E] = {}

    def add(self, key: int, value: E) -> None:
        """Store an embedding with the given key."""
        logger.debug("Storing embedding for key %d", key)
        self._embeddings[key] = value

    def get(self, key: int) -> E | None:
        """Retrieve an embedding by its key."""
        embedding = self._embeddings.get(key)
        if embedding is None:
            logger.debug("No embedding stored for key %d", key)
        return embedding

    def remove(self, key: int) -> None:
        """Remove an embedding by its key."""
        logger.debug("Removing embedding for key %d", key)
        self._embeddings.pop(key, None)


DenseEmbeddingStore = EmbeddingStore[DenseEmbedding]
SparseEmbeddingStore = EmbeddingStore[SparseEmbedding]


class ChunkStore:
    """In-memory store mapping chunk indexes to chunk text."""

    def __init__(self) -> None:
        """Initialize the store with no chunks."""
        self._chunks: dict[int, str] = {}

    def add(self, key: int, value: str) -> None:
        """Store chunk text with the given key."""
        logger.debug("Storing chunk of %d characters for key %d", len(value), key)
        self._chunks[key] = value

    def get(self, key: int) -> str | None:
        """Retrieve chunk text by its key."""
        chunk = self._chunks.get(key)
        if chunk is None:
            logger.debug("No chunk stored for key %d", key)
        return chunk

    def remove(self, key: int) -> None:
        """Remove chunk text by its key."""
        logger.debug("Removing chunk for key %d", key)
        self._chunks.pop(key, None)


class MetadataStore[M: Metadata]:
    """In-memory store mapping embedding keys to chunk metadata."""

    def __init__(self) -> None:
        """Initialize the store with no metadata."""
        self._metadata: dict[int, M] = {}

    def add(self, key: int, value: M) -> None:
        """Store metadata with the given key."""
        logger.debug("Storing metadata for key %d", key)
        self._metadata[key] = value

    def get(self, key: int) -> M | None:
        """Retrieve metadata by its key."""
        metadata = self._metadata.get(key)
        if metadata is None:
            logger.debug("No metadata stored for key %d", key)
        return metadata

    def remove(self, key: int) -> None:
        """Remove metadata by its key."""
        logger.debug("Removing metadata for key %d", key)
        self._metadata.pop(key, None)
