"""Key-value store protocol definitions."""

from typing import Protocol

from rager.types import DenseEmbedding, Metadata, SparseEmbedding


class Store[V, K](Protocol):
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
        self._embeddings[key] = value

    def get(self, key: int) -> E | None:
        """Retrieve an embedding by its key."""
        return self._embeddings.get(key)

    def remove(self, key: int) -> None:
        """Remove an embedding by its key."""
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
        self._chunks[key] = value

    def get(self, key: int) -> str | None:
        """Retrieve chunk text by its key."""
        return self._chunks.get(key)

    def remove(self, key: int) -> None:
        """Remove chunk text by its key."""
        self._chunks.pop(key, None)


class MetadataStore[M: Metadata]:
    """In-memory store mapping embedding keys to chunk metadata."""

    def __init__(self) -> None:
        """Initialize the store with no metadata."""
        self._metadata: dict[int, M] = {}

    def add(self, key: int, value: M) -> None:
        """Store metadata with the given key."""
        self._metadata[key] = value

    def get(self, key: int) -> M | None:
        """Retrieve metadata by its key."""
        return self._metadata.get(key)

    def remove(self, key: int) -> None:
        """Remove metadata by its key."""
        self._metadata.pop(key, None)
