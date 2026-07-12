"""Key-value store protocol definitions."""

from typing import Protocol

from rager.types import DenseEmbedding, SparseEmbedding


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
