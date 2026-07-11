"""Key-value store protocol definitions."""

from typing import Protocol


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
