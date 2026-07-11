"""Key-value store protocol definitions."""

from typing import Protocol


class Store[V, K](Protocol):
    """Protocol for a key-value store mapping keys of type K to values of type V."""

    def add(self, value: V) -> K:
        """Store a single value and return its key."""
        ...

    def get(self, key: K) -> V | None:
        """Retrieve a value by its key."""
        ...
