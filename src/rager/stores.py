"""Key-value store protocol definitions."""

from typing import Protocol


class Store[V, K](Protocol):
    """Protocol for a key-value store mapping keys of type K to values of type V."""

    async def add(self, key: K, value: V) -> None:
        """Store a single value and return its key."""
        ...

    async def get(self, key: K) -> V | None:
        """Retrieve a value by its key."""
        ...
