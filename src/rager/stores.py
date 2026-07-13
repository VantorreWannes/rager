"""Key-value store protocol definitions."""

import logging
from collections.abc import Hashable
from typing import Protocol

logger = logging.getLogger(__name__)


class Store[K: Hashable, V: Hashable](Protocol):
    """Protocol for a key-value store mapping keys of type K to values of type V."""

    def set(self, key: K, value: V) -> None:
        """Store a value with the given key."""
        ...

    def get(self, key: K) -> V | None:
        """Retrieve a value by its key."""
        ...

    def remove(self, key: K) -> None:
        """Remove a value by its key."""
        ...

    def keys(self) -> list[K]:
        """Return all keys in the store."""
        ...


class MemoryStore[K: Hashable, V: Hashable]:
    """In-memory store mapping index keys to values."""

    def __init__(self) -> None:
        """Initialize the store with no values."""
        self._map: dict[K, V] = {}

    def set(self, key: K, value: V) -> None:
        """Store a value with the given key."""
        logger.debug("Storing value for key %r", key)
        self._map[key] = value

    def get(self, key: K) -> V | None:
        """Retrieve an embedding by its key."""
        value = self._map.get(key)
        if value is None:
            logger.debug("No value stored for key %r", key)
        return value

    def remove(self, key: K) -> None:
        """Remove a value by its key."""
        logger.debug("Removing value for key %r", key)
        self._map.pop(key, None)

    def keys(self) -> list[K]:
        """Return all keys in the store."""
        return list(self._map.keys())
