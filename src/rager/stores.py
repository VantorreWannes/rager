"""Key-value store protocol definitions."""

import logging
from collections.abc import Buffer, Hashable
from pathlib import Path
from typing import Protocol

from belljar import Jar
from blake3 import blake3

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


class JarStore[K: Hashable, V: Buffer]:
    """JAR-based key-value store mapping index keys to values."""

    def __init__(self) -> None:
        """Initialize the store with no values."""
        self._map: dict[K, str] = {}

    def _digest(self, value: V) -> str:
        """Compute the digest for the given value."""
        return blake3(value).hexdigest()

    def _jar(self, key: K, digest: str) -> Jar[V]:
        """Open a jar positioned at the identity of the given key and digest."""
        jar = Jar[V](Path(".jar/stores"))
        jar.include(self.set.__code__)
        jar.include(self.get.__code__)
        jar.include(key)
        jar.include(digest)
        return jar

    def set(self, key: K, value: V) -> None:
        """Store a value with the given key."""
        logger.debug("Storing value for key %r", key)
        digest = self._digest(value)
        self._jar(key, digest).set(value)
        self._map[key] = digest

    def get(self, key: K) -> V | None:
        """Retrieve a value by its key."""
        digest = self._map.get(key)
        if digest is None:
            logger.debug("No value stored for key %r", key)
            return None
        value = self._jar(key, digest).get()
        if value is None:
            logger.debug("No value found in JAR for key %r", key)
        return value

    def remove(self, key: K) -> None:
        """Remove a value by its key."""
        logger.debug("Removing value for key %r", key)
        self._map.pop(key, None)

    def keys(self) -> list[K]:
        """Return all keys in the store."""
        return list(self._map.keys())
