"""Key-value store protocol definitions."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Generator, Hashable
from pathlib import Path
from typing import Protocol, cast, override

import dill
from belljar import Jar
from blake3 import blake3

logger = logging.getLogger(__name__)


class Store[K: Hashable, V](Protocol):
    """Protocol for a key-value store mapping keys of type K to values of type V."""

    def keys(self) -> list[K]:
        """Return the keys in the store."""
        ...

    def get(self, key: K) -> V | None:
        """Retrieve a value by its key, or None if the key is not present."""
        ...

    def set(self, key: K, value: V) -> None:
        """Store a value with the given key."""
        ...

    def remove(self, key: K) -> None:
        """Remove a value by its key."""
        ...

    def clear(self) -> None:
        """Remove all values from the store."""
        ...

    def __setitem__(self, key: K, value: V) -> None:
        """Store a value with the given key."""
        ...

    def __getitem__(self, key: K) -> V:
        """Retrieve a value by its key."""
        ...

    def __delitem__(self, key: K) -> None:
        """Remove a value by its key."""
        ...

    def __contains__(self, key: K) -> bool:
        """Check if a key is in the store."""
        ...

    def __len__(self) -> int:
        """Return the number of keys in the store."""
        ...

    def __iter__(self) -> Generator[K]:
        """Iterate over the keys in the store."""
        ...

    def __bool__(self) -> bool:
        """Check if the store is non-empty."""
        ...


class BaseStore[K: Hashable, V](ABC):
    """Abstract helper base deriving the ``Store`` protocol from four operations.

    Subclasses implement ``keys``, ``__setitem__``, ``__getitem__``, and
    ``__delitem__``; the helpers provide the rest of the protocol.
    """

    @abstractmethod
    def keys(self) -> list[K]:
        """Return the keys in the store."""

    @abstractmethod
    def __setitem__(self, key: K, value: V) -> None:
        """Store a value with the given key."""

    @abstractmethod
    def __getitem__(self, key: K) -> V:
        """Retrieve a value by its key."""

    @abstractmethod
    def __delitem__(self, key: K) -> None:
        """Remove a value by its key."""

    def get(self, key: K) -> V | None:
        """Retrieve a value by its key, or None if the key is not present."""
        if key in self:
            return self[key]
        return None

    def set(self, key: K, value: V) -> None:
        """Store a value with the given key."""
        self[key] = value

    def remove(self, key: K) -> None:
        """Remove a value by its key."""
        if key in self:
            del self[key]

    def clear(self) -> None:
        """Remove all values from the store."""
        for key in self.keys():
            del self[key]

    def __contains__(self, key: K) -> bool:
        """Check if a key is in the store."""
        return key in self.keys()

    def __len__(self) -> int:
        """Return the number of keys in the store."""
        return len(self.keys())

    def __iter__(self) -> Generator[K]:
        """Iterate over the keys in the store."""
        yield from self.keys()

    def __bool__(self) -> bool:
        """Check if the store is non-empty."""
        return len(self) > 0


class MemoryStore[K: Hashable, V](BaseStore[K, V]):
    """In-memory store mapping index keys to values."""

    def __init__(self) -> None:
        """Initialize the store with no values."""
        self._map: dict[K, V] = {}

    @override
    def keys(self) -> list[K]:
        """Return the keys in the store."""
        return list(self._map.keys())

    @override
    def __setitem__(self, key: K, value: V) -> None:
        """Store a value with the given key."""
        logger.debug("Storing value for key %r", key)
        self._map[key] = value

    @override
    def __getitem__(self, key: K) -> V:
        """Retrieve a value by its key."""
        logger.debug("Retrieving value for key %r", key)
        return self._map[key]

    @override
    def __delitem__(self, key: K) -> None:
        """Remove a value by its key."""
        logger.debug("Removing value for key %r", key)
        del self._map[key]


class FileStore[K: Hashable, V](BaseStore[K, V]):
    """File-based store mapping index keys to values."""

    def __init__(self) -> None:
        """Initialize FileStore."""
        self._key_map: dict[K, str] = {}

    def _jar(self, key: K, digest: str) -> Jar[V]:
        """Open a jar positioned at the identity of the given key and digest."""
        jar = Jar[V](Path(".jar/stores"))
        jar.include(self.keys.__code__)
        jar.include(self.__setitem__.__code__)
        jar.include(self.__getitem__.__code__)
        jar.include(self.__delitem__.__code__)
        jar.include(self._jar.__code__)
        jar.include(key)
        jar.include(digest)
        return jar

    def _digest(self, key: K) -> str:
        """Return the digest of the given key."""
        return blake3(dill.dumps(key)).hexdigest()

    @override
    def keys(self) -> list[K]:
        """Return the keys in the store."""
        return list(self._key_map.keys())

    @override
    def __setitem__(self, key: K, value: V) -> None:
        """Store a value with the given key."""
        logger.debug("Storing value for key %r", key)
        digest = self._digest(key)
        self._jar(key, digest).set(value)
        self._key_map[key] = digest

    @override
    def __getitem__(self, key: K) -> V:
        """Retrieve a value by its key."""
        logger.debug("Retrieving value for key %r", key)
        digest = self._key_map[key]
        return cast("V", self._jar(key, digest).get())

    @override
    def __delitem__(self, key: K) -> None:
        """Delete a value by its key."""
        logger.debug("Deleting value for key %r", key)
        del self._key_map[key]
