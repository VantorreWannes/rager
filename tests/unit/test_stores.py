"""Unit tests for stores."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import blake3
import pytest

from rager.stores import MemoryStore

if TYPE_CHECKING:
    from rager.types import Hash

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class ChunkMetadata:
    """Sample hashable value for exercising the store with structured values."""

    chunk: str
    file_id: Hash


def test_memory_store_add_and_get() -> None:
    """get() returns the value stored under the key."""
    # Arrange
    store: MemoryStore[int, str] = MemoryStore()

    # Act
    store.set(1, "a chunk")

    # Assert
    assert store.get(1) == "a chunk"


def test_memory_store_keys() -> None:
    """keys() returns the keys stored."""
    # Arrange
    store: MemoryStore[int, str] = MemoryStore()

    # Act
    store.set(1, "a chunk")
    store.set(2, "another chunk")

    # Assert
    assert store.keys() == [1, 2]


def test_memory_store_get_of_absent_key_returns_none() -> None:
    """get() returns None for a key that was never added."""
    # Arrange
    store: MemoryStore[int, str] = MemoryStore()

    # Act & Assert
    assert store.get(1) is None


def test_memory_store_add_overwrites_existing_key() -> None:
    """set() replaces the value stored under an existing key."""
    # Arrange
    store: MemoryStore[int, str] = MemoryStore()

    # Act
    store.set(1, "old")
    store.set(1, "new")

    # Assert
    assert store.get(1) == "new"


def test_memory_store_remove() -> None:
    """remove() deletes the value stored under the key."""
    # Arrange
    store: MemoryStore[int, str] = MemoryStore()
    store.set(1, "a chunk")

    # Act
    store.remove(1)

    # Assert
    assert store.get(1) is None


def test_memory_store_remove_of_absent_key_is_noop() -> None:
    """remove() of a key that was never added leaves the store unchanged."""
    # Arrange
    store: MemoryStore[int, str] = MemoryStore()
    store.set(1, "a chunk")

    # Act
    store.remove(2)

    # Assert
    assert store.get(1) == "a chunk"


def test_memory_store_with_embedding_values() -> None:
    """The store holds embedding-like tuple values."""
    # Arrange
    store: MemoryStore[int, tuple[float, ...]] = MemoryStore()

    # Act
    store.set(1, (0.1, 0.2))

    # Assert
    assert store.get(1) == (0.1, 0.2)


def test_memory_store_with_hash_keys_and_metadata_values() -> None:
    """The store is generic over key and value types."""
    # Arrange
    store: MemoryStore[Hash, ChunkMetadata] = MemoryStore()
    key = blake3.blake3(b"a file")
    metadata = ChunkMetadata(chunk="a chunk", file_id=key)

    # Act
    store.set(key, metadata)

    # Assert
    assert store.get(key) is metadata
