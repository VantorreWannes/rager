"""Unit tests for stores."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import blake3
import pytest

from rager.stores import CachedMemoryStore, MemoryStore

if TYPE_CHECKING:
    from pathlib import Path

    from rager.types import Hash

pytestmark = pytest.mark.unit


@pytest.fixture
def jar_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> CachedMemoryStore[int, bytes]:
    """Return a CachedMemoryStore whose jar directory lives under a temporary path."""
    monkeypatch.chdir(tmp_path)
    return CachedMemoryStore()


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


def test_jar_store_set_and_get(jar_store: CachedMemoryStore[int, bytes]) -> None:
    """get() returns the value sealed under the key."""
    # Act
    jar_store.set(1, b"a chunk")

    # Assert
    assert jar_store.get(1) == b"a chunk"


def test_jar_store_seals_values_on_disk(
    jar_store: CachedMemoryStore[int, bytes], tmp_path: Path
) -> None:
    """set() seals the value in the jar directory."""
    # Act
    jar_store.set(1, b"a chunk")

    # Assert
    assert list((tmp_path / ".jar" / "stores").iterdir())


def test_jar_store_keys(jar_store: CachedMemoryStore[int, bytes]) -> None:
    """keys() returns the keys stored."""
    # Act
    jar_store.set(1, b"a chunk")
    jar_store.set(2, b"another chunk")

    # Assert
    assert jar_store.keys() == [1, 2]


def test_jar_store_get_of_absent_key_returns_none(
    jar_store: CachedMemoryStore[int, bytes],
) -> None:
    """get() returns None for a key that was never added."""
    # Act & Assert
    assert jar_store.get(1) is None


def test_jar_store_get_of_unsealed_value_returns_none(
    jar_store: CachedMemoryStore[int, bytes], tmp_path: Path
) -> None:
    """get() returns None when the sealed value is gone from the jar."""
    # Arrange
    jar_store.set(1, b"a chunk")
    for file in (tmp_path / ".jar" / "stores").iterdir():
        file.unlink()

    # Act & Assert
    assert jar_store.get(1) is None


def test_jar_store_set_overwrites_existing_key(
    jar_store: CachedMemoryStore[int, bytes],
) -> None:
    """set() replaces the value sealed under an existing key."""
    # Act
    jar_store.set(1, b"old")
    jar_store.set(1, b"new")

    # Assert
    assert jar_store.get(1) == b"new"


def test_jar_store_get_of_empty_value_returns_it(
    jar_store: CachedMemoryStore[int, bytes],
) -> None:
    """get() returns a sealed empty buffer instead of treating it as a miss."""
    # Act
    jar_store.set(1, b"")

    # Assert
    assert jar_store.get(1) == b""


def test_jar_store_same_value_under_different_keys(
    jar_store: CachedMemoryStore[int, bytes],
) -> None:
    """Identical values sealed under different keys resolve independently."""
    # Act
    jar_store.set(1, b"a chunk")
    jar_store.set(2, b"a chunk")
    jar_store.remove(1)

    # Assert
    assert jar_store.get(1) is None
    assert jar_store.get(2) == b"a chunk"


def test_jar_store_remove(jar_store: CachedMemoryStore[int, bytes]) -> None:
    """remove() deletes the value stored under the key."""
    # Arrange
    jar_store.set(1, b"a chunk")

    # Act
    jar_store.remove(1)

    # Assert
    assert jar_store.get(1) is None
    assert jar_store.keys() == []


def test_jar_store_remove_of_absent_key_is_noop(
    jar_store: CachedMemoryStore[int, bytes],
) -> None:
    """remove() of a key that was never added leaves the store unchanged."""
    # Arrange
    jar_store.set(1, b"a chunk")

    # Act
    jar_store.remove(2)

    # Assert
    assert jar_store.get(1) == b"a chunk"
