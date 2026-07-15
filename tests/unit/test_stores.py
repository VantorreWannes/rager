"""Unit tests for stores."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import blake3
import pytest

from rager.stores import BaseStore, FileStore, MemoryStore

if TYPE_CHECKING:
    from pathlib import Path

    from rager.types import Hash

pytestmark = pytest.mark.unit


@pytest.fixture
def file_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> FileStore[int, bytes]:
    """Return a FileStore whose jar directory lives under a temporary path."""
    monkeypatch.chdir(tmp_path)
    return FileStore()


@dataclass(frozen=True)
class ChunkMetadata:
    """Sample hashable value for exercising the store with structured values."""

    chunk: str
    file_id: Hash


def test_base_store_is_abstract() -> None:
    """BaseStore cannot be instantiated without the four core operations."""
    # Act & Assert
    with pytest.raises(TypeError):
        BaseStore()  # type: ignore[abstract]


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


def test_memory_store_inherits_container_helpers() -> None:
    """The helpers derived from the four core operations behave as expected."""
    # Arrange
    store: MemoryStore[int, str] = MemoryStore()
    absent_key = 3
    expected_len = 2

    # Act & Assert
    assert not store
    store.set(1, "a chunk")
    store.set(2, "another chunk")
    assert store
    assert 1 in store
    assert absent_key not in store
    assert len(store) == expected_len
    assert sorted(store) == [1, 2]


def test_memory_store_clear_removes_every_key() -> None:
    """clear() empties the store."""
    # Arrange
    store: MemoryStore[int, str] = MemoryStore()
    store.set(1, "a chunk")
    store.set(2, "another chunk")

    # Act
    store.clear()

    # Assert
    assert store.keys() == []


def test_file_store_set_and_get(file_store: FileStore[int, bytes]) -> None:
    """get() returns the value sealed under the key."""
    # Act
    file_store.set(1, b"a chunk")

    # Assert
    assert file_store.get(1) == b"a chunk"


def test_file_store_seals_values_on_disk(
    file_store: FileStore[int, bytes], tmp_path: Path
) -> None:
    """set() seals the value in the jar directory."""
    # Act
    file_store.set(1, b"a chunk")

    # Assert
    assert list((tmp_path / ".jar" / "stores").iterdir())


def test_file_store_keys(file_store: FileStore[int, bytes]) -> None:
    """keys() returns the keys stored."""
    # Act
    file_store.set(1, b"a chunk")
    file_store.set(2, b"another chunk")

    # Assert
    assert file_store.keys() == [1, 2]


def test_file_store_get_of_absent_key_returns_none(
    file_store: FileStore[int, bytes],
) -> None:
    """get() returns None for a key that was never added."""
    # Act & Assert
    assert file_store.get(1) is None


def test_file_store_get_of_unsealed_value_returns_none(
    file_store: FileStore[int, bytes], tmp_path: Path
) -> None:
    """get() returns None when the sealed value is gone from the jar."""
    # Arrange
    file_store.set(1, b"a chunk")
    for file in (tmp_path / ".jar" / "stores").iterdir():
        file.unlink()

    # Act & Assert
    assert file_store.get(1) is None


def test_file_store_set_overwrites_existing_key(
    file_store: FileStore[int, bytes],
) -> None:
    """set() replaces the value sealed under an existing key."""
    # Act
    file_store.set(1, b"old")
    file_store.set(1, b"new")

    # Assert
    assert file_store.get(1) == b"new"


def test_file_store_get_of_empty_value_returns_it(
    file_store: FileStore[int, bytes],
) -> None:
    """get() returns a sealed empty value instead of treating it as a miss."""
    # Act
    file_store.set(1, b"")

    # Assert
    assert file_store.get(1) == b""


def test_file_store_with_embedding_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The store seals non-buffer values such as embedding-like tuples."""
    # Arrange
    monkeypatch.chdir(tmp_path)
    store: FileStore[int, tuple[float, ...]] = FileStore()

    # Act
    store.set(1, (0.1, 0.2))

    # Assert
    assert store.get(1) == (0.1, 0.2)


def test_file_store_same_value_under_different_keys(
    file_store: FileStore[int, bytes],
) -> None:
    """Identical values sealed under different keys resolve independently."""
    # Act
    file_store.set(1, b"a chunk")
    file_store.set(2, b"a chunk")
    file_store.remove(1)

    # Assert
    assert file_store.get(1) is None
    assert file_store.get(2) == b"a chunk"


def test_file_store_remove(file_store: FileStore[int, bytes]) -> None:
    """remove() deletes the value stored under the key."""
    # Arrange
    file_store.set(1, b"a chunk")

    # Act
    file_store.remove(1)

    # Assert
    assert file_store.get(1) is None
    assert file_store.keys() == []


def test_file_store_remove_of_absent_key_is_noop(
    file_store: FileStore[int, bytes],
) -> None:
    """remove() of a key that was never added leaves the store unchanged."""
    # Arrange
    file_store.set(1, b"a chunk")

    # Act
    file_store.remove(2)

    # Assert
    assert file_store.get(1) == b"a chunk"
