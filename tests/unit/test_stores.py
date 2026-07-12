"""Unit tests for stores."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import blake3
import pytest

from rager.stores import EmbeddingStore, MetadataStore

if TYPE_CHECKING:
    from rager.types import Hash

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class ChunkMetadata:
    """Sample metadata satisfying the Metadata protocol."""

    chunk: str
    file_id: Hash


def test_embedding_store_add_and_get() -> None:
    """get() returns the embedding stored under the key."""
    # Arrange
    store: EmbeddingStore[list[float]] = EmbeddingStore()

    # Act
    store.add(1, [0.1, 0.2])

    # Assert
    assert store.get(1) == [0.1, 0.2]


def test_embedding_store_get_of_absent_key_returns_none() -> None:
    """get() returns None for a key that was never added."""
    # Arrange
    store: EmbeddingStore[list[float]] = EmbeddingStore()

    # Act & Assert
    assert store.get(1) is None


def test_embedding_store_add_overwrites_existing_key() -> None:
    """add() replaces the embedding stored under an existing key."""
    # Arrange
    store: EmbeddingStore[list[float]] = EmbeddingStore()

    # Act
    store.add(1, [0.1])
    store.add(1, [0.2])

    # Assert
    assert store.get(1) == [0.2]


def test_embedding_store_remove() -> None:
    """remove() deletes the embedding stored under the key."""
    # Arrange
    store: EmbeddingStore[list[float]] = EmbeddingStore()
    store.add(1, [0.1])

    # Act
    store.remove(1)

    # Assert
    assert store.get(1) is None


def test_embedding_store_remove_of_absent_key_is_noop() -> None:
    """remove() of a key that was never added leaves the store unchanged."""
    # Arrange
    store: EmbeddingStore[list[float]] = EmbeddingStore()
    store.add(1, [0.1])

    # Act
    store.remove(2)

    # Assert
    assert store.get(1) == [0.1]


def test_metadata_store_add_and_get() -> None:
    """get() returns the metadata stored under the key."""
    # Arrange
    store: MetadataStore[ChunkMetadata] = MetadataStore()
    metadata = ChunkMetadata(chunk="a chunk", file_id=blake3.blake3(b"a file"))

    # Act
    store.add(1, metadata)

    # Assert
    assert store.get(1) is metadata


def test_metadata_store_get_of_absent_key_returns_none() -> None:
    """get() returns None for a key that was never added."""
    # Arrange
    store: MetadataStore[ChunkMetadata] = MetadataStore()

    # Act & Assert
    assert store.get(1) is None


def test_metadata_store_add_overwrites_existing_key() -> None:
    """add() replaces the metadata stored under an existing key."""
    # Arrange
    store: MetadataStore[ChunkMetadata] = MetadataStore()
    file_id = blake3.blake3(b"a file")

    # Act
    store.add(1, ChunkMetadata(chunk="old", file_id=file_id))
    store.add(1, ChunkMetadata(chunk="new", file_id=file_id))

    # Assert
    metadata = store.get(1)
    assert metadata is not None
    assert metadata.chunk == "new"


def test_metadata_store_remove() -> None:
    """remove() deletes the metadata stored under the key."""
    # Arrange
    store: MetadataStore[ChunkMetadata] = MetadataStore()
    store.add(1, ChunkMetadata(chunk="a chunk", file_id=blake3.blake3(b"a file")))

    # Act
    store.remove(1)

    # Assert
    assert store.get(1) is None


def test_metadata_store_remove_of_absent_key_is_noop() -> None:
    """remove() of a key that was never added leaves the store unchanged."""
    # Arrange
    store: MetadataStore[ChunkMetadata] = MetadataStore()
    metadata = ChunkMetadata(chunk="a chunk", file_id=blake3.blake3(b"a file"))
    store.add(1, metadata)

    # Act
    store.remove(2)

    # Assert
    assert store.get(1) is metadata
