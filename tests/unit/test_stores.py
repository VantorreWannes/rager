"""Unit tests for stores."""

import pytest

from rager.stores import EmbeddingStore

pytestmark = pytest.mark.unit


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
