"""Unit tests for indexes."""

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rager.indexes import DenseIndex

pytestmark = pytest.mark.unit


@patch("rager.indexes.faiss")
def test_dense_index_init(faiss: MagicMock) -> None:
    """__init__() wraps a flat inner-product index in an id map."""
    # Arrange
    dimensions = 3
    results = 5

    # Act
    index = DenseIndex(dimensions, results=results)

    # Assert
    faiss.IndexFlatIP.assert_called_once_with(dimensions)
    faiss.IndexIDMap2.assert_called_once_with(faiss.IndexFlatIP.return_value)
    assert index.dimensions == dimensions
    assert index.results == results


def test_dense_index_to_rows() -> None:
    """_to_rows() converts embeddings into a float32 matrix."""
    # Act
    rows = DenseIndex._to_rows([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    # Assert
    assert rows.shape == (2, 3)
    assert rows.dtype == np.float32


def test_dense_index_key_is_deterministic() -> None:
    """_key() derives the same int64 key for the same embedding content."""
    # Arrange
    row, other = DenseIndex._to_rows([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])

    # Act
    key = DenseIndex._key(row)

    # Assert
    assert key == DenseIndex._key(row)
    assert key != DenseIndex._key(other)
    assert np.asarray([key], dtype=np.int64)[0] == key


@patch("rager.indexes.faiss")
@pytest.mark.asyncio
async def test_dense_index_add(faiss: MagicMock) -> None:
    """add() removes any previous entry, adds the row, and returns its key."""
    # Arrange
    index = DenseIndex(3)
    faiss_index = faiss.IndexIDMap2.return_value
    embedding = [0.1, 0.2, 0.3]
    expected_key = DenseIndex._key(DenseIndex._to_rows([embedding])[0])

    # Act
    key = await index.add(embedding)

    # Assert
    assert key == expected_key
    faiss.IDSelectorBatch.assert_called_once()
    faiss_index.remove_ids.assert_called_once_with(faiss.IDSelectorBatch.return_value)
    rows, ids = faiss_index.add_with_ids.call_args.args
    np.testing.assert_array_equal(rows, DenseIndex._to_rows([embedding]))
    np.testing.assert_array_equal(ids, np.asarray([expected_key], dtype=np.int64))


@patch("rager.indexes.faiss")
@pytest.mark.asyncio
async def test_dense_index_add_batches_concurrent_calls(faiss: MagicMock) -> None:
    """Concurrent add() calls coalesce into one deduplicated FAISS call."""
    # Arrange
    index = DenseIndex(3)
    faiss_index = faiss.IndexIDMap2.return_value
    first = [1.0, 0.0, 0.0]
    second = [0.0, 1.0, 0.0]

    # Act
    keys = await asyncio.gather(index.add(first), index.add(second), index.add(first))

    # Assert
    assert keys[0] == keys[2]
    assert keys[0] != keys[1]
    faiss_index.add_with_ids.assert_called_once()
    rows, ids = faiss_index.add_with_ids.call_args.args
    np.testing.assert_array_equal(rows, DenseIndex._to_rows([first, second]))
    np.testing.assert_array_equal(ids, np.asarray(keys[:2], dtype=np.int64))


@patch("rager.indexes.faiss")
@pytest.mark.asyncio
async def test_dense_index_remove(faiss: MagicMock) -> None:
    """remove() removes the entry matching the given key."""
    # Arrange
    index = DenseIndex(3)
    faiss_index = faiss.IndexIDMap2.return_value

    # Act
    await index.remove(42)

    # Assert
    selector_ids = faiss.IDSelectorBatch.call_args.args[0]
    np.testing.assert_array_equal(selector_ids, np.asarray([42], dtype=np.int64))
    faiss_index.remove_ids.assert_called_once_with(faiss.IDSelectorBatch.return_value)


@patch("rager.indexes.faiss")
@pytest.mark.asyncio
async def test_dense_index_remove_batches_concurrent_calls(faiss: MagicMock) -> None:
    """Concurrent remove() calls coalesce into one FAISS call."""
    # Arrange
    index = DenseIndex(3)
    faiss_index = faiss.IndexIDMap2.return_value

    # Act
    await asyncio.gather(index.remove(1), index.remove(2))

    # Assert
    selector_ids = faiss.IDSelectorBatch.call_args.args[0]
    np.testing.assert_array_equal(selector_ids, np.asarray([1, 2], dtype=np.int64))
    faiss_index.remove_ids.assert_called_once_with(faiss.IDSelectorBatch.return_value)


@patch("rager.indexes.faiss")
@pytest.mark.asyncio
async def test_dense_index_similar(faiss: MagicMock) -> None:
    """similar() searches with the results count and drops -1 padding ids."""
    # Arrange
    expected_results = 4
    index = DenseIndex(3, results=expected_results)
    faiss_index = faiss.IndexIDMap2.return_value
    faiss_index.search.return_value = (
        np.asarray([[0.9, 0.5, -1.0, -1.0]], dtype=np.float32),
        np.asarray([[7, 3, -1, -1]], dtype=np.int64),
    )

    # Act
    result = await index.similar([0.1, 0.2, 0.3])

    # Assert
    query, results = faiss_index.search.call_args.args
    np.testing.assert_array_equal(query, DenseIndex._to_rows([[0.1, 0.2, 0.3]]))
    assert results == expected_results
    assert result == [7, 3]
    assert faiss.IDSelectorBatch.call_count == 0


@patch("rager.indexes.faiss")
@pytest.mark.asyncio
async def test_dense_index_similar_scatters_concurrent_queries(
    faiss: MagicMock,
) -> None:
    """Concurrent similar() calls share one search, each getting its own row."""
    # Arrange
    index = DenseIndex(3, results=2)
    faiss_index = faiss.IndexIDMap2.return_value
    faiss_index.search.return_value = (
        np.asarray([[0.9, 0.5], [0.8, -1.0]], dtype=np.float32),
        np.asarray([[7, 3], [5, -1]], dtype=np.int64),
    )

    # Act
    first, second = await asyncio.gather(
        index.similar([1.0, 0.0, 0.0]), index.similar([0.0, 1.0, 0.0])
    )

    # Assert
    faiss_index.search.assert_called_once()
    assert first == [7, 3]
    assert second == [5]
