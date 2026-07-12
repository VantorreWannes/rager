"""Unit tests for indexes."""

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


def test_dense_index_to_row() -> None:
    """_to_row() converts an embedding into a single-row float32 matrix."""
    # Act
    row = DenseIndex._to_row([0.1, 0.2, 0.3])

    # Assert
    assert row.shape == (1, 3)
    assert row.dtype == np.float32


def test_dense_index_key_is_deterministic() -> None:
    """_key() derives the same int64 key for the same embedding content."""
    # Arrange
    row = DenseIndex._to_row([0.1, 0.2, 0.3])
    other = DenseIndex._to_row([0.3, 0.2, 0.1])

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
    expected_key = DenseIndex._key(DenseIndex._to_row(embedding))

    # Act
    key = await index.add(embedding)

    # Assert
    assert key == expected_key
    faiss.IDSelectorBatch.assert_called_once()
    faiss_index.remove_ids.assert_called_once_with(faiss.IDSelectorBatch.return_value)
    row, ids = faiss_index.add_with_ids.call_args.args
    np.testing.assert_array_equal(row, DenseIndex._to_row(embedding))
    np.testing.assert_array_equal(ids, np.asarray([expected_key], dtype=np.int64))


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
async def test_dense_index_similar(faiss: MagicMock) -> None:
    """similar() searches with top_k and drops FAISS's -1 padding ids."""
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
    np.testing.assert_array_equal(query, DenseIndex._to_row([0.1, 0.2, 0.3]))
    assert results == expected_results
    assert result == [7, 3]
    assert faiss.IDSelectorBatch.call_count == 0
