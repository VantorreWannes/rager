"""Unit tests for indexes."""

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rager.indexes import DenseIndex, SparseIndex

pytestmark = pytest.mark.unit


@patch("rager.indexes.faiss")
def test_dense_index_init_defers_index_creation(faiss: MagicMock) -> None:
    """__init__() records the dimension but builds no FAISS index yet."""
    # Arrange
    dimensions = 3

    # Act
    index = DenseIndex(dimensions)

    # Assert
    faiss.IndexFlatIP.assert_not_called()
    faiss.IndexIDMap2.assert_not_called()
    assert index.dimensions == dimensions


@patch("rager.indexes.faiss")
@pytest.mark.asyncio
async def test_dense_index_infers_dimensions_on_first_add(faiss: MagicMock) -> None:
    """The first add() builds a FAISS index sized to the embedding width."""
    # Arrange
    dimensions = 3
    index = DenseIndex()

    # Act
    await index.add([0.1, 0.2, 0.3])

    # Assert
    faiss.IndexFlatIP.assert_called_once_with(dimensions)
    faiss.IndexIDMap2.assert_called_once_with(faiss.IndexFlatIP.return_value)
    assert index.dimensions == dimensions


@patch("rager.indexes.faiss")
@pytest.mark.asyncio
async def test_dense_index_rejects_mismatched_dimensions(faiss: MagicMock) -> None:
    """add() rejects an embedding whose width differs from the fixed dimension."""
    # Arrange
    index = DenseIndex(3)

    # Act & Assert
    with pytest.raises(ValueError, match="4 dimensions"):
        await index.add([0.1, 0.2, 0.3, 0.4])
    faiss.IndexIDMap2.assert_not_called()


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
async def test_dense_index_does_not_batch_across_instances(faiss: MagicMock) -> None:
    """Concurrent add() calls on different indexes stay in separate batches."""
    # Arrange
    first = DenseIndex(3)
    second = DenseIndex(3)
    faiss_index = faiss.IndexIDMap2.return_value

    # Act
    await asyncio.gather(first.add([1.0, 0.0, 0.0]), second.add([0.0, 1.0, 0.0]))

    # Assert
    expected_calls = 2
    assert faiss_index.add_with_ids.call_count == expected_calls


@patch("rager.indexes.faiss")
@pytest.mark.asyncio
async def test_dense_index_remove(faiss: MagicMock) -> None:
    """remove() removes the entry matching the given key."""
    # Arrange
    index = DenseIndex(3)
    await index.add([0.1, 0.2, 0.3])
    faiss_index = faiss.IndexIDMap2.return_value
    faiss_index.remove_ids.reset_mock()
    faiss.IDSelectorBatch.reset_mock()

    # Act
    await index.remove(42)

    # Assert
    selector_ids = faiss.IDSelectorBatch.call_args.args[0]
    np.testing.assert_array_equal(selector_ids, np.asarray([42], dtype=np.int64))
    faiss_index.remove_ids.assert_called_once_with(faiss.IDSelectorBatch.return_value)


@patch("rager.indexes.faiss")
@pytest.mark.asyncio
async def test_dense_index_remove_on_empty_index_is_noop(faiss: MagicMock) -> None:
    """remove() before any add touches no FAISS index."""
    # Arrange
    index = DenseIndex(3)

    # Act
    await index.remove(42)

    # Assert
    faiss.IndexIDMap2.assert_not_called()
    faiss.IDSelectorBatch.assert_not_called()


@patch("rager.indexes.faiss")
@pytest.mark.asyncio
async def test_dense_index_remove_batches_concurrent_calls(faiss: MagicMock) -> None:
    """Concurrent remove() calls coalesce into one FAISS call."""
    # Arrange
    index = DenseIndex(3)
    await index.add([0.1, 0.2, 0.3])
    faiss_index = faiss.IndexIDMap2.return_value
    faiss_index.remove_ids.reset_mock()
    faiss.IDSelectorBatch.reset_mock()

    # Act
    await asyncio.gather(index.remove(1), index.remove(2))

    # Assert
    selector_ids = faiss.IDSelectorBatch.call_args.args[0]
    np.testing.assert_array_equal(selector_ids, np.asarray([1, 2], dtype=np.int64))
    faiss_index.remove_ids.assert_called_once_with(faiss.IDSelectorBatch.return_value)


@patch("rager.indexes.faiss")
@pytest.mark.asyncio
async def test_dense_index_similar(faiss: MagicMock) -> None:
    """similar() searches with the requested count and drops -1 padding ids."""
    # Arrange
    expected_results = 4
    index = DenseIndex(3)
    await index.add([0.0, 0.0, 0.0])
    faiss_index = faiss.IndexIDMap2.return_value
    faiss_index.search.return_value = (
        np.asarray([[0.9, 0.5, -1.0, -1.0]], dtype=np.float32),
        np.asarray([[7, 3, -1, -1]], dtype=np.int64),
    )

    # Act
    result = await index.similar([0.1, 0.2, 0.3], results=expected_results)

    # Assert
    query, results = faiss_index.search.call_args.args
    np.testing.assert_array_equal(query, DenseIndex._to_rows([[0.1, 0.2, 0.3]]))
    assert results == expected_results
    assert result == [7, 3]


@patch("rager.indexes.faiss")
@pytest.mark.asyncio
async def test_dense_index_similar_on_empty_index(faiss: MagicMock) -> None:
    """similar() before any add returns no keys without searching."""
    # Arrange
    index = DenseIndex(3)

    # Act
    result = await index.similar([0.1, 0.2, 0.3])

    # Assert
    assert result == []
    faiss.IndexIDMap2.return_value.search.assert_not_called()


@patch("rager.indexes.faiss")
@pytest.mark.asyncio
async def test_dense_index_similar_scatters_concurrent_queries(
    faiss: MagicMock,
) -> None:
    """Concurrent similar() calls share one search, each trimmed to its own count."""
    # Arrange
    index = DenseIndex(3)
    await index.add([0.0, 0.0, 0.0])
    faiss_index = faiss.IndexIDMap2.return_value
    faiss_index.search.return_value = (
        np.asarray([[0.9, 0.5], [0.8, -1.0]], dtype=np.float32),
        np.asarray([[7, 3], [5, -1]], dtype=np.int64),
    )

    # Act
    first, second = await asyncio.gather(
        index.similar([1.0, 0.0, 0.0], results=2),
        index.similar([0.0, 1.0, 0.0], results=2),
    )

    # Assert
    faiss_index.search.assert_called_once()
    assert first == [7, 3]
    assert second == [5]


@patch("rager.indexes.faiss")
@pytest.mark.asyncio
async def test_dense_index_similar_trims_each_query_to_its_own_count(
    faiss: MagicMock,
) -> None:
    """A batched search uses the largest count, then trims per query."""
    # Arrange
    larger_count = 3
    index = DenseIndex(3)
    await index.add([0.0, 0.0, 0.0])
    faiss_index = faiss.IndexIDMap2.return_value
    faiss_index.search.return_value = (
        np.asarray([[0.9, 0.5, 0.4], [0.8, 0.3, 0.2]], dtype=np.float32),
        np.asarray([[7, 3, 1], [5, 8, 2]], dtype=np.int64),
    )

    # Act
    first, second = await asyncio.gather(
        index.similar([1.0, 0.0, 0.0], results=1),
        index.similar([0.0, 1.0, 0.0], results=larger_count),
    )

    # Assert
    _, results = faiss_index.search.call_args.args
    assert results == larger_count
    assert first == [7]
    assert second == [5, 8, 2]


def test_sparse_index_init() -> None:
    """__init__() starts with no stored embeddings."""
    # Act
    index = SparseIndex()

    # Assert
    assert index._embeddings == {}


def test_sparse_index_key_ignores_insertion_order() -> None:
    """_key() derives the same key regardless of dict insertion order."""
    # Act
    key = SparseIndex._key({1: 0.5, 9: 1.5})

    # Assert
    assert key == SparseIndex._key({9: 1.5, 1: 0.5})
    assert key != SparseIndex._key({1: 0.5, 9: 2.5})


def test_sparse_index_score() -> None:
    """_score() sums the weight products of shared tokens."""
    # Arrange
    query = {1: 2.0, 2: 3.0, 4: 1.0}
    stored = {1: 0.5, 2: 1.0}
    expected_score = 4.0

    # Act & Assert
    assert SparseIndex._score(query, stored) == expected_score
    assert SparseIndex._score(stored, query) == expected_score


@pytest.mark.asyncio
async def test_sparse_index_add_is_idempotent() -> None:
    """Adding the same embedding twice yields one key and one entry."""
    # Arrange
    index = SparseIndex()

    # Act
    first = await index.add({1: 1.0})
    second = await index.add({1: 1.0})

    # Assert
    assert first == second
    assert await index.similar({1: 1.0}) == [first]


@pytest.mark.asyncio
async def test_sparse_index_similar_ranks_nearest_first() -> None:
    """similar() returns keys ordered by inner-product similarity."""
    # Arrange
    index = SparseIndex()
    x_key = await index.add({1: 1.0})
    y_key = await index.add({1: 0.5, 2: 0.5})

    # Act & Assert
    assert await index.similar({1: 1.0}) == [x_key, y_key]


@pytest.mark.asyncio
async def test_sparse_index_similar_drops_non_matching_keys() -> None:
    """similar() omits stored embeddings sharing no tokens with the query."""
    # Arrange
    index = SparseIndex()
    x_key = await index.add({1: 1.0})
    await index.add({2: 1.0})

    # Act & Assert
    assert await index.similar({1: 1.0}) == [x_key]


@pytest.mark.asyncio
async def test_sparse_index_similar_caps_results() -> None:
    """similar() returns at most the requested number of results."""
    # Arrange
    index = SparseIndex()
    x_key = await index.add({1: 1.0})
    await index.add({1: 0.5})

    # Act & Assert
    assert await index.similar({1: 1.0}, results=1) == [x_key]


@pytest.mark.asyncio
async def test_sparse_index_similar_on_empty_index() -> None:
    """similar() on an empty index returns no keys."""
    # Arrange
    index = SparseIndex()

    # Act & Assert
    assert await index.similar({1: 1.0}) == []


@pytest.mark.asyncio
async def test_sparse_index_remove_drops_key_from_results() -> None:
    """Removed keys no longer appear in similarity results."""
    # Arrange
    index = SparseIndex()
    x_key = await index.add({1: 1.0})
    y_key = await index.add({1: 0.5})

    # Act
    await index.remove(x_key)

    # Assert
    assert await index.similar({1: 1.0}) == [y_key]


@pytest.mark.asyncio
async def test_sparse_index_remove_of_absent_key_is_noop() -> None:
    """Removing a key that was never added leaves the index unchanged."""
    # Arrange
    index = SparseIndex()
    x_key = await index.add({1: 1.0})

    # Act
    await index.remove(x_key + 1)

    # Assert
    assert await index.similar({1: 1.0}) == [x_key]


@pytest.mark.asyncio
async def test_sparse_index_batches_concurrent_calls() -> None:
    """Concurrent calls are batched, yet each caller gets its own result."""
    # Arrange
    index = SparseIndex()

    # Act
    x_key, y_key = await asyncio.gather(index.add({1: 1.0}), index.add({2: 1.0}))
    x_result, y_result = await asyncio.gather(
        index.similar({1: 1.0}), index.similar({2: 1.0})
    )

    # Assert
    assert x_result == [x_key]
    assert y_result == [y_key]


@pytest.mark.asyncio
async def test_sparse_index_does_not_batch_across_instances() -> None:
    """Concurrent calls on different indexes land in their own index."""
    # Arrange
    first = SparseIndex()
    second = SparseIndex()

    # Act
    x_key, y_key = await asyncio.gather(first.add({1: 1.0}), second.add({2: 1.0}))

    # Assert
    assert await first.similar({1: 1.0, 2: 1.0}) == [x_key]
    assert await second.similar({1: 1.0, 2: 1.0}) == [y_key]
