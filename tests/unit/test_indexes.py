"""Unit tests for indexes."""

import asyncio
from typing import TYPE_CHECKING, override
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rager.indexes import FaissIndex, SparseIndex, _scores, _top_keys
from rager.stores import FileStore, MemoryStore

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


@pytest.fixture
def dense_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> FaissIndex[str, list[float]]:
    """Return a FaissIndex whose jar directory lives under a temporary path."""
    monkeypatch.chdir(tmp_path)
    return FaissIndex(3, MemoryStore())


@pytest.fixture
def sparse_index() -> SparseIndex[str]:
    """Return a SparseIndex backed by in-memory stores."""
    return SparseIndex(MemoryStore(), MemoryStore())


# --- FaissIndex, real FAISS -------------------------------------------------


def test_faiss_index_id_is_deterministic() -> None:
    """_id() derives the same int64 id for the same key content."""
    # Act
    key = FaissIndex._id("x")

    # Assert
    assert key == FaissIndex._id("x")
    assert key != FaissIndex._id("y")
    assert np.asarray([key], dtype=np.int64)[0] == key


@pytest.mark.asyncio
async def test_faiss_index_rejects_mismatched_dimensions(
    dense_index: FaissIndex[str, list[float]],
) -> None:
    """Storing an embedding whose width differs from the fixed dimension fails."""
    # Act & Assert
    with pytest.raises(ValueError, match="4 dimensions"):
        await dense_index.set("x", [0.1, 0.2, 0.3, 0.4])


@pytest.mark.asyncio
async def test_faiss_index_get_returns_stored_embedding(
    dense_index: FaissIndex[str, list[float]],
) -> None:
    """get() returns the embedding sealed under the key."""
    # Act
    await dense_index.set("x", [1.0, 0.0, 0.0])

    # Assert
    assert await dense_index.get("x") == [1.0, 0.0, 0.0]


@pytest.mark.asyncio
async def test_faiss_index_get_of_absent_key_returns_none(
    dense_index: FaissIndex[str, list[float]],
) -> None:
    """get() returns None for a key that was never stored."""
    # Act & Assert
    assert await dense_index.get("x") is None


@pytest.mark.asyncio
async def test_faiss_index_delegates_keys_and_ids_to_key_map(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The injected key map records the FAISS id for every stored key."""
    # Arrange
    monkeypatch.chdir(tmp_path)
    key_map: MemoryStore[str, int] = MemoryStore()
    index: FaissIndex[str, list[float]] = FaissIndex(3, key_map)

    # Act
    await index.set("x", [1.0, 0.0, 0.0])

    # Assert
    assert key_map.get("x") == FaissIndex._id("x")
    assert await index.keys() == ["x"]


@pytest.mark.asyncio
async def test_faiss_index_similar_ranks_nearest_first(
    dense_index: FaissIndex[str, list[float]],
) -> None:
    """similar() returns keys ordered by inner-product similarity."""
    # Arrange
    await dense_index.set("x", [1.0, 0.0, 0.0])
    await dense_index.set("y", [0.0, 1.0, 0.0])
    await dense_index.set("z", [0.0, 0.0, 1.0])

    # Act & Assert
    assert await dense_index.similar([0.9, 0.4, 0.1], embedding_results=3) == [
        "x",
        "y",
        "z",
    ]


@pytest.mark.asyncio
async def test_faiss_index_similar_caps_results(
    dense_index: FaissIndex[str, list[float]],
) -> None:
    """similar() returns at most the requested number of results."""
    # Arrange
    await dense_index.set("x", [1.0, 0.0, 0.0])
    await dense_index.set("y", [0.9, 0.1, 0.0])

    # Act & Assert
    assert await dense_index.similar([1.0, 0.0, 0.0], embedding_results=1) == ["x"]


@pytest.mark.asyncio
async def test_faiss_index_similar_on_empty_index(
    dense_index: FaissIndex[str, list[float]],
) -> None:
    """similar() on an index with no embeddings returns no keys."""
    # Act & Assert
    assert await dense_index.similar([1.0, 0.0, 0.0]) == []


@pytest.mark.asyncio
async def test_faiss_index_overwrite_reuses_the_key_slot(
    dense_index: FaissIndex[str, list[float]],
) -> None:
    """Restoring a key replaces its embedding without growing the index."""
    # Act
    await dense_index.set("x", [1.0, 0.0, 0.0])
    await dense_index.set("x", [0.0, 1.0, 0.0])

    # Assert
    assert dense_index.index.ntotal == 1
    assert await dense_index.get("x") == [0.0, 1.0, 0.0]
    assert await dense_index.similar([0.0, 1.0, 0.0], embedding_results=1) == ["x"]


@pytest.mark.asyncio
async def test_faiss_index_remove_drops_key_from_results(
    dense_index: FaissIndex[str, list[float]],
) -> None:
    """A removed key no longer appears in similarity results."""
    # Arrange
    await dense_index.set("x", [1.0, 0.0, 0.0])
    await dense_index.set("y", [0.0, 1.0, 0.0])

    # Act
    await dense_index.remove("x")

    # Assert
    assert await dense_index.keys() == ["y"]
    assert await dense_index.similar([1.0, 0.0, 0.0], embedding_results=2) == ["y"]


@pytest.mark.asyncio
async def test_faiss_index_remove_of_absent_key_is_noop(
    dense_index: FaissIndex[str, list[float]],
) -> None:
    """Removing a key that was never stored leaves the index unchanged."""
    # Arrange
    await dense_index.set("x", [1.0, 0.0, 0.0])

    # Act
    await dense_index.remove("y")

    # Assert
    assert await dense_index.keys() == ["x"]
    assert await dense_index.similar([1.0, 0.0, 0.0]) == ["x"]


@pytest.mark.asyncio
async def test_faiss_index_can_use_a_file_backed_key_map(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A FileStore serves as the key map, sealing the mapping on disk."""
    # Arrange
    monkeypatch.chdir(tmp_path)
    index: FaissIndex[str, list[float]] = FaissIndex(3, FileStore())

    # Act
    await index.set("x", [1.0, 0.0, 0.0])
    await index.set("y", [0.0, 1.0, 0.0])

    # Assert
    assert await index.similar([1.0, 0.1, 0.0], embedding_results=2) == ["x", "y"]


# --- FaissIndex, mocked FAISS wiring ----------------------------------------


@patch("rager.indexes.faiss")
def test_faiss_index_init_builds_the_index(faiss: MagicMock) -> None:
    """__init__() builds an id-mapped inner-product index of the given width."""
    # Act
    index: FaissIndex[str, list[float]] = FaissIndex(3, MemoryStore())

    # Assert
    faiss.IndexFlatIP.assert_called_once_with(3)
    faiss.IndexIDMap.assert_called_once_with(faiss.IndexFlatIP.return_value)
    assert index.index is faiss.IndexIDMap.return_value


@patch("rager.indexes.faiss")
@pytest.mark.asyncio
async def test_faiss_index_set_dedups_then_adds(
    faiss: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """set() removes any previous entry, then adds the row under its id."""
    # Arrange
    monkeypatch.chdir(tmp_path)
    index: FaissIndex[str, list[float]] = FaissIndex(3, MemoryStore())
    faiss_index = faiss.IndexIDMap.return_value

    # Act
    await index.set("x", [0.1, 0.2, 0.3])

    # Assert
    faiss.IDSelectorBatch.assert_called_once()
    faiss_index.remove_ids.assert_called_once_with(faiss.IDSelectorBatch.return_value)
    rows, ids = faiss_index.add_with_ids.call_args.args
    np.testing.assert_array_equal(rows, np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32))
    np.testing.assert_array_equal(ids, np.asarray([FaissIndex._id("x")], np.int64))


@patch("rager.indexes.faiss")
@pytest.mark.asyncio
async def test_faiss_index_similar_coalesces_concurrent_queries(
    faiss: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Concurrent similar() calls share one search, each trimmed to its own count."""
    # Arrange
    monkeypatch.chdir(tmp_path)
    index: FaissIndex[str, list[float]] = FaissIndex(3, MemoryStore())
    faiss_index = faiss.IndexIDMap.return_value
    faiss_index.ntotal = 2
    await index.set("x", [1.0, 0.0, 0.0])
    await index.set("y", [0.0, 1.0, 0.0])
    x, y = FaissIndex._id("x"), FaissIndex._id("y")
    faiss_index.search.return_value = (
        np.asarray([[0.9, 0.5], [0.8, -1.0]], dtype=np.float32),
        np.asarray([[x, y], [y, -1]], dtype=np.int64),
    )

    # Act
    first, second = await asyncio.gather(
        index.similar([1.0, 0.0, 0.0], embedding_results=2),
        index.similar([0.0, 1.0, 0.0], embedding_results=2),
    )

    # Assert
    faiss_index.search.assert_called_once()
    assert first == ["x", "y"]
    assert second == ["y"]


@patch("rager.indexes.faiss")
@pytest.mark.asyncio
async def test_faiss_index_similar_on_empty_index_does_not_search(
    faiss: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """similar() skips the FAISS search when the index holds nothing."""
    # Arrange
    monkeypatch.chdir(tmp_path)
    index: FaissIndex[str, list[float]] = FaissIndex(3, MemoryStore())
    faiss.IndexIDMap.return_value.ntotal = 0

    # Act
    result = await index.similar([1.0, 0.0, 0.0])

    # Assert
    assert result == []
    faiss.IndexIDMap.return_value.search.assert_not_called()


@patch("rager.indexes.faiss")
@pytest.mark.asyncio
async def test_faiss_index_coalesces_across_instances(
    faiss: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Concurrent similar() calls on different indexes share one search().

    concresce 0.2 coalesces batches per event-loop turn rather than per
    instance, so the second index's query is resolved through the leader
    (the first index)'s own key map instead of its own. Callers must not
    mix instances of the same batch-owning class in concurrent calls; this
    documents the resulting shared-batch behavior.
    """
    # Arrange
    monkeypatch.chdir(tmp_path)
    first: FaissIndex[str, list[float]] = FaissIndex(3, MemoryStore())
    second: FaissIndex[str, list[float]] = FaissIndex(3, MemoryStore())
    faiss_index = faiss.IndexIDMap.return_value
    faiss_index.ntotal = 1
    await first.set("x", [1.0, 0.0, 0.0])
    await second.set("y", [0.0, 1.0, 0.0])
    faiss_index.search.return_value = (
        np.asarray([[0.9], [0.9]], dtype=np.float32),
        np.asarray([[FaissIndex._id("x")], [FaissIndex._id("x")]], dtype=np.int64),
    )

    # Act
    first_result, second_result = await asyncio.gather(
        first.similar([1.0, 0.0, 0.0], embedding_results=1),
        second.similar([0.0, 1.0, 0.0], embedding_results=1),
    )

    # Assert
    faiss_index.search.assert_called_once()
    assert first_result == ["x"]
    assert second_result == ["x"]


# --- SparseIndex ------------------------------------------------------------


class _CountingTokenStore(MemoryStore[int, dict[str, float]]):
    """A token store that records which tokens are read from disk."""

    def __init__(self) -> None:
        """Initialize the store with an empty read log."""
        super().__init__()
        self.reads: list[int] = []

    @override
    def __getitem__(self, key: int) -> dict[str, float]:
        """Record the read and delegate to the in-memory map."""
        self.reads.append(key)
        return super().__getitem__(key)


def test_scores_accumulates_shared_token_weights() -> None:
    """_scores() sums the weight products across every token of the query."""
    # Arrange
    postings = {1: {"a": 1.0, "b": 0.5}, 2: {"a": 2.0}}
    query = {1: 2.0, 2: 3.0}

    # Act & Assert
    assert _scores(postings, query) == {"a": 8.0, "b": 1.0}


def test_top_keys_ranks_and_drops_non_positive() -> None:
    """_top_keys() ranks by score, caps results, and drops non-positive matches."""
    # Act & Assert
    assert _top_keys({"a": 8.0, "b": 1.0, "c": 0.0}, 2) == ["a", "b"]
    assert _top_keys({"a": 8.0, "b": 1.0}, 1) == ["a"]


@pytest.mark.asyncio
async def test_sparse_index_get_returns_stored_embedding(
    sparse_index: SparseIndex[str],
) -> None:
    """get() returns the weight map stored under the key."""
    # Act
    await sparse_index.set("x", {1: 1.0, 2: 0.5})

    # Assert
    assert await sparse_index.get("x") == {1: 1.0, 2: 0.5}


@pytest.mark.asyncio
async def test_sparse_index_get_of_absent_key_returns_none(
    sparse_index: SparseIndex[str],
) -> None:
    """get() returns None for a key that was never stored."""
    # Act & Assert
    assert await sparse_index.get("x") is None


@pytest.mark.asyncio
async def test_sparse_index_similar_ranks_nearest_first(
    sparse_index: SparseIndex[str],
) -> None:
    """similar() returns keys ordered by inner-product similarity."""
    # Arrange
    await sparse_index.set("x", {1: 1.0})
    await sparse_index.set("y", {1: 0.5, 2: 0.5})

    # Act & Assert
    assert await sparse_index.similar({1: 1.0}) == ["x", "y"]


@pytest.mark.asyncio
async def test_sparse_index_similar_drops_non_matching_keys(
    sparse_index: SparseIndex[str],
) -> None:
    """similar() omits stored embeddings sharing no tokens with the query."""
    # Arrange
    await sparse_index.set("x", {1: 1.0})
    await sparse_index.set("y", {2: 1.0})

    # Act & Assert
    assert await sparse_index.similar({1: 1.0}) == ["x"]


@pytest.mark.asyncio
async def test_sparse_index_similar_caps_results(
    sparse_index: SparseIndex[str],
) -> None:
    """similar() returns at most the requested number of results."""
    # Arrange
    await sparse_index.set("x", {1: 1.0})
    await sparse_index.set("y", {1: 0.5})

    # Act & Assert
    assert await sparse_index.similar({1: 1.0}, embedding_results=1) == ["x"]


@pytest.mark.asyncio
async def test_sparse_index_similar_on_empty_index(
    sparse_index: SparseIndex[str],
) -> None:
    """similar() on an index with no embeddings returns no keys."""
    # Act & Assert
    assert await sparse_index.similar({1: 1.0}) == []


@pytest.mark.asyncio
async def test_sparse_index_overwrite_rewires_the_inverted_index(
    sparse_index: SparseIndex[str],
) -> None:
    """Restoring a key retires its stale tokens and registers the new ones."""
    # Act
    await sparse_index.set("x", {1: 1.0})
    await sparse_index.set("x", {2: 1.0})

    # Assert
    assert await sparse_index.similar({1: 1.0}) == []
    assert await sparse_index.similar({2: 1.0}) == ["x"]


@pytest.mark.asyncio
async def test_sparse_index_remove_drops_key_from_results(
    sparse_index: SparseIndex[str],
) -> None:
    """A removed key no longer appears in similarity results."""
    # Arrange
    await sparse_index.set("x", {1: 1.0})
    await sparse_index.set("y", {1: 0.5})

    # Act
    await sparse_index.remove("x")

    # Assert
    assert await sparse_index.keys() == ["y"]
    assert await sparse_index.similar({1: 1.0}) == ["y"]


@pytest.mark.asyncio
async def test_sparse_index_remove_of_absent_key_is_noop(
    sparse_index: SparseIndex[str],
) -> None:
    """Removing a key that was never stored leaves the index unchanged."""
    # Arrange
    await sparse_index.set("x", {1: 1.0})

    # Act
    await sparse_index.remove("y")

    # Assert
    assert await sparse_index.keys() == ["x"]
    assert await sparse_index.similar({1: 1.0}) == ["x"]


@pytest.mark.asyncio
async def test_sparse_index_remove_tolerates_a_desynced_token_index() -> None:
    """Removing a key survives a token index cleared out from under the index."""
    # Arrange
    token_map: MemoryStore[int, dict[str, float]] = MemoryStore()
    index: SparseIndex[str] = SparseIndex(MemoryStore(), token_map)
    await index.set("x", {1: 1.0})
    token_map.clear()  # drop the inverted index behind the index's back

    # Act
    await index.remove("x")  # must not raise despite the missing posting list

    # Assert
    assert await index.keys() == []


@pytest.mark.asyncio
async def test_sparse_index_similar_loads_only_query_tokens() -> None:
    """similar() reads the posting lists of the query's tokens and no others."""
    # Arrange
    token_map = _CountingTokenStore()
    index: SparseIndex[str] = SparseIndex(MemoryStore(), token_map)
    await index.set("a", {1: 1.0})
    await index.set("b", {2: 1.0, 3: 1.0})
    token_map.reads.clear()

    # Act
    await index.similar({2: 1.0})

    # Assert
    assert token_map.reads == [2]


@pytest.mark.asyncio
async def test_sparse_index_batches_concurrent_calls(
    sparse_index: SparseIndex[str],
) -> None:
    """Concurrent calls are batched, yet each caller gets its own result."""
    # Arrange
    await sparse_index.set("x", {1: 1.0})
    await sparse_index.set("y", {2: 1.0})

    # Act
    x_result, y_result = await asyncio.gather(
        sparse_index.similar({1: 1.0}), sparse_index.similar({2: 1.0})
    )

    # Assert
    assert x_result == ["x"]
    assert y_result == ["y"]


@pytest.mark.asyncio
async def test_sparse_index_coalesces_across_instances() -> None:
    """Concurrent calls on different indexes share one leader's posting lists.

    concresce 0.2 coalesces batches per event-loop turn rather than per
    instance, so the second index's query is resolved through the leader
    (the first index)'s own postings instead of its own. Callers must not
    mix instances of the same batch-owning class in concurrent calls; this
    documents the resulting shared-batch behavior.
    """
    # Arrange
    first: SparseIndex[str] = SparseIndex(MemoryStore(), MemoryStore())
    second: SparseIndex[str] = SparseIndex(MemoryStore(), MemoryStore())
    await first.set("x", {1: 1.0})
    await second.set("y", {1: 1.0})

    # Act
    x_result, y_result = await asyncio.gather(
        first.similar({1: 1.0}), second.similar({1: 1.0})
    )

    # Assert
    assert x_result == ["x"]
    assert y_result == ["x"]


@pytest.mark.asyncio
async def test_sparse_index_works_with_file_backed_stores(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """File-backed stores seal the embeddings and token index on disk."""
    # Arrange
    monkeypatch.chdir(tmp_path)
    index: SparseIndex[str] = SparseIndex(FileStore(), FileStore())

    # Act
    await index.set("x", {1: 1.0})
    await index.set("y", {1: 0.5})

    # Assert
    assert await index.similar({1: 1.0}) == ["x", "y"]
    assert await index.get("x") == {1: 1.0}
    assert list((tmp_path / ".jar" / "stores").iterdir())


@pytest.mark.asyncio
async def test_sparse_index_drops_unsealed_postings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """similar() skips tokens whose sealed posting list is gone from the jar."""
    # Arrange
    monkeypatch.chdir(tmp_path)
    index: SparseIndex[str] = SparseIndex(FileStore(), FileStore())
    await index.set("x", {1: 1.0})
    for file in (tmp_path / ".jar" / "stores").iterdir():
        file.unlink()

    # Act & Assert
    assert await index.similar({1: 1.0}) == []
