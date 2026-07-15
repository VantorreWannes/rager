"""Integration tests for indexes."""

import asyncio

import pytest

from rager.embedders import SentenceTransformerDenseEmbedder, SpladeSparseEmbedder
from rager.indexes import FaissIndex, SparseIndex
from rager.stores import FileStore, MemoryStore

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_dense_index_similar_ranks_nearest_first() -> None:
    """similar() returns keys ordered by inner-product similarity."""
    index: FaissIndex[str, list[float]] = FaissIndex(3, MemoryStore())
    await index.set("x", [1.0, 0.0, 0.0])
    await index.set("y", [0.0, 1.0, 0.0])
    await index.set("z", [0.0, 0.0, 1.0])

    result = await index.similar([0.9, 0.4, 0.1])

    assert result == ["x", "y", "z"]


@pytest.mark.asyncio
async def test_dense_index_overwrite_replaces_embedding() -> None:
    """Restoring a key replaces its embedding without growing the index."""
    index: FaissIndex[str, list[float]] = FaissIndex(3, MemoryStore())

    await index.set("x", [1.0, 0.0, 0.0])
    await index.set("x", [0.0, 1.0, 0.0])

    assert index.index.ntotal == 1
    assert await index.similar([0.0, 1.0, 0.0]) == ["x"]


@pytest.mark.asyncio
async def test_dense_index_remove_drops_key_from_results() -> None:
    """Removed keys no longer appear in similarity results."""
    index: FaissIndex[str, list[float]] = FaissIndex(3, MemoryStore())
    await index.set("x", [1.0, 0.0, 0.0])
    await index.set("y", [0.0, 1.0, 0.0])

    await index.remove("x")

    assert await index.similar([1.0, 0.0, 0.0]) == ["y"]


@pytest.mark.asyncio
async def test_dense_index_remove_of_absent_key_is_noop() -> None:
    """Removing a key that was never added leaves the index unchanged."""
    index: FaissIndex[str, list[float]] = FaissIndex(3, MemoryStore())
    await index.set("x", [1.0, 0.0, 0.0])

    await index.remove("y")

    assert await index.similar([1.0, 0.0, 0.0]) == ["x"]


@pytest.mark.asyncio
async def test_dense_index_similar_on_empty_index() -> None:
    """similar() on an empty index returns no keys."""
    index: FaissIndex[str, list[float]] = FaissIndex(3, MemoryStore())

    assert await index.similar([1.0, 0.0, 0.0]) == []


@pytest.mark.asyncio
async def test_dense_index_answers_concurrent_queries() -> None:
    """Concurrent similar() calls are batched, yet each gets its own ranking."""
    index: FaissIndex[str, list[float]] = FaissIndex(3, MemoryStore())
    await index.set("x", [1.0, 0.0, 0.0])
    await index.set("y", [0.0, 1.0, 0.0])

    x_result, y_result = await asyncio.gather(
        index.similar([1.0, 0.0, 0.0]), index.similar([0.0, 1.0, 0.0])
    )

    assert x_result == ["x", "y"]
    assert y_result == ["y", "x"]


@pytest.mark.asyncio
async def test_dense_index_instances_do_not_share_queries() -> None:
    """Concurrent calls on different indexes each answer from their own index."""
    first: FaissIndex[str, list[float]] = FaissIndex(3, MemoryStore())
    second: FaissIndex[str, list[float]] = FaissIndex(3, MemoryStore())
    await first.set("x", [1.0, 0.0, 0.0])
    await second.set("y", [0.0, 1.0, 0.0])

    x_result, y_result = await asyncio.gather(
        first.similar([1.0, 0.0, 0.0]), second.similar([0.0, 1.0, 0.0])
    )

    assert x_result == ["x"]
    assert y_result == ["y"]


@pytest.mark.asyncio
async def test_sparse_index_ranks_nearest_first() -> None:
    """similar() ranks stored weight maps by inner product with the query."""
    index: SparseIndex[str] = SparseIndex(MemoryStore(), MemoryStore())
    await index.set("x", {1: 1.0})
    await index.set("y", {1: 0.5, 2: 0.5})
    await index.set("z", {3: 1.0})

    assert await index.similar({1: 1.0}) == ["x", "y"]


@pytest.mark.asyncio
async def test_dense_index_with_dense_embedder() -> None:
    """Embeddings from the dense embedder retrieve the semantically closest chunk."""
    embedder = SentenceTransformerDenseEmbedder("all-MiniLM-L6-v2")
    index: FaissIndex[str, list[float]] = FaissIndex(384, MemoryStore())
    await index.set("cats", await embedder.embed("Cats purr when they are happy."))
    await index.set(
        "stocks", await embedder.embed("The stock market closed higher today.")
    )

    query = await embedder.embed("A kitten is purring.")
    result = await index.similar(query, embedding_results=2)

    assert result[0] == "cats"


@pytest.mark.asyncio
async def test_sparse_index_with_sparse_embedder() -> None:
    """Embeddings from the sparse embedder retrieve the closest chunk."""
    embedder = SpladeSparseEmbedder("prithivida/Splade_PP_en_v1")
    index: SparseIndex[str] = SparseIndex(MemoryStore(), MemoryStore())
    await index.set("cats", await embedder.embed("Cats purr when they are happy."))
    await index.set(
        "stocks", await embedder.embed("The stock market closed higher today.")
    )

    query = await embedder.embed("A kitten is purring.")
    result = await index.similar(query, embedding_results=2)

    assert result[0] == "cats"


@pytest.mark.asyncio
async def test_sparse_index_with_file_backed_stores() -> None:
    """Embeddings sealed on disk through FileStores retrieve the closest chunk."""
    embedder = SpladeSparseEmbedder("prithivida/Splade_PP_en_v1")
    index: SparseIndex[str] = SparseIndex(FileStore(), FileStore())
    await index.set("cats", await embedder.embed("Cats purr when they are happy."))
    await index.set(
        "stocks", await embedder.embed("The stock market closed higher today.")
    )

    query = await embedder.embed("A kitten is purring.")
    result = await index.similar(query, embedding_results=2)

    assert result[0] == "cats"
