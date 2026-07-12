"""Integration tests for indexes."""

import asyncio

import pytest

from rager.embedders import SentenceTransformerDenseEmbedder
from rager.indexes import DenseIndex

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_dense_index_similar_ranks_nearest_first() -> None:
    """similar() returns keys ordered by inner-product similarity."""
    index = DenseIndex(3)
    x_key = await index.add([1.0, 0.0, 0.0])
    y_key = await index.add([0.0, 1.0, 0.0])
    z_key = await index.add([0.0, 0.0, 1.0])

    result = await index.similar([0.9, 0.4, 0.1])

    assert result == [x_key, y_key, z_key]


@pytest.mark.asyncio
async def test_dense_index_add_is_idempotent() -> None:
    """Adding the same embedding twice yields one key and one entry."""
    index = DenseIndex(3)

    first = await index.add([1.0, 0.0, 0.0])
    second = await index.add([1.0, 0.0, 0.0])

    assert first == second
    assert await index.similar([1.0, 0.0, 0.0]) == [first]


@pytest.mark.asyncio
async def test_dense_index_remove_drops_key_from_results() -> None:
    """Removed keys no longer appear in similarity results."""
    index = DenseIndex(3)
    x_key = await index.add([1.0, 0.0, 0.0])
    y_key = await index.add([0.0, 1.0, 0.0])

    await index.remove(x_key)

    assert await index.similar([1.0, 0.0, 0.0]) == [y_key]


@pytest.mark.asyncio
async def test_dense_index_remove_of_absent_key_is_noop() -> None:
    """Removing a key that was never added leaves the index unchanged."""
    index = DenseIndex(3)
    x_key = await index.add([1.0, 0.0, 0.0])

    await index.remove(x_key + 1)

    assert await index.similar([1.0, 0.0, 0.0]) == [x_key]


@pytest.mark.asyncio
async def test_dense_index_similar_on_empty_index() -> None:
    """similar() on an empty index returns no keys."""
    index = DenseIndex(3)

    assert await index.similar([1.0, 0.0, 0.0]) == []


@pytest.mark.asyncio
async def test_dense_index_batches_concurrent_adds() -> None:
    """Concurrent add() calls are batched, yet each caller gets its own key."""
    index = DenseIndex(3)

    x_key, y_key = await asyncio.gather(
        index.add([1.0, 0.0, 0.0]), index.add([0.0, 1.0, 0.0])
    )

    assert x_key != y_key
    assert await index.similar([1.0, 0.0, 0.0]) == [x_key, y_key]


@pytest.mark.asyncio
async def test_dense_index_answers_concurrent_queries() -> None:
    """Concurrent similar() calls are batched, yet each gets its own ranking."""
    index = DenseIndex(3)
    x_key = await index.add([1.0, 0.0, 0.0])
    y_key = await index.add([0.0, 1.0, 0.0])

    x_result, y_result = await asyncio.gather(
        index.similar([1.0, 0.0, 0.0]), index.similar([0.0, 1.0, 0.0])
    )

    assert x_result == [x_key, y_key]
    assert y_result == [y_key, x_key]


@pytest.mark.asyncio
async def test_dense_index_with_dense_embedder() -> None:
    """Embeddings from the dense embedder retrieve the semantically closest chunk."""
    embedder = SentenceTransformerDenseEmbedder("all-MiniLM-L6-v2")
    index = DenseIndex(384, results=2)
    cats_key = await index.add(await embedder.embed("Cats purr when they are happy."))
    await index.add(await embedder.embed("The stock market closed higher today."))

    result = await index.similar(await embedder.embed("A kitten is purring."))

    assert result[0] == cats_key
