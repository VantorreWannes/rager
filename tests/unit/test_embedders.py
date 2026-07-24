"""Unit tests for embedders."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rager.embedders import BaseEmbedder, SentenceTransformerEmbedder, SpladeEmbedder
from rager.types import DenseEmbedding, SparseEmbedding

pytestmark = pytest.mark.unit


def test_base_embedder_is_abstract() -> None:
    """BaseEmbedder cannot be instantiated without the jar and batch operations."""
    # Act & Assert
    with pytest.raises(TypeError):
        BaseEmbedder()  # type: ignore[abstract]


@patch("rager.embedders.SentenceTransformer")
def test_sentence_transformer_embedder_model(sentence_transformer: MagicMock) -> None:
    """Test that the model property returns the embedder."""
    # Arrange
    embedder = SentenceTransformerEmbedder("test-model")

    # Act
    _ = embedder.model

    # Assert
    sentence_transformer.assert_called_once_with("test-model")


@patch("rager.embedders.SentenceTransformerEmbedder.model")
@pytest.mark.asyncio
async def test_sentence_transformer_embedder_batched(model: MagicMock) -> None:
    """_batched() encodes the collected batch in a single encode() call."""
    # Arrange
    embedder = SentenceTransformerEmbedder("test-model")
    model.encode.return_value.tolist.return_value = [[0.1, 0.2, 0.3]]

    # Act
    result = await embedder._batched(["chunk"])

    # Assert
    model.encode.assert_called_once_with(["chunk"], normalize_embeddings=True)
    assert result == [[0.1, 0.2, 0.3]]


@patch.object(SentenceTransformerEmbedder, "_batched", new_callable=AsyncMock)
@patch("rager.embedders.Jar")
@pytest.mark.asyncio
async def test_sentence_transformer_embedder_embed(
    jar_cls: MagicMock, batched: AsyncMock
) -> None:
    """embed() folds its identity into the jar and seals the embedding."""
    # Arrange
    embedder = SentenceTransformerEmbedder("test-model")
    batched.return_value = [[0.1, 0.2, 0.3]]
    jar = jar_cls[DenseEmbedding].return_value
    jar.get.return_value = None
    jar.set.side_effect = lambda value: value

    # Act
    result = await embedder.embed("chunk")

    # Assert
    batched.assert_awaited_once_with(["chunk"])
    jar.include.assert_any_call("test-model")
    jar.include.assert_any_call("chunk")
    jar.set.assert_called_once_with([0.1, 0.2, 0.3])
    assert result == [0.1, 0.2, 0.3]


@patch.object(SentenceTransformerEmbedder, "_batched", new_callable=AsyncMock)
@patch("rager.embedders.Jar")
@pytest.mark.asyncio
async def test_sentence_transformer_embedder_embed_cached(
    jar_cls: MagicMock, batched: AsyncMock
) -> None:
    """embed() returns the sealed embedding without encoding on a cache hit."""
    # Arrange
    embedder = SentenceTransformerEmbedder("test-model")
    jar = jar_cls[DenseEmbedding].return_value
    jar.get.return_value = [0.1, 0.2, 0.3]

    # Act
    result = await embedder.embed("chunk")

    # Assert
    batched.assert_not_awaited()
    jar.set.assert_not_called()
    assert result == [0.1, 0.2, 0.3]


@patch.object(SentenceTransformerEmbedder, "_batched", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_sentence_transformer_embedder_batches_concurrent_embed_calls(
    batched: AsyncMock,
) -> None:
    """Concurrent embed() calls for different chunks share one _batched() call."""
    # Arrange
    embedder = SentenceTransformerEmbedder("test-model")
    batched.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    # Act
    first_result, second_result = await asyncio.gather(
        embedder._embed("one"), embedder._embed("two")
    )

    # Assert
    batched.assert_awaited_once_with(["one", "two"])
    assert first_result == [0.1, 0.2, 0.3]
    assert second_result == [0.4, 0.5, 0.6]


@patch("rager.embedders.SentenceTransformer")
@pytest.mark.asyncio
async def test_sentence_transformer_embedder_coalesces_across_instances(
    sentence_transformer: MagicMock,
) -> None:
    """Concurrent _embed() calls on different embedders share one encode() call.

    concresce 0.2 coalesces batches per event-loop turn rather than per
    instance. Callers must not mix instances of the same batch-owning class
    in concurrent calls; this documents the resulting shared-batch behavior.
    """
    # Arrange
    first = SentenceTransformerEmbedder("model-a")
    second = SentenceTransformerEmbedder("model-b")
    model = sentence_transformer.return_value
    model.encode.return_value.tolist.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    # Act
    first_result, second_result = await asyncio.gather(
        first._embed("one"), second._embed("two")
    )

    # Assert
    model.encode.assert_called_once_with(["one", "two"], normalize_embeddings=True)
    assert first_result == [0.1, 0.2, 0.3]
    assert second_result == [0.4, 0.5, 0.6]


@patch("rager.embedders.SparseEncoder")
def test_splade_embedder_model(sparse_encoder: MagicMock) -> None:
    """Test that the model property returns the sparse encoder."""
    # Arrange
    embedder = SpladeEmbedder("test-model")

    # Act
    _ = embedder.model

    # Assert
    sparse_encoder.assert_called_once_with("test-model")


def test_splade_embedder_coalesced_to_embeddings() -> None:
    """_coalesced_to_embeddings() buckets (token, weight) pairs by their row."""
    # Arrange
    coalesced = MagicMock()
    coalesced.indices.return_value = [[0, 1, 1], [5, 9, 7]]
    coalesced.values.return_value = [1.5, 2.5, 3.5]

    # Act
    result = SpladeEmbedder._coalesced_to_embeddings(coalesced, 2)

    # Assert
    assert result == [{5: 1.5}, {9: 2.5, 7: 3.5}]


@patch("rager.embedders.SpladeEmbedder.model")
@pytest.mark.asyncio
async def test_splade_embedder_batched(model: MagicMock) -> None:
    """_batched() coalesces the batch into a {token_id: weight} mapping per chunk."""
    # Arrange
    embedder = SpladeEmbedder("test-model")
    coalesced = model.encode.return_value.coalesce.return_value
    coalesced.indices.return_value = [[0, 0], [5, 9]]
    coalesced.values.return_value = [1.5, 2.5]

    # Act
    result = await embedder._batched(["chunk"])

    # Assert
    model.encode.assert_called_once_with(["chunk"])
    assert result == [{5: 1.5, 9: 2.5}]


@patch.object(SpladeEmbedder, "_batched", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_splade_embedder_batches_concurrent_embed_calls(
    batched: AsyncMock,
) -> None:
    """Concurrent embed() calls for different chunks share one _batched() call."""
    # Arrange
    embedder = SpladeEmbedder("test-model")
    batched.return_value = [{5: 1.5}, {9: 2.5}]

    # Act
    first_result, second_result = await asyncio.gather(
        embedder._embed("one"), embedder._embed("two")
    )

    # Assert
    batched.assert_awaited_once_with(["one", "two"])
    assert first_result == {5: 1.5}
    assert second_result == {9: 2.5}


@patch("rager.embedders.SparseEncoder")
@pytest.mark.asyncio
async def test_splade_embedder_coalesces_across_instances(
    sparse_encoder: MagicMock,
) -> None:
    """Concurrent _embed() calls on different embedders share one encode() call.

    concresce 0.2 coalesces batches per event-loop turn rather than per
    instance. Callers must not mix instances of the same batch-owning class
    in concurrent calls; this documents the resulting shared-batch behavior.
    """
    # Arrange
    first = SpladeEmbedder("model-a")
    second = SpladeEmbedder("model-b")
    coalesced = sparse_encoder.return_value.encode.return_value.coalesce.return_value
    coalesced.indices.return_value = [[0, 1], [5, 9]]
    coalesced.values.return_value = [1.5, 2.5]

    # Act
    first_result, second_result = await asyncio.gather(
        first._embed("one"), second._embed("two")
    )

    # Assert
    sparse_encoder.return_value.encode.assert_called_once_with(["one", "two"])
    assert first_result == {5: 1.5}
    assert second_result == {9: 2.5}


@patch.object(SpladeEmbedder, "_batched", new_callable=AsyncMock)
@patch("rager.embedders.Jar")
@pytest.mark.asyncio
async def test_splade_embedder_embed(jar_cls: MagicMock, batched: AsyncMock) -> None:
    """embed() folds its identity into the jar and seals the embedding."""
    # Arrange
    embedder = SpladeEmbedder("test-model")
    batched.return_value = [{5: 1.5, 9: 2.5}]
    jar = jar_cls[SparseEmbedding].return_value
    jar.get.return_value = None
    jar.set.side_effect = lambda value: value

    # Act
    result = await embedder.embed("chunk")

    # Assert
    batched.assert_awaited_once_with(["chunk"])
    jar.include.assert_any_call("test-model")
    jar.include.assert_any_call("chunk")
    jar.set.assert_called_once_with({5: 1.5, 9: 2.5})
    assert result == {5: 1.5, 9: 2.5}


@patch.object(SpladeEmbedder, "_batched", new_callable=AsyncMock)
@patch("rager.embedders.Jar")
@pytest.mark.asyncio
async def test_splade_embedder_embed_cached(
    jar_cls: MagicMock, batched: AsyncMock
) -> None:
    """embed() returns the sealed embedding without encoding on a cache hit."""
    # Arrange
    embedder = SpladeEmbedder("test-model")
    jar = jar_cls[SparseEmbedding].return_value
    jar.get.return_value = {5: 1.5, 9: 2.5}

    # Act
    result = await embedder.embed("chunk")

    # Assert
    batched.assert_not_awaited()
    jar.set.assert_not_called()
    assert result == {5: 1.5, 9: 2.5}
