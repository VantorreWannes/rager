"""Unit tests for embedders."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rager.embedders import (
    BaseEmbedder,
    SentenceTransformerDenseEmbedder,
    SpladeSparseEmbedder,
)
from rager.types import DenseEmbedding, SparseEmbedding

pytestmark = pytest.mark.unit


def test_base_embedder_is_abstract() -> None:
    """BaseEmbedder cannot be instantiated without the jar and encode operations."""
    # Act & Assert
    with pytest.raises(TypeError):
        BaseEmbedder()  # type: ignore[abstract]


@patch("rager.embedders.SentenceTransformer")
def test_sentence_transformer_dense_embedder_model(
    sentence_transformer: MagicMock,
) -> None:
    """Test that the model property returns the embedder."""
    # Arrange
    embedder = SentenceTransformerDenseEmbedder("test-model")

    # Act
    _ = embedder.model

    # Assert
    sentence_transformer.assert_called_once_with("test-model")


@patch("rager.embedders.SentenceTransformerDenseEmbedder.model")
@pytest.mark.asyncio
async def test_sentence_transformer_dense_embedder_encode(model: MagicMock) -> None:
    """_encode() encodes the collected batch and returns this caller's vector."""
    # Arrange
    embedder = SentenceTransformerDenseEmbedder("test-model")
    model.encode.return_value.tolist.return_value = [[0.1, 0.2, 0.3]]

    # Act
    result = await embedder._encode("chunk")

    # Assert
    model.encode.assert_called_once_with(["chunk"], normalize_embeddings=True)
    assert result == [0.1, 0.2, 0.3]


@patch.object(SentenceTransformerDenseEmbedder, "_encode", new_callable=AsyncMock)
@patch("rager.embedders.Jar")
@pytest.mark.asyncio
async def test_sentence_transformer_dense_embedder_embed(
    jar_cls: MagicMock, encode: AsyncMock
) -> None:
    """embed() folds its identity into the jar and seals the embedding."""
    # Arrange
    embedder = SentenceTransformerDenseEmbedder("test-model")
    encode.return_value = [0.1, 0.2, 0.3]
    jar = jar_cls[DenseEmbedding].return_value
    jar.get.return_value = None
    jar.set.side_effect = lambda value: value

    # Act
    result = await embedder.embed("chunk")

    # Assert
    encode.assert_awaited_once_with("chunk")
    jar.include.assert_any_call("test-model")
    jar.include.assert_any_call("chunk")
    jar.set.assert_called_once_with([0.1, 0.2, 0.3])
    assert result == [0.1, 0.2, 0.3]


@patch.object(SentenceTransformerDenseEmbedder, "_encode", new_callable=AsyncMock)
@patch("rager.embedders.Jar")
@pytest.mark.asyncio
async def test_sentence_transformer_dense_embedder_embed_cached(
    jar_cls: MagicMock, encode: AsyncMock
) -> None:
    """embed() returns the sealed embedding without encoding on a cache hit."""
    # Arrange
    embedder = SentenceTransformerDenseEmbedder("test-model")
    jar = jar_cls[DenseEmbedding].return_value
    jar.get.return_value = [0.1, 0.2, 0.3]

    # Act
    result = await embedder.embed("chunk")

    # Assert
    encode.assert_not_awaited()
    jar.set.assert_not_called()
    assert result == [0.1, 0.2, 0.3]


@patch("rager.embedders.SentenceTransformer")
@pytest.mark.asyncio
async def test_sentence_transformer_dense_embedder_coalesces_across_instances(
    sentence_transformer: MagicMock,
) -> None:
    """Concurrent _encode() calls on different embedders share one encode() call.

    concresce 0.2 coalesces batches per event-loop turn rather than per
    instance. Callers must not mix instances of the same batch-owning class
    in concurrent calls; this documents the resulting shared-batch behavior.
    """
    # Arrange
    first = SentenceTransformerDenseEmbedder("model-a")
    second = SentenceTransformerDenseEmbedder("model-b")
    model = sentence_transformer.return_value
    model.encode.return_value.tolist.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    # Act
    first_result, second_result = await asyncio.gather(
        first._encode("one"), second._encode("two")
    )

    # Assert
    model.encode.assert_called_once_with(["one", "two"], normalize_embeddings=True)
    assert first_result == [0.1, 0.2, 0.3]
    assert second_result == [0.4, 0.5, 0.6]


@patch("rager.embedders.SparseEncoder")
def test_splade_sparse_embedder_model(sparse_encoder: MagicMock) -> None:
    """Test that the model property returns the sparse encoder."""
    # Arrange
    embedder = SpladeSparseEmbedder("test-model")

    # Act
    _ = embedder.model

    # Assert
    sparse_encoder.assert_called_once_with("test-model")


def test_splade_sparse_embedder_coalesced_to_embeddings() -> None:
    """_coalesced_to_embeddings() buckets (token, weight) pairs by their row."""
    # Arrange
    coalesced = MagicMock()
    coalesced.indices.return_value = [[0, 1, 1], [5, 9, 7]]
    coalesced.values.return_value = [1.5, 2.5, 3.5]

    # Act
    result = SpladeSparseEmbedder._coalesced_to_embeddings(coalesced, 2)

    # Assert
    assert result == [{5: 1.5}, {9: 2.5, 7: 3.5}]


@patch("rager.embedders.SpladeSparseEmbedder.model")
@pytest.mark.asyncio
async def test_splade_sparse_embedder_encode(model: MagicMock) -> None:
    """_encode() coalesces the batch into a {token_id: weight} mapping."""
    # Arrange
    embedder = SpladeSparseEmbedder("test-model")
    coalesced = model.encode.return_value.coalesce.return_value
    coalesced.indices.return_value = [[0, 0], [5, 9]]
    coalesced.values.return_value = [1.5, 2.5]

    # Act
    result = await embedder._encode("chunk")

    # Assert
    model.encode.assert_called_once_with(["chunk"])
    assert result == {5: 1.5, 9: 2.5}


@patch("rager.embedders.SparseEncoder")
@pytest.mark.asyncio
async def test_splade_sparse_embedder_coalesces_across_instances(
    sparse_encoder: MagicMock,
) -> None:
    """Concurrent _encode() calls on different embedders share one encode() call.

    concresce 0.2 coalesces batches per event-loop turn rather than per
    instance. Callers must not mix instances of the same batch-owning class
    in concurrent calls; this documents the resulting shared-batch behavior.
    """
    # Arrange
    first = SpladeSparseEmbedder("model-a")
    second = SpladeSparseEmbedder("model-b")
    coalesced = sparse_encoder.return_value.encode.return_value.coalesce.return_value
    coalesced.indices.return_value = [[0, 1], [5, 9]]
    coalesced.values.return_value = [1.5, 2.5]

    # Act
    first_result, second_result = await asyncio.gather(
        first._encode("one"), second._encode("two")
    )

    # Assert
    sparse_encoder.return_value.encode.assert_called_once_with(["one", "two"])
    assert first_result == {5: 1.5}
    assert second_result == {9: 2.5}


@patch.object(SpladeSparseEmbedder, "_encode", new_callable=AsyncMock)
@patch("rager.embedders.Jar")
@pytest.mark.asyncio
async def test_splade_sparse_embedder_embed(
    jar_cls: MagicMock, encode: AsyncMock
) -> None:
    """embed() folds its identity into the jar and seals the embedding."""
    # Arrange
    embedder = SpladeSparseEmbedder("test-model")
    encode.return_value = {5: 1.5, 9: 2.5}
    jar = jar_cls[SparseEmbedding].return_value
    jar.get.return_value = None
    jar.set.side_effect = lambda value: value

    # Act
    result = await embedder.embed("chunk")

    # Assert
    encode.assert_awaited_once_with("chunk")
    jar.include.assert_any_call("test-model")
    jar.include.assert_any_call("chunk")
    jar.set.assert_called_once_with({5: 1.5, 9: 2.5})
    assert result == {5: 1.5, 9: 2.5}


@patch.object(SpladeSparseEmbedder, "_encode", new_callable=AsyncMock)
@patch("rager.embedders.Jar")
@pytest.mark.asyncio
async def test_splade_sparse_embedder_embed_cached(
    jar_cls: MagicMock, encode: AsyncMock
) -> None:
    """embed() returns the sealed embedding without encoding on a cache hit."""
    # Arrange
    embedder = SpladeSparseEmbedder("test-model")
    jar = jar_cls[SparseEmbedding].return_value
    jar.get.return_value = {5: 1.5, 9: 2.5}

    # Act
    result = await embedder.embed("chunk")

    # Assert
    encode.assert_not_awaited()
    jar.set.assert_not_called()
    assert result == {5: 1.5, 9: 2.5}
