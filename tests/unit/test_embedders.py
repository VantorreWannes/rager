"""Unit tests for embedders."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rager.embedders import SentenceTransformerDenseEmbedder, SpladeSparseEmbedder

pytestmark = pytest.mark.unit


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
@patch("rager.embedders.belljar.check")
@patch("rager.embedders.belljar.include")
@pytest.mark.asyncio
async def test_sentence_transformer_dense_embedder_embed(
    include: MagicMock, check: MagicMock, encode: AsyncMock
) -> None:
    """embed() folds its identity into belljar and delegates to _encode."""
    # Arrange
    embedder = SentenceTransformerDenseEmbedder("test-model")
    encode.return_value = [0.1, 0.2, 0.3]

    # Act
    result = await embedder.embed("chunk")

    # Assert
    encode.assert_awaited_once_with("chunk")
    include.assert_any_call("test-model")
    include.assert_any_call("chunk")
    check.assert_called_once()
    assert result == [0.1, 0.2, 0.3]


@patch("rager.embedders.SentenceTransformer")
@pytest.mark.asyncio
async def test_sentence_transformer_dense_embedder_does_not_batch_across_instances(
    sentence_transformer: MagicMock,
) -> None:
    """Concurrent _encode() calls on different embedders batch separately."""
    # Arrange
    first = SentenceTransformerDenseEmbedder("model-a")
    second = SentenceTransformerDenseEmbedder("model-b")
    model = sentence_transformer.return_value
    model.encode.return_value.tolist.return_value = [[0.1, 0.2, 0.3]]

    # Act
    await asyncio.gather(first._encode("one"), second._encode("two"))

    # Assert
    expected_calls = 2
    assert model.encode.call_count == expected_calls


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
async def test_splade_sparse_embedder_does_not_batch_across_instances(
    sparse_encoder: MagicMock,
) -> None:
    """Concurrent _encode() calls on different embedders batch separately."""
    # Arrange
    first = SpladeSparseEmbedder("model-a")
    second = SpladeSparseEmbedder("model-b")
    coalesced = sparse_encoder.return_value.encode.return_value.coalesce.return_value
    coalesced.indices.return_value = [[0], [5]]
    coalesced.values.return_value = [1.5]

    # Act
    await asyncio.gather(first._encode("one"), second._encode("two"))

    # Assert
    expected_calls = 2
    assert sparse_encoder.return_value.encode.call_count == expected_calls


@patch.object(SpladeSparseEmbedder, "_encode", new_callable=AsyncMock)
@patch("rager.embedders.belljar.check")
@patch("rager.embedders.belljar.include")
@pytest.mark.asyncio
async def test_splade_sparse_embedder_embed(
    include: MagicMock, check: MagicMock, encode: AsyncMock
) -> None:
    """embed() folds its identity into belljar and delegates to _encode."""
    # Arrange
    embedder = SpladeSparseEmbedder("test-model")
    encode.return_value = {5: 1.5, 9: 2.5}

    # Act
    result = await embedder.embed("chunk")

    # Assert
    encode.assert_awaited_once_with("chunk")
    include.assert_any_call("test-model")
    include.assert_any_call("chunk")
    check.assert_called_once()
    assert result == {5: 1.5, 9: 2.5}
