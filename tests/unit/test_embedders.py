"""Unit tests for embedders."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rager.embedders import SentenceTransformerDenseEmbedder

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
def test_sentence_transformer_dense_embedder_encode(model: MagicMock) -> None:
    """_encode() encodes the collected batch and returns this caller's vector."""
    # Arrange
    embedder = SentenceTransformerDenseEmbedder("test-model")
    model.encode.return_value.tolist.return_value = [[0.1, 0.2, 0.3]]

    # Act
    result = asyncio.run(embedder._encode("chunk"))  # noqa: SLF001

    # Assert
    model.encode.assert_called_once_with(["chunk"], normalize_embeddings=True)
    assert result == [0.1, 0.2, 0.3]


@patch.object(SentenceTransformerDenseEmbedder, "_encode", new_callable=AsyncMock)
@patch("rager.embedders.belljar.check")
@patch("rager.embedders.belljar.include")
def test_sentence_transformer_dense_embedder_embed(
    include: MagicMock, check: MagicMock, encode: AsyncMock
) -> None:
    """embed() folds its identity into belljar and delegates to _encode."""
    # Arrange
    embedder = SentenceTransformerDenseEmbedder("test-model")
    encode.return_value = [0.1, 0.2, 0.3]

    # Act
    result = asyncio.run(embedder.embed("chunk"))

    # Assert
    encode.assert_awaited_once_with("chunk")
    include.assert_any_call("test-model")
    include.assert_any_call("chunk")
    check.assert_called_once()
    assert result == [0.1, 0.2, 0.3]
