"""Unit tests for chunkers."""

from unittest.mock import MagicMock, patch

import pytest
from semantic_text_splitter import TextSplitter

from rager.chunkers import SemanticChunker

pytestmark = pytest.mark.unit


@patch("rager.chunkers.get_chunker")
def test_semantic_chunker_model(get_chunker: MagicMock) -> None:
    """Test that the model property returns the chunker."""
    # Arrange
    get_chunker.return_value = TextSplitter
    chunker = SemanticChunker("test-model", 1000, 0)

    # Act
    _ = chunker.model

    # Assert
    get_chunker.assert_called_once_with(
        "test-model",
        chunking_type="text",
        tree_sitter_language=None,
        max_tokens=1000,
        overlap=0,
        trim=True,
    )


@patch("rager.chunkers.belljar.check")
@patch("rager.chunkers.belljar.include")
@patch("rager.chunkers.SemanticChunker.model")
def test_semantic_chunker_chunks(
    model: MagicMock, include: MagicMock, check: MagicMock
) -> None:
    """chunks() returns the text of every element from units()."""
    # Arrange
    chunker = SemanticChunker("test-model", 1000, 0)
    text = "This is a test document. It has multiple sentences to be split into chunks."
    model.chunks.return_value = [
        "This is a test document.",
        "It has multiple sentences.",
    ]

    # Act
    _ = chunker.chunks(text)

    # Assert
    model.chunks.assert_called_once_with(text)
    include.assert_any_call(text)
    check.assert_called()
