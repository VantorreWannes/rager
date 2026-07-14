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


@patch("rager.chunkers.Jar")
@patch("rager.chunkers.SemanticChunker.model")
def test_semantic_chunker_chunks(model: MagicMock, jar_cls: MagicMock) -> None:
    """chunks() folds its identity into the jar and seals the chunks."""
    # Arrange
    chunker = SemanticChunker("test-model", 1000, 0)
    text = "This is a test document. It has multiple sentences to be split into chunks."
    chunks = ["This is a test document.", "It has multiple sentences."]
    model.chunks.return_value = chunks
    jar = jar_cls[list[str]].return_value
    jar.get.return_value = None
    jar.set.side_effect = lambda value: value

    # Act
    result = chunker.chunks(text)

    # Assert
    model.chunks.assert_called_once_with(text)
    jar.include.assert_any_call("test-model")
    jar.include.assert_any_call(text)
    jar.set.assert_called_once_with(chunks)
    assert result == chunks


@patch("rager.chunkers.Jar")
@patch("rager.chunkers.SemanticChunker.model")
def test_semantic_chunker_chunks_cached(model: MagicMock, jar_cls: MagicMock) -> None:
    """chunks() returns the sealed chunks without chunking on a cache hit."""
    # Arrange
    chunker = SemanticChunker("test-model", 1000, 0)
    cached = ["This is a test document."]
    jar = jar_cls[list[str]].return_value
    jar.get.return_value = cached

    # Act
    result = chunker.chunks("This is a test document.")

    # Assert
    model.chunks.assert_not_called()
    jar.set.assert_not_called()
    assert result == cached
