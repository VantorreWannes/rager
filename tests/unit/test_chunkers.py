"""Unit tests for chunkers."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from semantic_text_splitter import TextSplitter

from rager.chunkers import BaseChunker, SemanticChunker

pytestmark = pytest.mark.unit


def test_base_chunker_is_abstract() -> None:
    """BaseChunker cannot be instantiated without the jar and batch operations."""
    # Act & Assert
    with pytest.raises(TypeError):
        BaseChunker()  # type: ignore[abstract]


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


@patch("rager.chunkers.SemanticChunker.model")
@pytest.mark.asyncio
async def test_semantic_chunker_batched(model: MagicMock) -> None:
    """_batched() splits the collected batch with a single chunk_all() call."""
    # Arrange
    chunker = SemanticChunker("test-model", 1000, 0)
    model.chunk_all.return_value = [["first chunk"], ["second chunk", "third chunk"]]

    # Act
    result = await chunker._batched(["first unit", "second unit"])

    # Assert
    model.chunk_all.assert_called_once_with(["first unit", "second unit"])
    assert result == [("first chunk",), ("second chunk", "third chunk")]


@patch("rager.chunkers.Jar")
@patch("rager.chunkers.SemanticChunker.model")
@pytest.mark.asyncio
async def test_semantic_chunker_chunks(model: MagicMock, jar_cls: MagicMock) -> None:
    """chunks() folds its identity into the jar and seals the chunks."""
    # Arrange
    chunker = SemanticChunker("test-model", 1000, 0)
    text = "This is a test document. It has multiple sentences to be split into chunks."
    chunks = ["This is a test document.", "It has multiple sentences."]
    model.chunk_all.return_value = [chunks]
    jar = jar_cls[tuple].return_value
    jar.get.return_value = None
    jar.set.side_effect = lambda value: value

    # Act
    result = await chunker.chunks(text)

    # Assert
    model.chunk_all.assert_called_once_with([text])
    jar.include.assert_any_call(chunker._jar.__code__)
    jar.include.assert_any_call(chunker._batched.__code__)
    jar.include.assert_any_call(chunker.chunks.__code__)
    jar.include.assert_any_call("test-model")
    jar.include.assert_any_call(1000)
    jar.include.assert_any_call(0)
    jar.include.assert_any_call(text)
    jar.set.assert_called_once_with(tuple(chunks))
    assert result == tuple(chunks)


@patch("rager.chunkers.Jar")
@patch("rager.chunkers.SemanticChunker.model")
@pytest.mark.asyncio
async def test_semantic_chunker_chunks_cached(
    model: MagicMock, jar_cls: MagicMock
) -> None:
    """chunks() returns the sealed chunks without chunking on a cache hit."""
    # Arrange
    chunker = SemanticChunker("test-model", 1000, 0)
    cached = ("This is a test document.",)
    jar = jar_cls[tuple].return_value
    jar.get.return_value = cached

    # Act
    result = await chunker.chunks("This is a test document.")

    # Assert
    model.chunk_all.assert_not_called()
    jar.set.assert_not_called()
    assert result is cached


@patch.object(SemanticChunker, "_batched", new_callable=AsyncMock)
@patch("rager.chunkers.Jar")
@pytest.mark.asyncio
async def test_semantic_chunker_batches_concurrent_chunks_calls(
    jar_cls: MagicMock, batched: AsyncMock
) -> None:
    """Concurrent chunks() calls for different units share one _batched() call."""
    # Arrange
    chunker = SemanticChunker("test-model", 1000, 0)
    first_chunks, second_chunks = ("first chunk",), ("second chunk",)
    batched.return_value = [first_chunks, second_chunks]
    jar = jar_cls[tuple].return_value
    jar.get.return_value = None
    jar.set.side_effect = lambda value: value

    # Act
    first_result, second_result = await asyncio.gather(
        chunker.chunks("first unit"), chunker.chunks("second unit")
    )

    # Assert
    batched.assert_awaited_once_with(["first unit", "second unit"])
    assert first_result == first_chunks
    assert second_result == second_chunks
