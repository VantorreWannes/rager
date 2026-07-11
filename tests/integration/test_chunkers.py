"""Integration tests for chunkers."""

import pytest

from rager.chunkers import SemanticChunker

pytestmark = pytest.mark.integration


def test_semantic_chunker_chunks() -> None:
    """Test that the SemanticChunker correctly chunks text."""
    chunker = SemanticChunker("gpt-3.5-turbo", 1000, 0)
    text = "This is a test document. It has multiple sentences to be split into chunks."

    result = chunker.chunk(text)
    assert len(result) == 1
