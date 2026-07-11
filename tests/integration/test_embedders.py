"""Integration tests for chunkers."""

import pytest

from rager.embedders import SentenceTransformerDenseEmbedder

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_sentence_transformer_dense_embedder_embed() -> None:
    """Test that the SentenceTransformerDenseEmbedder correctly embeds text."""
    embedder = SentenceTransformerDenseEmbedder("all-MiniLM-L6-v2")
    text = "This is a test document."

    result = await embedder.embed(text)
    expected_length = 384
    assert len(result) == expected_length
