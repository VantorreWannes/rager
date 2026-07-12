"""Integration tests for embedders."""

import asyncio

import pytest

from rager.embedders import SentenceTransformerDenseEmbedder, SpladeSparseEmbedder

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_sentence_transformer_dense_embedder_embed() -> None:
    """Test that the SentenceTransformerDenseEmbedder correctly embeds text."""
    embedder = SentenceTransformerDenseEmbedder("all-MiniLM-L6-v2")
    text = "This is a test document."

    result = await embedder.embed(text)
    expected_length = 384
    assert len(result) == expected_length


@pytest.mark.asyncio
async def test_splade_sparse_embedder_embed() -> None:
    """Test that the SpladeSparseEmbedder returns a non-empty weight map."""
    embedder = SpladeSparseEmbedder("prithivida/Splade_PP_en_v1")
    text = "This is a test document."

    result = await embedder.embed(text)
    expected_length = 19
    assert len(result) == expected_length


@pytest.mark.asyncio
async def test_splade_sparse_embedder_embeds_concurrent_chunks() -> None:
    """Concurrent calls are batched, yet each caller gets its own weight map."""
    embedder = SpladeSparseEmbedder("prithivida/Splade_PP_en_v1")

    cats, physics = await asyncio.gather(
        embedder.embed("cats and dogs"),
        embedder.embed("quantum physics equations"),
    )
    assert cats
    assert physics
    assert cats != physics
