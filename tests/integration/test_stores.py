"""Integration tests for stores."""

import pytest

from rager.indexes import DenseIndex, SparseIndex
from rager.stores import DenseEmbeddingStore, SparseEmbeddingStore

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_dense_embedding_store_resolves_dense_index_keys() -> None:
    """Keys returned by a dense index resolve back to stored embeddings."""
    index = DenseIndex(3)
    store = DenseEmbeddingStore()
    embedding = [1.0, 0.0, 0.0]
    store.add(await index.add(embedding), embedding)

    (key,) = await index.similar([0.9, 0.1, 0.0])

    assert store.get(key) == embedding


@pytest.mark.asyncio
async def test_sparse_embedding_store_resolves_sparse_index_keys() -> None:
    """Keys returned by a sparse index resolve back to stored embeddings."""
    index = SparseIndex()
    store = SparseEmbeddingStore()
    embedding = {1: 1.0, 2: 0.5}
    store.add(await index.add(embedding), embedding)

    (key,) = await index.similar({1: 0.5})

    assert store.get(key) == embedding
