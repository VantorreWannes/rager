"""Integration tests for stores."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import blake3
import pytest

from rager.indexes import MemoryDenseIndex, MemorySparseIndex
from rager.stores import MemoryStore

if TYPE_CHECKING:
    from rager.types import DenseEmbedding, Hash, SparseEmbedding

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class ChunkMetadata:
    """Sample metadata stored as a value."""

    chunk: str
    file_id: Hash


@pytest.mark.asyncio
async def test_memory_store_resolves_dense_index_keys_to_embeddings() -> None:
    """Keys returned by a dense index resolve back to stored embeddings."""
    index = MemoryDenseIndex(3)
    store: MemoryStore[int, DenseEmbedding] = MemoryStore()
    embedding = [1.0, 0.0, 0.0]
    store.set(await index.add(embedding), embedding)

    (key,) = await index.similar([0.9, 0.1, 0.0])

    assert store.get(key) == embedding


@pytest.mark.asyncio
async def test_memory_store_resolves_sparse_index_keys_to_embeddings() -> None:
    """Keys returned by a sparse index resolve back to stored embeddings."""
    index = MemorySparseIndex()
    store: MemoryStore[int, SparseEmbedding] = MemoryStore()
    embedding = {1: 1.0, 2: 0.5}
    store.set(await index.add(embedding), embedding)

    (key,) = await index.similar({1: 0.5})

    assert store.get(key) == embedding


@pytest.mark.asyncio
async def test_memory_store_resolves_dense_index_keys_to_chunks() -> None:
    """Keys returned by a dense index resolve back to stored chunk text."""
    index = MemoryDenseIndex(3)
    store: MemoryStore[int, str] = MemoryStore()
    store.set(await index.add([1.0, 0.0, 0.0]), "a chunk")

    (key,) = await index.similar([0.9, 0.1, 0.0])

    assert store.get(key) == "a chunk"


@pytest.mark.asyncio
async def test_memory_store_resolves_dense_index_keys_to_metadata() -> None:
    """Keys returned by a dense index resolve back to stored metadata."""
    index = MemoryDenseIndex(3)
    store: MemoryStore[int, ChunkMetadata] = MemoryStore()
    metadata = ChunkMetadata(chunk="a chunk", file_id=blake3.blake3(b"a file"))
    store.set(await index.add([1.0, 0.0, 0.0]), metadata)

    (key,) = await index.similar([0.9, 0.1, 0.0])

    assert store.get(key) is metadata
