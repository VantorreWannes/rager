"""Integration tests for stores."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import blake3
import pytest

from rager.indexes import FaissIndex, SparseIndex
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
    index: FaissIndex[str, DenseEmbedding] = FaissIndex(3, MemoryStore())
    store: MemoryStore[str, DenseEmbedding] = MemoryStore()
    embedding = [1.0, 0.0, 0.0]
    index["chunk"] = embedding
    store.set("chunk", embedding)

    (key,) = await index.similar([0.9, 0.1, 0.0])

    assert store.get(key) == embedding


@pytest.mark.asyncio
async def test_memory_store_resolves_sparse_index_keys_to_embeddings() -> None:
    """Keys returned by a sparse index resolve back to stored embeddings."""
    index: SparseIndex[str] = SparseIndex(MemoryStore(), MemoryStore())
    store: MemoryStore[str, SparseEmbedding] = MemoryStore()
    embedding = {1: 1.0, 2: 0.5}
    index["chunk"] = embedding
    store.set("chunk", embedding)

    (key,) = await index.similar({1: 0.5})

    assert store.get(key) == embedding


@pytest.mark.asyncio
async def test_memory_store_resolves_dense_index_keys_to_chunks() -> None:
    """Keys returned by a dense index resolve back to stored chunk text."""
    index: FaissIndex[str, DenseEmbedding] = FaissIndex(3, MemoryStore())
    store: MemoryStore[str, str] = MemoryStore()
    index["chunk"] = [1.0, 0.0, 0.0]
    store.set("chunk", "a chunk")

    (key,) = await index.similar([0.9, 0.1, 0.0])

    assert store.get(key) == "a chunk"


@pytest.mark.asyncio
async def test_memory_store_resolves_dense_index_keys_to_metadata() -> None:
    """Keys returned by a dense index resolve back to stored metadata."""
    index: FaissIndex[str, DenseEmbedding] = FaissIndex(3, MemoryStore())
    store: MemoryStore[str, ChunkMetadata] = MemoryStore()
    metadata = ChunkMetadata(chunk="a chunk", file_id=blake3.blake3(b"a file"))
    index["chunk"] = [1.0, 0.0, 0.0]
    store.set("chunk", metadata)

    (key,) = await index.similar([0.9, 0.1, 0.0])

    assert store.get(key) is metadata
