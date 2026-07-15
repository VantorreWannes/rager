"""Integration tests for fusers."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import blake3
import pytest

from rager.fusers import ReciprocalRankFuser
from rager.indexes import FaissIndex, SparseIndex
from rager.stores import MemoryStore

if TYPE_CHECKING:
    from rager.types import DenseEmbedding, Hash, SparseEmbedding

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class ChunkMetadata:
    """Sample metadata satisfying the Metadata protocol."""

    chunk: str
    file_id: Hash


def _chunks(store: MemoryStore[str, ChunkMetadata], keys: list[str]) -> list[str]:
    """Resolve index keys to their chunk texts."""
    return [metadata.chunk for key in keys if (metadata := store.get(key)) is not None]


@pytest.mark.asyncio
async def test_reciprocal_rank_fuser_fuses_dense_and_sparse_retrieval() -> None:
    """Chunks retrieved from dense and sparse indexes fuse into one ranking."""
    dense_index: FaissIndex[str, DenseEmbedding] = FaissIndex(3, MemoryStore())
    sparse_index: SparseIndex[str] = SparseIndex(MemoryStore(), MemoryStore())
    store: MemoryStore[str, ChunkMetadata] = MemoryStore()
    file_id = blake3.blake3(b"a file")
    embeddings: dict[str, tuple[DenseEmbedding, SparseEmbedding]] = {
        "apple": ([1.0, 0.0, 0.0], {1: 1.0}),
        "banana": ([0.0, 1.0, 0.0], {2: 1.0}),
    }
    for chunk, (dense, sparse) in embeddings.items():
        await dense_index.set(chunk, dense)
        await sparse_index.set(chunk, sparse)
        store.set(chunk, ChunkMetadata(chunk=chunk, file_id=file_id))
    fuser: ReciprocalRankFuser[str] = ReciprocalRankFuser()

    dense_keys = await dense_index.similar([0.9, 0.1, 0.0])
    sparse_keys = await sparse_index.similar({1: 0.5})
    fused = fuser.fuse(_chunks(store, dense_keys), _chunks(store, sparse_keys))

    assert fused == ["apple", "banana"]
