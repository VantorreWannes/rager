"""Integration tests for fusers."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import blake3
import pytest

from rager.fusers import ReciprocalRankFuser
from rager.indexes import DenseIndex, SparseIndex
from rager.stores import MetadataStore

if TYPE_CHECKING:
    from rager.types import Hash

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class ChunkMetadata:
    """Sample metadata satisfying the Metadata protocol."""

    chunk: str
    file_id: Hash


def _chunks(store: MetadataStore[ChunkMetadata], keys: list[int]) -> list[str]:
    """Resolve index keys to their chunk texts."""
    return [metadata.chunk for key in keys if (metadata := store.get(key)) is not None]


@pytest.mark.asyncio
async def test_reciprocal_rank_fuser_fuses_dense_and_sparse_retrieval() -> None:
    """Chunks retrieved from dense and sparse indexes fuse into one ranking."""
    dense_index = DenseIndex(3)
    sparse_index = SparseIndex()
    store: MetadataStore[ChunkMetadata] = MetadataStore()
    file_id = blake3.blake3(b"a file")
    embeddings = {
        "apple": ([1.0, 0.0, 0.0], {1: 1.0}),
        "banana": ([0.0, 1.0, 0.0], {2: 1.0}),
    }
    for chunk, (dense, sparse) in embeddings.items():
        metadata = ChunkMetadata(chunk=chunk, file_id=file_id)
        store.add(await dense_index.add(dense), metadata)
        store.add(await sparse_index.add(sparse), metadata)
    fuser: ReciprocalRankFuser[str] = ReciprocalRankFuser()

    dense_keys = await dense_index.similar([0.9, 0.1, 0.0])
    sparse_keys = await sparse_index.similar({1: 0.5})
    fused = fuser.fuse(_chunks(store, dense_keys), _chunks(store, sparse_keys))

    assert fused == ["apple", "banana"]
