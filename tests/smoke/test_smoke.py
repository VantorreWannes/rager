"""Smoke tests: the package imports and its model-free core works.

These need no network access, model downloads, or mocking, so they finish in
seconds and make a cheap first tier to run: a failure here means the package
is broken at install or import time, not inside any particular model.
"""

import pytest

import rager
from rager.fusers import BordaCountFuser, ReciprocalRankFuser
from rager.indexes import DenseIndex, SparseIndex
from rager.stores import MemoryStore

pytestmark = pytest.mark.smoke


def test_public_api_exports_resolve() -> None:
    """Every name in ``__all__`` resolves to an attribute on the package."""
    missing = [name for name in rager.__all__ if not hasattr(rager, name)]
    assert missing == []


@pytest.mark.asyncio
async def test_dense_retrieval_round_trip() -> None:
    """A hand-made dense embedding is indexed, retrieved, and mapped to text."""
    # Arrange
    index = DenseIndex(3)
    chunks: MemoryStore[int, str] = MemoryStore()
    key = await index.add([1.0, 0.0, 0.0])
    chunks.add(key, "cats")
    chunks.add(await index.add([0.0, 1.0, 0.0]), "stocks")

    # Act
    (nearest,) = await index.similar([0.9, 0.1, 0.0], results=1)

    # Assert
    assert nearest == key
    assert chunks.get(nearest) == "cats"


@pytest.mark.asyncio
async def test_sparse_retrieval_round_trip() -> None:
    """A hand-made sparse embedding is indexed and retrieved by overlap."""
    # Arrange
    index = SparseIndex()
    key = await index.add({1: 1.0, 2: 0.5})
    await index.add({3: 1.0})

    # Act
    (nearest,) = await index.similar({1: 1.0}, results=1)

    # Assert
    assert nearest == key


def test_fusers_rank_unanimous_winner_first() -> None:
    """Both fusers put the value every ranking prefers at the top."""
    # Arrange
    rankings = ([1, 2, 3], [1, 3, 2])

    # Act & Assert
    assert ReciprocalRankFuser().fuse(*rankings)[0] == 1
    assert BordaCountFuser().fuse(*rankings)[0] == 1
