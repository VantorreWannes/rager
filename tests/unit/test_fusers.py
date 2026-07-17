"""Unit tests for fusers."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from rager.fusers import BaseFuser, BordaCountFuser, ReciprocalRankFuser

pytestmark = pytest.mark.unit


def test_base_fuser_is_abstract() -> None:
    """BaseFuser cannot be instantiated without the batch operation."""
    # Act & Assert
    with pytest.raises(TypeError):
        BaseFuser()  # type: ignore[abstract]


@pytest.mark.asyncio
async def test_reciprocal_rank_fuser_batched() -> None:
    """_batched() returns the reciprocal rank weight for each rank."""
    # Arrange
    fuser = ReciprocalRankFuser(k=1)

    # Act
    result = await fuser._batched([1, 2], [3, 3])

    # Assert
    assert result == [1 / 2, 1 / 3]


@pytest.mark.asyncio
async def test_reciprocal_rank_fuser_weight_batches_concurrent_calls() -> None:
    """Concurrent _weight() calls for different ranks share one _batched() call."""
    # Arrange
    fuser = ReciprocalRankFuser()
    first_weight, second_weight = 1.0, 2.0
    with patch.object(
        ReciprocalRankFuser, "_batched", new_callable=AsyncMock
    ) as batched:
        batched.return_value = [first_weight, second_weight]

        # Act
        first_result, second_result = await asyncio.gather(
            fuser._weight(1, 3), fuser._weight(2, 3)
        )

    # Assert
    batched.assert_awaited_once_with([1, 2], [3, 3])
    assert first_result == first_weight
    assert second_result == second_weight


@pytest.mark.asyncio
async def test_reciprocal_rank_fuser_coalesces_across_instances() -> None:
    """Concurrent _weight() calls on different fusers share one _batched() call.

    concresce 0.2 coalesces batches per event-loop turn rather than per
    instance. Callers must not mix instances of the same batch-owning class
    in concurrent calls; this documents the resulting shared-batch behavior.
    """
    # Arrange
    first, second = ReciprocalRankFuser(k=1), ReciprocalRankFuser(k=2)
    first_weight, second_weight = 1.0, 2.0
    with patch.object(
        ReciprocalRankFuser, "_batched", new_callable=AsyncMock
    ) as batched:
        batched.return_value = [first_weight, second_weight]

        # Act
        first_result, second_result = await asyncio.gather(
            first._weight(1, 3), second._weight(2, 3)
        )

    # Assert
    batched.assert_awaited_once_with([1, 2], [3, 3])
    assert first_result == first_weight
    assert second_result == second_weight


@pytest.mark.asyncio
async def test_reciprocal_rank_fuser_of_no_rankings_returns_empty_list() -> None:
    """fuse() without rankings returns an empty list."""
    # Arrange
    fuser: ReciprocalRankFuser[str] = ReciprocalRankFuser()

    # Act & Assert
    assert await fuser.fuse() == []


@pytest.mark.asyncio
async def test_reciprocal_rank_fuser_preserves_single_ranking() -> None:
    """fuse() of a single ranking returns it unchanged."""
    # Arrange
    fuser: ReciprocalRankFuser[str] = ReciprocalRankFuser()

    # Act
    fused = await fuser.fuse(["a", "b", "c"])

    # Assert
    assert fused == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_reciprocal_rank_fuser_ranks_consensus_values_first() -> None:
    """fuse() ranks a value found in every ranking above single-ranking ones."""
    # Arrange
    fuser: ReciprocalRankFuser[str] = ReciprocalRankFuser()

    # Act
    fused = await fuser.fuse(["a", "b"], ["c", "b"])

    # Assert
    assert fused[0] == "b"


@pytest.mark.asyncio
async def test_reciprocal_rank_fuser_deduplicates_values() -> None:
    """fuse() returns each value once even when rankings overlap."""
    # Arrange
    fuser: ReciprocalRankFuser[str] = ReciprocalRankFuser()

    # Act
    fused = await fuser.fuse(["a", "b"], ["b", "a"])

    # Assert
    assert sorted(fused) == ["a", "b"]


@pytest.mark.asyncio
async def test_reciprocal_rank_fuser_breaks_ties_by_first_seen() -> None:
    """fuse() keeps first-seen order for values with equal scores."""
    # Arrange
    fuser: ReciprocalRankFuser[str] = ReciprocalRankFuser()

    # Act
    fused = await fuser.fuse(["a"], ["b"])

    # Assert
    assert fused == ["a", "b"]


@pytest.mark.asyncio
async def test_borda_count_fuser_batched() -> None:
    """_batched() returns the Borda count weight for each rank and size."""
    # Arrange
    fuser = BordaCountFuser()

    # Act
    result = await fuser._batched([1, 2], [3, 3])

    # Assert
    assert result == [3, 2]


@pytest.mark.asyncio
async def test_borda_count_fuser_of_no_rankings_returns_empty_list() -> None:
    """fuse() without rankings returns an empty list."""
    # Arrange
    fuser: BordaCountFuser[str] = BordaCountFuser()

    # Act & Assert
    assert await fuser.fuse() == []


@pytest.mark.asyncio
async def test_borda_count_fuser_preserves_single_ranking() -> None:
    """fuse() of a single ranking returns it unchanged."""
    # Arrange
    fuser: BordaCountFuser[str] = BordaCountFuser()

    # Act
    fused = await fuser.fuse(["a", "b", "c"])

    # Assert
    assert fused == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_borda_count_fuser_ranks_consensus_values_first() -> None:
    """fuse() ranks a value found in every ranking above single-ranking ones."""
    # Arrange
    fuser: BordaCountFuser[str] = BordaCountFuser()

    # Act
    fused = await fuser.fuse(["a", "b", "c"], ["d", "b", "e"])

    # Assert
    assert fused[0] == "b"


@pytest.mark.asyncio
async def test_borda_count_fuser_deduplicates_values() -> None:
    """fuse() returns each value once even when rankings overlap."""
    # Arrange
    fuser: BordaCountFuser[str] = BordaCountFuser()

    # Act
    fused = await fuser.fuse(["a", "b"], ["b", "a"])

    # Assert
    assert sorted(fused) == ["a", "b"]
