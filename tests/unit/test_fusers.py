"""Unit tests for fusers."""

import pytest

from rager.fusers import BaseFuser, BordaCountFuser, ReciprocalRankFuser

pytestmark = pytest.mark.unit


def test_base_fuser_is_abstract() -> None:
    """BaseFuser cannot be instantiated without the weight operation."""
    # Act & Assert
    with pytest.raises(TypeError):
        BaseFuser()  # type: ignore[abstract]


def test_reciprocal_rank_fuser_of_no_rankings_returns_empty_list() -> None:
    """fuse() without rankings returns an empty list."""
    # Arrange
    fuser: ReciprocalRankFuser[str] = ReciprocalRankFuser()

    # Act & Assert
    assert fuser.fuse() == []


def test_reciprocal_rank_fuser_preserves_single_ranking() -> None:
    """fuse() of a single ranking returns it unchanged."""
    # Arrange
    fuser: ReciprocalRankFuser[str] = ReciprocalRankFuser()

    # Act
    fused = fuser.fuse(["a", "b", "c"])

    # Assert
    assert fused == ["a", "b", "c"]


def test_reciprocal_rank_fuser_ranks_consensus_values_first() -> None:
    """fuse() ranks a value found in every ranking above single-ranking ones."""
    # Arrange
    fuser: ReciprocalRankFuser[str] = ReciprocalRankFuser()

    # Act
    fused = fuser.fuse(["a", "b"], ["c", "b"])

    # Assert
    assert fused[0] == "b"


def test_reciprocal_rank_fuser_deduplicates_values() -> None:
    """fuse() returns each value once even when rankings overlap."""
    # Arrange
    fuser: ReciprocalRankFuser[str] = ReciprocalRankFuser()

    # Act
    fused = fuser.fuse(["a", "b"], ["b", "a"])

    # Assert
    assert sorted(fused) == ["a", "b"]


def test_reciprocal_rank_fuser_breaks_ties_by_first_seen() -> None:
    """fuse() keeps first-seen order for values with equal scores."""
    # Arrange
    fuser: ReciprocalRankFuser[str] = ReciprocalRankFuser()

    # Act
    fused = fuser.fuse(["a"], ["b"])

    # Assert
    assert fused == ["a", "b"]


def test_borda_count_fuser_of_no_rankings_returns_empty_list() -> None:
    """fuse() without rankings returns an empty list."""
    # Arrange
    fuser: BordaCountFuser[str] = BordaCountFuser()

    # Act & Assert
    assert fuser.fuse() == []


def test_borda_count_fuser_preserves_single_ranking() -> None:
    """fuse() of a single ranking returns it unchanged."""
    # Arrange
    fuser: BordaCountFuser[str] = BordaCountFuser()

    # Act
    fused = fuser.fuse(["a", "b", "c"])

    # Assert
    assert fused == ["a", "b", "c"]


def test_borda_count_fuser_ranks_consensus_values_first() -> None:
    """fuse() ranks a value found in every ranking above single-ranking ones."""
    # Arrange
    fuser: BordaCountFuser[str] = BordaCountFuser()

    # Act
    fused = fuser.fuse(["a", "b", "c"], ["d", "b", "e"])

    # Assert
    assert fused[0] == "b"


def test_borda_count_fuser_deduplicates_values() -> None:
    """fuse() returns each value once even when rankings overlap."""
    # Arrange
    fuser: BordaCountFuser[str] = BordaCountFuser()

    # Act
    fused = fuser.fuse(["a", "b"], ["b", "a"])

    # Assert
    assert sorted(fused) == ["a", "b"]
