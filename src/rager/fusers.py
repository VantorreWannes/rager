"""Fusers for combining ranked lists of values."""

import logging
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Hashable

logger = logging.getLogger(__name__)


class Fuser[V](Protocol):
    """Protocol for fusing values into ranked lists."""

    def fuse(self, *values: list[V]) -> list[V]:
        """Fuse multiple values into one ranked list."""
        ...


class ReciprocalRankFuser[V: Hashable]:
    """Fuser that combines ranked lists by reciprocal rank fusion."""

    def __init__(self, k: int = 60) -> None:
        """Initialize the fuser with the reciprocal rank smoothing constant."""
        self.k = k

    def fuse(self, *values: list[V]) -> list[V]:
        """Fuse multiple ranked lists into one list ranked by reciprocal rank."""
        logger.debug(
            "Fusing %d rankings by reciprocal rank (k=%d)", len(values), self.k
        )
        scores: dict[V, float] = {}
        for ranking in values:
            for rank, value in enumerate(ranking, start=1):
                scores[value] = scores.get(value, 0.0) + 1 / (self.k + rank)
        fused = sorted(scores, key=lambda value: scores[value], reverse=True)
        logger.debug("Fused rankings into %d distinct values", len(fused))
        return fused


class BordaCountFuser[V: Hashable]:
    """Fuser that combines ranked lists by Borda count."""

    def fuse(self, *values: list[V]) -> list[V]:
        """Fuse multiple ranked lists into one list ranked by Borda count."""
        logger.debug("Fusing %d rankings by Borda count", len(values))
        scores: dict[V, int] = {}
        for ranking in values:
            for rank, value in enumerate(ranking):
                scores[value] = scores.get(value, 0) + len(ranking) - rank
        fused = sorted(scores, key=lambda value: scores[value], reverse=True)
        logger.debug("Fused rankings into %d distinct values", len(fused))
        return fused
