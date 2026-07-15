"""Fusers for combining ranked lists of values."""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, override

if TYPE_CHECKING:
    from collections.abc import Hashable

logger = logging.getLogger(__name__)


class Fuser[V](Protocol):
    """Protocol for fusing values into ranked lists."""

    def fuse(self, *values: list[V]) -> list[V]:
        """Fuse multiple values into one ranked list."""
        ...


class BaseFuser[V: Hashable](ABC):
    """Abstract helper base deriving the ``Fuser`` protocol from one operation.

    Subclasses implement ``_weight``; ``fuse`` sums each value's weights
    across the rankings and sorts by total weight.
    """

    @abstractmethod
    def _weight(self, rank: int, size: int, /) -> float:
        """Return the weight of holding 1-based ``rank`` in a ranking of ``size``."""

    def fuse(self, *values: list[V]) -> list[V]:
        """Fuse multiple ranked lists into one list ranked by total weight."""
        logger.debug("Fusing %d rankings with %s", len(values), type(self).__name__)
        scores: dict[V, float] = {}
        for ranking in values:
            for rank, value in enumerate(ranking, start=1):
                weight = self._weight(rank, len(ranking))
                scores[value] = scores.get(value, 0.0) + weight
        fused = sorted(scores, key=lambda value: scores[value], reverse=True)
        logger.debug("Fused rankings into %d distinct values", len(fused))
        return fused


class ReciprocalRankFuser[V: Hashable](BaseFuser[V]):
    """Fuser that combines ranked lists by reciprocal rank fusion."""

    def __init__(self, k: int = 60) -> None:
        """Initialize the fuser with the reciprocal rank smoothing constant."""
        self.k = k

    @override
    def _weight(self, rank: int, size: int, /) -> float:
        """Return the reciprocal rank weight, smoothed by the constant ``k``."""
        return 1 / (self.k + rank)


class BordaCountFuser[V: Hashable](BaseFuser[V]):
    """Fuser that combines ranked lists by Borda count."""

    @override
    def _weight(self, rank: int, size: int, /) -> float:
        """Return the Borda count: the number of values at or below this rank."""
        return size - rank + 1
