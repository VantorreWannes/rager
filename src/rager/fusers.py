"""Fusers for combining ranked lists of values."""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, override

import concresce

if TYPE_CHECKING:
    from collections.abc import Awaitable, Hashable

logger = logging.getLogger(__name__)


class Fuser[V](Protocol):
    """Protocol for fusing values into ranked lists."""

    def fuse(self, *values: list[V]) -> Awaitable[list[V]]:
        """Fuse multiple values into one ranked list."""
        ...


class BaseFuser[V: Hashable](ABC):
    """Abstract helper base deriving the ``Fuser`` protocol from one operation.

    Subclasses implement ``_batched``; ``fuse`` sums each value's weights
    across the rankings and sorts by total weight.
    """

    @abstractmethod
    def _batched(self, ranks: list[int], sizes: list[int]) -> Awaitable[list[float]]:
        """Return the weights of holding 1-based ``ranks`` in rankings of ``sizes``."""

    @abstractmethod
    def _weight(self, rank: int, size: int) -> Awaitable[float]:
        """Return the weight of holding 1-based ``rank`` in a ranking of ``size``."""

    async def _collect(self, rank: int, size: int) -> float:
        """Pool a weight query into the current batch."""
        collected = await concresce.collect((rank, size))
        ranks = [rank for rank, _ in collected]
        sizes = [size for _, size in collected]
        fused = await self._batched(ranks, sizes)
        return concresce.scatter(fused)

    async def fuse(self, *values: list[V]) -> list[V]:
        """Fuse multiple ranked lists into one list ranked by total weight."""
        logger.debug("Fusing %d rankings with %s", len(values), type(self).__name__)
        scores: dict[V, float] = {}
        for ranking in values:
            for rank, value in enumerate(ranking, start=1):
                weight = await self._weight(rank, len(ranking))
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
    @concresce.batch
    async def _weight(self, rank: int, size: int) -> float:
        """Return the weight of holding 1-based ``rank`` in a ranking of ``size``."""
        return await self._collect(rank, size)

    @override
    async def _batched(self, ranks: list[int], sizes: list[int]) -> list[float]:
        """Return the reciprocal rank weights for a batch of ranks and sizes."""
        return [1 / (self.k + rank) for rank in ranks]


class BordaCountFuser[V: Hashable](BaseFuser[V]):
    """Fuser that combines ranked lists by Borda count."""

    @override
    @concresce.batch
    async def _weight(self, rank: int, size: int) -> float:
        """Return the weight of holding 1-based ``rank`` in a ranking of ``size``."""
        return await self._collect(rank, size)

    @override
    async def _batched(self, ranks: list[int], sizes: list[int]) -> list[float]:
        """Return the Borda count weights for a batch of ranks and sizes."""
        return [size - rank + 1 for rank, size in zip(ranks, sizes, strict=True)]
