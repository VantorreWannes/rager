"""Rerankers for scoring chunks based on relevance."""

import logging
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, override

import concresce
from belljar import Jar
from sentence_transformers import CrossEncoder

if TYPE_CHECKING:
    from collections.abc import Awaitable

logger = logging.getLogger(__name__)


class Scorer(Protocol):
    """Protocol for chunk scorers."""

    def score(self, query: str, chunk: str) -> Awaitable[float]:
        """Score a chunk based on its semantic similarity to the query."""
        ...


class BaseScorer(ABC):
    """Abstract helper base deriving the ``Scorer`` protocol from two operations.

    Subclasses implement ``_jar`` and ``_predict``; ``score`` seals every
    predicted score in the jar and returns it from there on later calls.
    """

    @abstractmethod
    def _jar(self, query: str, chunk: str) -> Jar[float]:
        """Open a jar positioned at the identity of the given query-chunk pair."""

    @abstractmethod
    def _predict(self, query: str, chunk: str) -> Awaitable[float]:
        """Score a query-chunk pair."""

    async def score(self, query: str, chunk: str) -> float:
        """Score a chunk based on its semantic similarity to the query."""
        jar = self._jar(query, chunk)
        if (cached := jar.get()) is not None:
            return cached
        logger.debug("Cache miss; scoring chunk of %d characters", len(chunk))
        return jar.set(await self._predict(query, chunk))


class CrossEncoderScorer(BaseScorer):
    """Scorer that reranks chunks with a sentence-transformers cross-encoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2") -> None:
        """Initialize the scorer with a specific model."""
        self.model_name = model_name

    @cached_property
    def model(self) -> CrossEncoder:
        """Load the CrossEncoder model."""
        logger.info("Loading CrossEncoder model %r", self.model_name)
        return CrossEncoder(self.model_name)

    @override
    def _jar(self, query: str, chunk: str) -> Jar[float]:
        """Open a jar positioned at the identity of the given query-chunk pair."""
        jar = Jar[float](Path(".jar/scorers"))
        jar.include(self._jar.__code__)
        jar.include(self.model_name)
        jar.include(query)
        jar.include(chunk)
        return jar

    @concresce.batch
    @override
    async def _predict(self, query: str, chunk: str) -> float:
        """Score a query-chunk pair with the cross-encoder."""
        pairs = await concresce.collect((query, chunk))
        logger.debug(
            "Scoring batch of %d query-chunk pairs with %r",
            len(pairs),
            self.model_name,
        )
        scores = self.model.predict(pairs).tolist()
        return concresce.scatter(scores)
