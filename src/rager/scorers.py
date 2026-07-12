"""Rerankers for scoring chunks based on relevance."""

import logging
from datetime import timedelta
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import belljar
import concresce
from sentence_transformers import CrossEncoder

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


class Scorer(Protocol):
    """Protocol for chunk scorers."""

    async def score(self, query: str, chunk: str) -> float:
        """Score a chunk based on its semantic similarity to the query."""
        ...


class CrossEncoderScorer:
    """Scorer that reranks chunks with a sentence-transformers cross-encoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2") -> None:
        """Initialize the scorer with a specific model."""
        self.model_name = model_name

    @cached_property
    def model(self) -> CrossEncoder:
        """Load the CrossEncoder model."""
        logger.info("Loading CrossEncoder model %r", self.model_name)
        return CrossEncoder(self.model_name)

    @cached_property
    def _predict(self) -> Callable[[str, str], Awaitable[float]]:
        """Coalesce concurrent calls into per-instance prediction batches."""
        return concresce.batch(window=timedelta(milliseconds=1))(self._predict_batch)

    async def _predict_batch(self, query: str, chunk: str) -> float:
        """Score a query-chunk pair with the cross-encoder."""
        pairs = await concresce.collect((query, chunk))
        logger.debug(
            "Scoring batch of %d query-chunk pairs with %r",
            len(pairs),
            self.model_name,
        )
        scores = self.model.predict(pairs).tolist()
        return concresce.scatter(scores)

    @belljar.store(Path(".jar/scorers"))
    async def score(self, query: str, chunk: str) -> float:
        """Score a chunk based on its semantic similarity to the query."""
        belljar.include(self.model_name)
        belljar.include(query)
        belljar.include(chunk)
        belljar.check()
        logger.debug("Cache miss; scoring chunk of %d characters", len(chunk))
        return await self._predict(query, chunk)
