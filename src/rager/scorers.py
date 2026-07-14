"""Rerankers for scoring chunks based on relevance."""

import logging
from functools import cached_property
from pathlib import Path
from typing import Protocol, cast

import concresce
from belljar import Jar
from sentence_transformers import CrossEncoder

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

    @concresce.batch
    async def _predict(self, query: str, chunk: str) -> float:
        """Score a query-chunk pair with the cross-encoder."""
        pairs = await concresce.collect((query, chunk))
        logger.debug(
            "Scoring batch of %d query-chunk pairs with %r",
            len(pairs),
            self.model_name,
        )
        scores = self.model.predict(pairs).tolist()
        return cast("float", scores)

    async def score(self, query: str, chunk: str) -> float:
        """Score a chunk based on its semantic similarity to the query."""
        jar = Jar[float](Path(".jar/scorers"))
        jar.include(self.score.__code__)
        jar.include(self.model_name)
        jar.include(query)
        jar.include(chunk)
        if (cached := jar.get()) is not None:
            return cached
        logger.debug("Cache miss; scoring chunk of %d characters", len(chunk))
        return jar.set(await self._predict(query, chunk))
