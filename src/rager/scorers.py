"""Rerankers for scoring chunks based on relevance."""

from typing import Protocol


class Scorer(Protocol):
    """Protocol for chunk scorers."""

    async def score(self, query: str, chunk: str) -> float:
        """Score a chunk based on its semantic similarity to the query."""
        ...
