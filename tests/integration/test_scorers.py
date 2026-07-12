"""Integration tests for scorers."""

import asyncio

import pytest

from rager.scorers import CrossEncoderScorer

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_cross_encoder_scorer_ranks_relevant_chunk_higher() -> None:
    """score() gives a relevant chunk a higher score than an irrelevant one."""
    scorer = CrossEncoderScorer("cross-encoder/ms-marco-MiniLM-L6-v2")
    query = "How many people live in Berlin?"

    relevant, irrelevant = await asyncio.gather(
        scorer.score(query, "Berlin has a population of 3.7 million people."),
        scorer.score(query, "A cheetah can run at speeds of over 100 km/h."),
    )

    assert relevant > irrelevant
