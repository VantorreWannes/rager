"""Integration tests for generators."""

import pytest

from rager.generators import TransformersGenerator

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_transformers_generator_answers_query() -> None:
    """prompt() generates a relevant answer for a simple query."""
    generator = TransformersGenerator(
        "HuggingFaceTB/SmolLM2-135M-Instruct", max_new_tokens=32
    )

    answer = await generator.prompt("What is the capital of France?")

    assert "Paris" in answer
