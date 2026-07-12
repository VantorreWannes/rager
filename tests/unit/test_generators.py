"""Unit tests for generators."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rager.generators import TransformersGenerator

pytestmark = pytest.mark.unit


@patch("rager.generators.pipeline")
def test_transformers_generator_model(pipeline: MagicMock) -> None:
    """Test that the model property returns the text-generation pipeline."""
    # Arrange
    generator = TransformersGenerator("test-model")

    # Act
    _ = generator.model

    # Assert
    pipeline.assert_called_once_with("text-generation", model="test-model")


@patch("rager.generators.TransformersGenerator.model")
@pytest.mark.asyncio
async def test_transformers_generator_generate(model: MagicMock) -> None:
    """_generate() answers the collected batch and returns this caller's answer."""
    # Arrange
    generator = TransformersGenerator("test-model")
    expected_answer = "Paris is the capital of France."
    chat = [
        {"role": "user", "content": "query"},
        {"role": "assistant", "content": expected_answer},
    ]
    model.return_value = [[{"generated_text": chat}]]

    # Act
    result = await generator._generate("query")

    # Assert
    model.assert_called_once_with(
        [[{"role": "user", "content": "query"}]],
        max_new_tokens=generator.max_new_tokens,
        do_sample=False,
    )
    assert result == expected_answer


@patch("rager.generators.pipeline")
@pytest.mark.asyncio
async def test_transformers_generator_does_not_batch_across_instances(
    pipeline: MagicMock,
) -> None:
    """Concurrent _generate() calls on different generators batch separately."""
    # Arrange
    first = TransformersGenerator("model-a")
    second = TransformersGenerator("model-b")
    model = pipeline.return_value
    chat = [{"role": "assistant", "content": "answer"}]
    model.return_value = [[{"generated_text": chat}]]

    # Act
    await asyncio.gather(first._generate("one"), second._generate("two"))

    # Assert
    expected_calls = 2
    assert model.call_count == expected_calls


@patch.object(TransformersGenerator, "_generate", new_callable=AsyncMock)
@patch("rager.generators.belljar.check")
@patch("rager.generators.belljar.include")
@pytest.mark.asyncio
async def test_transformers_generator_prompt(
    include: MagicMock, check: MagicMock, generate: AsyncMock
) -> None:
    """prompt() folds its identity into belljar and delegates to _generate."""
    # Arrange
    generator = TransformersGenerator("test-model")
    expected_answer = "answer"
    generate.return_value = expected_answer

    # Act
    result = await generator.prompt("query")

    # Assert
    generate.assert_awaited_once_with("query")
    include.assert_any_call("test-model")
    include.assert_any_call(generator.max_new_tokens)
    include.assert_any_call("query")
    check.assert_called_once()
    assert result == expected_answer
