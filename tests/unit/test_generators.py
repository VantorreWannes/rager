"""Unit tests for generators."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rager.generators import BaseGenerator, TransformersGenerator

pytestmark = pytest.mark.unit


def test_base_generator_is_abstract() -> None:
    """BaseGenerator cannot be instantiated without the jar and generate operations."""
    # Act & Assert
    with pytest.raises(TypeError):
        BaseGenerator()  # type: ignore[abstract]


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
        do_sample=True,
    )
    assert result == expected_answer


@patch("rager.generators.pipeline")
@pytest.mark.asyncio
async def test_transformers_generator_coalesces_across_instances(
    pipeline: MagicMock,
) -> None:
    """Concurrent _generate() calls on different generators share one call.

    concresce 0.2 coalesces batches per event-loop turn rather than per
    instance. Callers must not mix instances of the same batch-owning class
    in concurrent calls; this documents the resulting shared-batch behavior.
    """
    # Arrange
    first = TransformersGenerator("model-a")
    second = TransformersGenerator("model-b")
    model = pipeline.return_value
    first_chat = [{"role": "assistant", "content": "first answer"}]
    second_chat = [{"role": "assistant", "content": "second answer"}]
    model.return_value = [
        [{"generated_text": first_chat}],
        [{"generated_text": second_chat}],
    ]

    # Act
    first_result, second_result = await asyncio.gather(
        first._generate("one"), second._generate("two")
    )

    # Assert
    model.assert_called_once_with(
        [
            [{"role": "user", "content": "one"}],
            [{"role": "user", "content": "two"}],
        ],
        max_new_tokens=first.max_new_tokens,
        do_sample=True,
    )
    assert first_result == "first answer"
    assert second_result == "second answer"


@patch.object(TransformersGenerator, "_generate", new_callable=AsyncMock)
@patch("rager.generators.Jar")
@pytest.mark.asyncio
async def test_transformers_generator_prompt(
    jar_cls: MagicMock, generate: AsyncMock
) -> None:
    """prompt() folds its identity into the jar and seals the answer."""
    # Arrange
    generator = TransformersGenerator("test-model")
    expected_answer = "answer"
    generate.return_value = expected_answer
    jar = jar_cls[str].return_value
    jar.get.return_value = None
    jar.set.side_effect = lambda value: value

    # Act
    result = await generator.prompt("query")

    # Assert
    generate.assert_awaited_once_with("query")
    jar.include.assert_any_call("test-model")
    jar.include.assert_any_call(generator.max_new_tokens)
    jar.include.assert_any_call("query")
    jar.set.assert_called_once_with(expected_answer)
    assert result == expected_answer


@patch.object(TransformersGenerator, "_generate", new_callable=AsyncMock)
@patch("rager.generators.Jar")
@pytest.mark.asyncio
async def test_transformers_generator_prompt_cached(
    jar_cls: MagicMock, generate: AsyncMock
) -> None:
    """prompt() returns the sealed answer without generating on a cache hit."""
    # Arrange
    generator = TransformersGenerator("test-model")
    expected_answer = "answer"
    jar = jar_cls[str].return_value
    jar.get.return_value = expected_answer

    # Act
    result = await generator.prompt("query")

    # Assert
    generate.assert_not_awaited()
    jar.set.assert_not_called()
    assert result == expected_answer
