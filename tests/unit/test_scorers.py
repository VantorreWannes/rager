"""Unit tests for scorers."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rager.scorers import BaseScorer, CrossEncoderScorer

pytestmark = pytest.mark.unit


def test_base_scorer_is_abstract() -> None:
    """BaseScorer cannot be instantiated without the jar and predict operations."""
    # Act & Assert
    with pytest.raises(TypeError):
        BaseScorer()  # type: ignore[abstract]


@patch("rager.scorers.CrossEncoder")
def test_cross_encoder_scorer_model(cross_encoder: MagicMock) -> None:
    """Test that the model property returns the cross-encoder."""
    # Arrange
    scorer = CrossEncoderScorer("test-model")

    # Act
    _ = scorer.model

    # Assert
    cross_encoder.assert_called_once_with("test-model")


@patch("rager.scorers.CrossEncoderScorer.model")
@pytest.mark.asyncio
async def test_cross_encoder_scorer_predict(model: MagicMock) -> None:
    """_predict() scores the collected batch and returns this caller's score."""
    # Arrange
    scorer = CrossEncoderScorer("test-model")
    expected_score = 0.9
    model.predict.return_value.tolist.return_value = [expected_score]

    # Act
    result = await scorer._predict("query", "chunk")

    # Assert
    model.predict.assert_called_once_with([("query", "chunk")])
    assert result == expected_score


@patch("rager.scorers.CrossEncoder")
@pytest.mark.asyncio
async def test_cross_encoder_scorer_does_not_batch_across_instances(
    cross_encoder: MagicMock,
) -> None:
    """Concurrent _predict() calls on different scorers batch separately."""
    # Arrange
    first = CrossEncoderScorer("model-a")
    second = CrossEncoderScorer("model-b")
    model = cross_encoder.return_value
    model.predict.return_value.tolist.return_value = [0.9]

    # Act
    await asyncio.gather(
        first._predict("query", "one"), second._predict("query", "two")
    )

    # Assert
    expected_calls = 2
    assert model.predict.call_count == expected_calls


@patch.object(CrossEncoderScorer, "_predict", new_callable=AsyncMock)
@patch("rager.scorers.Jar")
@pytest.mark.asyncio
async def test_cross_encoder_scorer_score(
    jar_cls: MagicMock, predict: AsyncMock
) -> None:
    """score() folds its identity into the jar and seals the score."""
    # Arrange
    scorer = CrossEncoderScorer("test-model")
    expected_score = 0.9
    predict.return_value = expected_score
    jar = jar_cls[float].return_value
    jar.get.return_value = None
    jar.set.side_effect = lambda value: value

    # Act
    result = await scorer.score("query", "chunk")

    # Assert
    predict.assert_awaited_once_with("query", "chunk")
    jar.include.assert_any_call("test-model")
    jar.include.assert_any_call("query")
    jar.include.assert_any_call("chunk")
    jar.set.assert_called_once_with(expected_score)
    assert result == expected_score


@patch.object(CrossEncoderScorer, "_predict", new_callable=AsyncMock)
@patch("rager.scorers.Jar")
@pytest.mark.asyncio
async def test_cross_encoder_scorer_score_cached(
    jar_cls: MagicMock, predict: AsyncMock
) -> None:
    """score() returns the sealed score without predicting on a cache hit."""
    # Arrange
    scorer = CrossEncoderScorer("test-model")
    expected_score = 0.0
    jar = jar_cls[float].return_value
    jar.get.return_value = expected_score

    # Act
    result = await scorer.score("query", "chunk")

    # Assert
    predict.assert_not_awaited()
    jar.set.assert_not_called()
    assert result == expected_score
