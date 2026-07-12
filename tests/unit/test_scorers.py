"""Unit tests for scorers."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rager.scorers import CrossEncoderScorer

pytestmark = pytest.mark.unit


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
@patch("rager.scorers.belljar.check")
@patch("rager.scorers.belljar.include")
@pytest.mark.asyncio
async def test_cross_encoder_scorer_score(
    include: MagicMock, check: MagicMock, predict: AsyncMock
) -> None:
    """score() folds its identity into belljar and delegates to _predict."""
    # Arrange
    scorer = CrossEncoderScorer("test-model")
    expected_score = 0.9
    predict.return_value = expected_score

    # Act
    result = await scorer.score("query", "chunk")

    # Assert
    predict.assert_awaited_once_with("query", "chunk")
    include.assert_any_call("test-model")
    include.assert_any_call("query")
    include.assert_any_call("chunk")
    check.assert_called_once()
    assert result == expected_score
