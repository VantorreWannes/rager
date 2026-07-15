"""Unit tests for parsers."""

from unittest.mock import MagicMock, patch

import pytest

from rager.parsers import UnstructuredFileParser

pytestmark = pytest.mark.unit


@patch("rager.parsers.blake3.blake3")
def test_unstructured_file_parser_id(blake3: MagicMock) -> None:
    """id() hashes the raw bytes of the file."""
    # Arrange
    payload = b"payload"
    file = MagicMock()
    file.read_bytes.return_value = payload

    # Act
    result = UnstructuredFileParser().id(file)

    # Assert
    file.read_bytes.assert_called_once_with()
    blake3.assert_called_once_with(b"payload")
    assert result is blake3.return_value


@patch("rager.parsers.Jar")
@patch("rager.parsers.partition")
def test_unstructured_file_parser_units(
    partition: MagicMock, jar_cls: MagicMock
) -> None:
    """units() folds the file digest into the jar and seals the units."""
    # Arrange
    partition.return_value = [
        MagicMock(text="first"),
        MagicMock(text="second"),
    ]
    file = MagicMock()
    file.read_bytes.return_value = b"payload"
    jar = jar_cls[list[str]].return_value
    jar.get.return_value = None
    jar.set.side_effect = lambda value: value

    # Act
    result = UnstructuredFileParser().units(file)

    # Assert
    assert result == ["first", "second"]
    partition.assert_called_once_with(
        filename=str(file),
        chunking_strategy="by_title",
        max_characters=1500,
        new_after_n_chars=1200,
        combine_text_under_n_chars=500,
        overlap=150,
    )
    jar.include.assert_any_call(UnstructuredFileParser().id(file).digest())
    jar.set.assert_called_once_with(["first", "second"])


@patch("rager.parsers.Jar")
@patch("rager.parsers.partition")
def test_unstructured_file_parser_units_cached(
    partition: MagicMock, jar_cls: MagicMock
) -> None:
    """units() returns the sealed units without parsing on a cache hit."""
    # Arrange
    file = MagicMock()
    file.read_bytes.return_value = b"payload"
    jar = jar_cls[list[str]].return_value
    jar.get.return_value = ["first", "second"]

    # Act
    result = UnstructuredFileParser().units(file)

    # Assert
    assert result == ["first", "second"]
    partition.assert_not_called()
    jar.set.assert_not_called()
