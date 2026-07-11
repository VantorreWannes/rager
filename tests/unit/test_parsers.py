"""Unit tests for parsers."""

from unittest.mock import MagicMock, patch

import pytest

from rager.parsers import PdfParser

pytestmark = pytest.mark.unit


@patch("rager.parsers.belljar.check")
@patch("rager.parsers.belljar.include")
@patch("rager.parsers.partition")
def test_pdf_parser_units(
    partition: MagicMock,
    include: MagicMock,
    check: MagicMock,
) -> None:
    """units() returns the text of every element from partition()."""
    # Arrange
    partition.return_value = [
        MagicMock(text="first"),
        MagicMock(text="second"),
    ]
    file = MagicMock()

    # Act
    result = PdfParser().units(file)

    # Assert
    assert result == ["first", "second"]
    partition.assert_called_once_with(filename=str(file))
    include.assert_called_once()
    check.assert_called_once()


@patch("rager.parsers.blake3.blake3")
def test_pdf_parser_id(blake3: MagicMock) -> None:
    """id() hashes the raw bytes of the file."""
    # Arrange
    payload = b"payload"
    file = MagicMock()
    file.read_bytes.return_value = payload

    # Act
    result = PdfParser().id(file)

    # Assert
    file.read_bytes.assert_called_once_with()
    blake3.assert_called_once_with(b"payload")
    assert result is blake3.return_value
