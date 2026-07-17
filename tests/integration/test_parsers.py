"""Integration tests for document parsers."""

from typing import TYPE_CHECKING

import pytest

from rager.parsers import CsvFileParser, MarkdownFileParser, PdfFileParser

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration


def test_pdf_file_parser_id(pdf_file: Path) -> None:
    """Test that the PDF parser correctly extracts the document ID."""
    parser = PdfFileParser()
    document_id = parser.id(pdf_file)
    assert (
        document_id.hexdigest()
        == "d19a399b6bbf0b1c5e1f5b0cd08c1ee1b79534035abed9c66e0163d1703c653f"
    )


def test_pdf_file_parser_units(pdf_file: Path) -> None:
    """Test that the PDF parser correctly extracts the document content."""
    parser = PdfFileParser()
    content = parser.units(pdf_file)
    assert len(content) == 1


def test_markdown_file_parser_id(markdown_file: Path) -> None:
    """Test that the Markdown parser correctly extracts the document ID."""
    parser = MarkdownFileParser()
    document_id = parser.id(markdown_file)
    assert (
        document_id.hexdigest()
        == "5b99e74be511c0d4dbbbcfcbc3f151e5e5823da25efc84e40f22163c28a38730"
    )


def test_markdown_file_parser_units(markdown_file: Path) -> None:
    """Test that the Markdown parser correctly extracts the document content."""
    parser = MarkdownFileParser()
    content = parser.units(markdown_file)
    expected_length = 1
    assert len(content) == expected_length


def test_csv_file_parser_id(csv_file: Path) -> None:
    """Test that the CSV parser correctly extracts the document ID."""
    parser = CsvFileParser()
    document_id = parser.id(csv_file)
    assert (
        document_id.hexdigest()
        == "505ee0e5156d1ea7e642c2e0b64a868bd1485d582a418dd6c42ed66378bdcc0c"
    )


def test_csv_file_parser_units(csv_file: Path) -> None:
    """Test that the CSV parser correctly extracts the document content."""
    parser = CsvFileParser()
    content = parser.units(csv_file)
    expected_length = 1
    assert len(content) == expected_length
