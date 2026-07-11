"""Integration tests for document parsers."""

from typing import TYPE_CHECKING

from rager.parsers import PdfParser

if TYPE_CHECKING:
    from pathlib import Path


def test_pdf_parser_id(pdf_file: Path) -> None:
    """Test that the PDF parser correctly extracts the document ID."""
    parser = PdfParser()
    document_id = parser.id(pdf_file)
    assert (
        document_id.hexdigest()
        == "d19a399b6bbf0b1c5e1f5b0cd08c1ee1b79534035abed9c66e0163d1703c653f"
    )


def test_pdf_parser_content(pdf_file: Path) -> None:
    """Test that the PDF parser correctly extracts the document content."""
    parser = PdfParser()
    content = parser.units(pdf_file)
    assert len(content) == 1
