"""Integration tests for document parsers."""

from typing import TYPE_CHECKING

import pytest

from rager.parsers import (
    CsvFileParser,
    CsvPageParser,
    MarkdownFileParser,
    MarkdownPageParser,
    PdfFileParser,
    PdfPageParser,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_pdf_file_parser_units(pdf_file: Path) -> None:
    """Test that the PDF parser correctly extracts the document elements."""
    parser = PdfFileParser()
    content = await parser.units(pdf_file)
    assert len(content) > 0


@pytest.mark.asyncio
async def test_markdown_file_parser_units(markdown_file: Path) -> None:
    """Test that the Markdown parser correctly extracts the document elements."""
    parser = MarkdownFileParser()
    content = await parser.units(markdown_file)
    assert len(content) > 0


@pytest.mark.asyncio
async def test_csv_file_parser_units(csv_file: Path) -> None:
    """Test that the CSV parser correctly extracts the document elements."""
    parser = CsvFileParser()
    content = await parser.units(csv_file)
    assert len(content) > 0


@pytest.mark.asyncio
async def test_pdf_page_parser_units(pdf_file: Path) -> None:
    """Test that the PDF page parser correctly extracts one string per page."""
    parser = PdfPageParser()
    pages = await parser.units(pdf_file)
    assert len(pages) == 1
    assert all(isinstance(page, str) for page in pages)


@pytest.mark.asyncio
async def test_markdown_page_parser_units(markdown_file: Path) -> None:
    """Test that the Markdown page parser correctly extracts one string per page."""
    parser = MarkdownPageParser()
    pages = await parser.units(markdown_file)
    assert len(pages) == 1
    assert all(isinstance(page, str) for page in pages)


@pytest.mark.asyncio
async def test_csv_page_parser_units(csv_file: Path) -> None:
    """Test that the CSV page parser correctly extracts one string per page."""
    parser = CsvPageParser()
    pages = await parser.units(csv_file)
    assert len(pages) == 1
    assert all(isinstance(page, str) for page in pages)
