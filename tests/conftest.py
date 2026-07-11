"""Fixtures for tests suite."""

from typing import TYPE_CHECKING

import belljar
import httpx
import pytest
from lorem_text.lorem import paragraphs, words

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def words_text() -> str:
    """Return sample text words for testing."""
    return words(100)


@pytest.fixture
def paragraphs_text() -> str:
    """Return sample text paragraphs for testing."""
    return paragraphs(100)


@pytest.fixture
def words_file(tmp_path: Path, words_text: str) -> Path:
    """Return sample text words file for testing."""
    file_path = tmp_path / "words.txt"
    file_path.write_text(words_text)
    return file_path


@pytest.fixture
def paragraphs_file(tmp_path: Path, paragraphs_text: str) -> Path:
    """Return sample text paragraphs file for testing."""
    file_path = tmp_path / "paragraphs.txt"
    file_path.write_text(paragraphs_text)
    return file_path


@pytest.fixture
@belljar.store
def pdf_data() -> bytes:
    """Return sample PDF file data for testing."""
    url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    belljar.include(url)
    belljar.check()
    response = httpx.get(url, timeout=30)
    response.raise_for_status()
    return response.content


@pytest.fixture
def pdf_file(tmp_path: Path, pdf_data: bytes) -> Path:
    """Return sample PDF file path for testing."""
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(pdf_data)
    return file_path
