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


@pytest.fixture
@belljar.store
def markdown_data() -> bytes:
    """Return sample Markdown file data for testing."""
    url = "https://gist.githubusercontent.com/rt2zz/e0a1d6ab2682d2c47746950b84c0b6ee/raw/83b8b4814c3417111b9b9bef86a552608506603e/markdown-sample.md"
    belljar.include(url)
    belljar.check()
    response = httpx.get(url, timeout=30)
    response.raise_for_status()
    return response.content


@pytest.fixture
def markdown_file(tmp_path: Path, markdown_data: bytes) -> Path:
    """Return sample Markdown file path for testing."""
    file_path = tmp_path / "sample.md"
    file_path.write_bytes(markdown_data)
    return file_path


@pytest.fixture
@belljar.store
def csv_data() -> bytes:
    """Return sample CSV file data for testing."""
    url = "https://people.sc.fsu.edu/~jburkardt/data/csv/airtravel.csv"
    belljar.include(url)
    belljar.check()
    response = httpx.get(url, timeout=30)
    response.raise_for_status()
    return response.content


@pytest.fixture
def csv_file(tmp_path: Path, csv_data: bytes) -> Path:
    """Return sample CSV file path for testing."""
    file_path = tmp_path / "airtravel.csv"
    file_path.write_bytes(csv_data)
    return file_path
