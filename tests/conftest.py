"""Fixtures for tests suite."""

from typing import TYPE_CHECKING

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
