"""Parsers for extracting units from files."""

from typing import TYPE_CHECKING, Protocol

import belljar
import blake3
from unstructured.partition.auto import partition

if TYPE_CHECKING:
    from pathlib import Path

    from rager.types import Hash


class Parser[D](Protocol):
    """Protocol for parsers."""

    def units(self, file: D) -> list[str]:
        """Return the extracted text units."""
        ...

    def id(self, file: D) -> Hash:
        """Return the file ID."""
        ...


class PdfParser:
    """Parser for PDF documents."""

    @belljar.store
    def units(self, file: Path) -> list[str]:
        """Extract text units from PDF content."""
        belljar.include(file)
        belljar.check()
        elements = partition(filename=str(file))
        return [element.text for element in elements]

    def id(self, file: Path) -> Hash:
        """Return the content ID for PDF."""
        data = file.read_bytes()
        return blake3.blake3(data)
