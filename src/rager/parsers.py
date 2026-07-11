"""Parsers for extracting units from files."""

from typing import TYPE_CHECKING, Protocol

import belljar
import blake3
from unstructured.partition.auto import partition

if TYPE_CHECKING:
    from pathlib import Path

    from rager.types import Hash


class Parser[F](Protocol):
    """Protocol for parsers."""

    def units(self, file: F) -> list[str]:
        """Return the extracted text units."""
        ...

    def id(self, file: F) -> Hash:
        """Return the file ID."""
        ...


class UnstructuredFileParser:
    """Parser for unstructured files."""

    @belljar.store
    def units(self, file: Path) -> list[str]:
        """Extract text units from the unstructured file."""
        file_id = self.id(file)
        belljar.include(file_id.digest())
        belljar.check()
        elements = partition(filename=str(file))
        return [element.text for element in elements]

    def id(self, file: Path) -> Hash:
        """Return the content ID for the unstructured file."""
        data = file.read_bytes()
        return blake3.blake3(data)


PdfFileParser = UnstructuredFileParser
MarkdownFileParser = UnstructuredFileParser
CsvFileParser = UnstructuredFileParser
