"""Parsers for extracting units from files."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import belljar
import blake3
from unstructured.partition.auto import partition

if TYPE_CHECKING:
    from rager.types import Hash

logger = logging.getLogger(__name__)


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

    @belljar.store(Path(".jar/parsers"))
    def units(self, file: Path) -> list[str]:
        """Extract text units from the unstructured file."""
        file_id = self.id(file)
        belljar.include(file_id.digest())
        belljar.check()
        logger.info("Cache miss; parsing file %s", file)
        elements = partition(filename=str(file))
        units = [element.text for element in elements]
        logger.debug("Extracted %d units from %s", len(units), file)
        return units

    def id(self, file: Path) -> Hash:
        """Return the content ID for the unstructured file."""
        data = file.read_bytes()
        logger.debug("Hashing %d bytes from %s", len(data), file)
        return blake3.blake3(data)


PdfFileParser = UnstructuredFileParser
MarkdownFileParser = UnstructuredFileParser
CsvFileParser = UnstructuredFileParser
