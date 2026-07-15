"""Parsers for extracting units from files."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, override

import blake3
from belljar import Jar
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


class BaseParser[F](ABC):
    """Abstract helper base deriving the ``Parser`` protocol from three operations.

    Subclasses implement ``_jar``, ``_parse``, and ``id``; ``units`` seals
    every parsed file in the jar and returns it from there on later calls.
    """

    @abstractmethod
    def _jar(self, file: F) -> Jar[list[str]]:
        """Open a jar positioned at the identity of the given file."""

    @abstractmethod
    def _parse(self, file: F) -> list[str]:
        """Extract text units from the file."""

    @abstractmethod
    def id(self, file: F) -> Hash:
        """Return the file ID."""

    def units(self, file: F) -> list[str]:
        """Return the extracted text units."""
        jar = self._jar(file)
        if (cached := jar.get()) is not None:
            return cached
        logger.info("Cache miss; parsing file %s", file)
        units = self._parse(file)
        logger.debug("Extracted %d units from %s", len(units), file)
        return jar.set(units)


class UnstructuredFileParser(BaseParser[Path]):
    """Parser for unstructured files."""

    @override
    def _jar(self, file: Path) -> Jar[list[str]]:
        """Open a jar positioned at the identity of the given file's content."""
        jar = Jar[list[str]](Path(".jar/parsers"))
        jar.include(self._jar.__code__)
        jar.include(self.id(file).digest())
        return jar

    @override
    def _parse(self, file: Path) -> list[str]:
        """Extract text units from the unstructured file."""
        elements = partition(
            filename=str(file),
            chunking_strategy="by_title",
            max_characters=1500,
            new_after_n_chars=1200,
            combine_text_under_n_chars=500,
            overlap=150,
        )
        return [element.text for element in elements]

    @override
    def id(self, file: Path) -> Hash:
        """Return the content ID for the unstructured file."""
        data = file.read_bytes()
        logger.debug("Hashing %d bytes from %s", len(data), file)
        return blake3.blake3(data)


PdfFileParser = UnstructuredFileParser
MarkdownFileParser = UnstructuredFileParser
CsvFileParser = UnstructuredFileParser
