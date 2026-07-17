"""Parsers for extracting units from files."""

import asyncio
import logging
from abc import ABC, abstractmethod
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, override

import blake3
import concresce
from belljar import Jar
from unstructured.documents.elements import Element
from unstructured.partition.auto import partition

if TYPE_CHECKING:
    from collections.abc import Awaitable

logger = logging.getLogger(__name__)


class Parser[D, V](Protocol):
    """Protocol for parsers."""

    def units(self, data: D) -> Awaitable[tuple[V, ...]]:
        """Return the extracted text units."""
        ...


class BaseParser[D, V](ABC):
    """Abstract helper base deriving the ``Parser`` protocol from two operations.

    Subclasses implement ``_jar`` and ``_batched``; ``units`` seals every
    parsed data instance in the jar and returns it from there on later calls.
    """

    @abstractmethod
    def _jar(self, data: D) -> Jar[tuple[V, ...]]:
        """Open a jar positioned at the identity of the given data."""

    @abstractmethod
    def _batched(self, datas: list[D]) -> Awaitable[list[tuple[V, ...]]]:
        """Extract text units from each data instance."""

    @concresce.batch
    async def _units(self, data: D) -> tuple[V, ...]:
        """Extract text units from the data."""
        datas = await concresce.collect(data)
        results = await self._batched(datas)
        return concresce.scatter(results)

    async def units(self, data: D) -> tuple[V, ...]:
        """Return the extracted text units."""
        jar = self._jar(data)
        if (cached := jar.get()) is not None:
            return cached
        logger.info("Cache miss; parsing data")
        units = await self._units(data)
        logger.debug("Extracted %d units", len(units))
        return jar.set(units)


class UnstructuredFileParser(BaseParser[Path, Element]):
    """Parser for unstructured files."""

    @override
    def _jar(self, data: Path) -> Jar[tuple[Element, ...]]:
        """Open a jar positioned at the identity of the given file's content."""
        jar = Jar[tuple[Element, ...]](Path(".jar/parsers"))
        jar.include(self._jar.__code__)
        jar.include(self._batched.__code__)
        jar.include(self._units.__code__)
        jar.include(self.units.__code__)
        jar.include(blake3.blake3(data.read_bytes()).digest())
        return jar

    @override
    async def _batched(self, datas: list[Path]) -> list[tuple[Element, ...]]:
        """Extract text units from each data instance."""
        return [tuple(partition(str(file))) for file in datas]


class UnstructuredPageParser(BaseParser[Path, str]):
    """Parser for unstructured file pages."""

    def __init__(self) -> None:
        """Initialize the page parser."""
        self._file_parser = UnstructuredFileParser()

    @override
    def _jar(self, data: Path) -> Jar[tuple[str, ...]]:
        """Open a jar positioned at the identity of the given file's content."""
        jar = Jar[tuple[str, ...]](Path(".jar/parsers"))
        jar.include(self._jar.__code__)
        jar.include(self._batched.__code__)
        jar.include(self._units.__code__)
        jar.include(self.units.__code__)
        jar.include(blake3.blake3(data.read_bytes()).digest())
        return jar

    @override
    async def _batched(self, datas: list[Path]) -> list[tuple[str, ...]]:
        """Extract text units from each data instance."""
        all_units = await asyncio.gather(
            *[self._file_parser.units(file) for file in datas]
        )

        file_pages = [
            groupby(
                sorted(file_units, key=lambda e: e.metadata.page_number or -1),
                key=lambda e: e.metadata.page_number or -1,
            )
            for file_units in all_units
        ]

        return [
            tuple("\n".join(e.text for e in group) for _, group in pages)
            for pages in file_pages
        ]


PdfFileParser = UnstructuredFileParser
MarkdownFileParser = UnstructuredFileParser
CsvFileParser = UnstructuredFileParser

PdfPageParser = UnstructuredPageParser
MarkdownPageParser = UnstructuredPageParser
CsvPageParser = UnstructuredPageParser
