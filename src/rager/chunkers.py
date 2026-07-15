"""Chunkers for splitting units into smaller chunks."""

import logging
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, override

from belljar import Jar
from semantic_chunker import get_chunker

if TYPE_CHECKING:
    from semantic_text_splitter import TextSplitter

logger = logging.getLogger(__name__)


class Chunker(Protocol):
    """Protocol for chunking units into smaller chunks."""

    def chunks(self, unit: str) -> list[str]:
        """Split a unit into smaller chunks."""
        ...


class BaseChunker(ABC):
    """Abstract helper base deriving the ``Chunker`` protocol from two operations.

    Subclasses implement ``_jar`` and ``_split``; ``chunks`` seals every
    split unit in the jar and returns it from there on later calls.
    """

    @abstractmethod
    def _jar(self, unit: str) -> Jar[list[str]]:
        """Open a jar positioned at the identity of the given unit."""

    @abstractmethod
    def _split(self, unit: str) -> list[str]:
        """Split a unit into smaller chunks."""

    def chunks(self, unit: str) -> list[str]:
        """Split a unit into smaller chunks."""
        jar = self._jar(unit)
        if (cached := jar.get()) is not None:
            return cached
        logger.debug("Cache miss; chunking unit of %d characters", len(unit))
        chunks = self._split(unit)
        logger.debug("Split unit into %d chunks", len(chunks))
        return jar.set(chunks)


class SemanticChunker(BaseChunker):
    """Chunker that splits units based on semantic boundaries."""

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        chunk_size: int = 1000,
        overlap: int = 0,
    ) -> None:
        """Initialize the semantic chunker."""
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.overlap = overlap

    @cached_property
    def model(self) -> TextSplitter:
        """Get the text splitter instance."""
        logger.info(
            "Loading semantic chunker for model %r (chunk_size=%d, overlap=%d)",
            self.model_name,
            self.chunk_size,
            self.overlap,
        )
        return cast(
            "TextSplitter",
            get_chunker(
                self.model_name,
                chunking_type="text",
                tree_sitter_language=None,
                max_tokens=self.chunk_size,
                overlap=self.overlap,
                trim=True,
            ),
        )

    @override
    def _jar(self, unit: str) -> Jar[list[str]]:
        """Open a jar positioned at the identity of the given unit."""
        jar = Jar[list[str]](Path(".jar/chunkers"))
        jar.include(self._jar.__code__)
        jar.include(self.model_name)
        jar.include(self.chunk_size)
        jar.include(self.overlap)
        jar.include(unit)
        return jar

    @override
    def _split(self, unit: str) -> list[str]:
        """Split a unit into semantically meaningful chunks."""
        return self.model.chunks(unit)
