"""Chunkers for splitting units into smaller chunks."""

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import belljar
from semantic_chunker import get_chunker

if TYPE_CHECKING:
    from semantic_text_splitter import TextSplitter


class Chunker(Protocol):
    """Protocol for chunking units into smaller chunks."""

    def chunk(self, unit: str) -> list[str]:
        """Split a unit into smaller chunks."""
        ...


class SemanticChunker:
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

    @belljar.store(Path(".jar/chunkers"))
    def chunk(self, unit: str) -> list[str]:
        """Split a unit into semantically meaningful chunks."""
        belljar.include(self.model_name)
        belljar.include(self.chunk_size)
        belljar.include(self.overlap)
        belljar.include(unit)
        belljar.check()
        return self.model.chunks(unit)
