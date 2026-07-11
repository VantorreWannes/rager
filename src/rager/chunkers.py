"""Chunkers for splitting units into smaller chunks."""

from typing import Protocol


class Chunker(Protocol):
    """Protocol for chunking units into smaller chunks."""

    def chunk(self, unit: str) -> list[str]:
        """Split a unit into smaller chunks."""
        ...
