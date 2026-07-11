"""Protocols for content generators."""

from typing import Protocol


class Generator(Protocol):
    """Protocol for prompting content generators."""

    def prompt(self, query: str) -> str:
        """Generate content based on the query."""
        ...
