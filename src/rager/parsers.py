"""Text parsers for extracting units from data."""

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from rager.types import Id


class TextParser[D](Protocol):
    """Protocol for text parsers."""

    def units(self, content: D) -> list[str]:
        """Return the extracted text units."""
        ...

    def id(self, content: D) -> Id:
        """Return the content ID."""
        ...
