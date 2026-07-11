"""Content parsers for extracting units from documents."""

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from rager.types import Id


class ContentParser(Protocol):
    """Protocol for content parsers."""

    def units(self) -> list[str]:
        """Return the extracted text units."""
        ...

    def content(self) -> str:
        """Return the content."""
        ...

    def content_id(self) -> Id:
        """Return the content ID."""
        ...
