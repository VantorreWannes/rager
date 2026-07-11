"""Protocol definitions for value fusing."""

from typing import Protocol


class Fuser[V](Protocol):
    """Protocol for fusing values into ranked lists."""

    def fuse(self, *values: list[V]) -> list[V]:
        """Fuse multiple values into one ranked list."""
        ...
