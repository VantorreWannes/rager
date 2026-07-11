"""Type definitions for rager."""

from typing import Protocol

from blake3 import blake3

Id = blake3
Embedding = list[float]


class Metadata(Protocol):
    """Metadata protocol for rager."""

    chunk: Id
    unit: Id
    content: Id
    text: str
