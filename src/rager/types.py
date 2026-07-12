"""Type definitions for rager."""

from typing import Protocol

from blake3 import blake3

Hash = blake3
DenseEmbedding = list[float]
SparseEmbedding = dict[int, float]


class Metadata(Protocol):
    """Protocol for the fields every chunk metadata class must provide."""

    @property
    def chunk(self) -> str:
        """The chunk text this metadata describes."""
        ...

    @property
    def file_id(self) -> Hash:
        """The content ID of the file the chunk was extracted from."""
        ...
