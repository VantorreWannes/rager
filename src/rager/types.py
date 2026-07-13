"""Type definitions for rager."""

from typing import Protocol

from blake3 import blake3

Hash = blake3
DenseEmbedding = list[float]
SparseEmbedding = dict[int, float]


class Metadata(Protocol):
    """Protocol for the fields every chunk metadata class must provide."""
