"""Type definitions for rager."""

from blake3 import blake3

Hash = blake3
DenseEmbedding = list[float]
SparseEmbedding = dict[int, float]
