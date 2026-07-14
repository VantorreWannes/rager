"""Caching based RAG primitives."""

import logging

from rager.chunkers import Chunker, SemanticChunker
from rager.embedders import (
    Embedder,
    SentenceTransformerDenseEmbedder,
    SpladeSparseEmbedder,
)
from rager.fusers import BordaCountFuser, Fuser, ReciprocalRankFuser
from rager.generators import Generator, TransformersGenerator
from rager.indexes import DenseIndex, Index, SparseIndex
from rager.parsers import (
    CsvFileParser,
    MarkdownFileParser,
    Parser,
    PdfFileParser,
    UnstructuredFileParser,
)
from rager.scorers import CrossEncoderScorer, Scorer
from rager.stores import CachedMemoryStore, MemoryStore, Store
from rager.types import DenseEmbedding, Hash, SparseEmbedding

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "BordaCountFuser",
    "CachedMemoryStore",
    "Chunker",
    "CrossEncoderScorer",
    "CsvFileParser",
    "DenseEmbedding",
    "DenseIndex",
    "Embedder",
    "Fuser",
    "Generator",
    "Hash",
    "Index",
    "MarkdownFileParser",
    "MemoryStore",
    "Parser",
    "PdfFileParser",
    "ReciprocalRankFuser",
    "Scorer",
    "SemanticChunker",
    "SentenceTransformerDenseEmbedder",
    "SparseEmbedding",
    "SparseIndex",
    "SpladeSparseEmbedder",
    "Store",
    "TransformersGenerator",
    "UnstructuredFileParser",
]
