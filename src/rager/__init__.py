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
from rager.indexes import (
    FileSparseIndex,
    Index,
    MemoryDenseIndex,
    MemorySparseIndex,
)
from rager.parsers import (
    CsvFileParser,
    MarkdownFileParser,
    Parser,
    PdfFileParser,
    UnstructuredFileParser,
)
from rager.scorers import CrossEncoderScorer, Scorer
from rager.stores import FileStore, MemoryStore, Store
from rager.types import DenseEmbedding, Hash, SparseEmbedding

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "BordaCountFuser",
    "Chunker",
    "CrossEncoderScorer",
    "CsvFileParser",
    "DenseEmbedding",
    "Embedder",
    "FileSparseIndex",
    "FileStore",
    "Fuser",
    "Generator",
    "Hash",
    "Index",
    "MarkdownFileParser",
    "MemoryDenseIndex",
    "MemorySparseIndex",
    "MemoryStore",
    "Parser",
    "PdfFileParser",
    "ReciprocalRankFuser",
    "Scorer",
    "SemanticChunker",
    "SentenceTransformerDenseEmbedder",
    "SparseEmbedding",
    "SpladeSparseEmbedder",
    "Store",
    "TransformersGenerator",
    "UnstructuredFileParser",
]
