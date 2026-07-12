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
from rager.stores import ChunkStore, MetadataStore, Store
from rager.types import DenseEmbedding, Hash, Metadata, SparseEmbedding

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "BordaCountFuser",
    "ChunkStore",
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
    "Metadata",
    "MetadataStore",
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
