"""End-to-end file ingestion across the supported parser formats.

Drives the parse -> chunk -> embed -> index -> store path that users run to
turn a file on disk into retrievable chunks. The assertion is a structural
round-trip -- an ingested chunk, re-embedded, retrieves itself and resolves
back to its own text -- rather than semantic quality, since the sample files
carry no fixed fact to query for.
"""

from typing import TYPE_CHECKING

import pytest

from rager.chunkers import SemanticChunker
from rager.embedders import SentenceTransformerDenseEmbedder
from rager.indexes import MemoryDenseIndex
from rager.parsers import CsvFileParser, MarkdownFileParser, PdfFileParser
from rager.stores import MemoryStore

if TYPE_CHECKING:
    from pathlib import Path

    from rager.parsers import UnstructuredFileParser

pytestmark = pytest.mark.application


@pytest.mark.parametrize(
    ("parser", "file_fixture"),
    [
        (PdfFileParser, "pdf_file"),
        (MarkdownFileParser, "markdown_file"),
        (CsvFileParser, "csv_file"),
    ],
)
@pytest.mark.asyncio
async def test_file_ingestion_round_trips(
    parser: type[UnstructuredFileParser],
    file_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    """Parsing, chunking, and indexing a file lets its chunks retrieve themselves."""
    # Arrange
    file: Path = request.getfixturevalue(file_fixture)
    embedder = SentenceTransformerDenseEmbedder("all-MiniLM-L6-v2")
    index = MemoryDenseIndex(384)
    chunks: MemoryStore[int, str] = MemoryStore()
    ingested: list[str] = []
    for unit in parser().units(file):
        for chunk in SemanticChunker().chunks(unit):
            key = await index.add(await embedder.embed(chunk))
            chunks.set(key, chunk)
            ingested.append(chunk)

    # Act
    assert ingested, "the sample file produced no chunks to ingest"
    target = max(ingested, key=len)
    (key,) = await index.similar(await embedder.embed(target), results=1)

    # Assert
    assert chunks.get(key) == target
