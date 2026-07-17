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
from rager.embedders import SentenceTransformerEmbedder
from rager.indexes import FaissIndex
from rager.parsers import CsvPageParser, MarkdownPageParser, PdfPageParser
from rager.stores import MemoryStore

if TYPE_CHECKING:
    from pathlib import Path

    from rager.parsers import UnstructuredPageParser

pytestmark = pytest.mark.application


@pytest.mark.parametrize(
    ("parser", "file_fixture"),
    [
        (PdfPageParser, "pdf_file"),
        (MarkdownPageParser, "markdown_file"),
        (CsvPageParser, "csv_file"),
    ],
)
@pytest.mark.asyncio
async def test_file_ingestion_round_trips(
    parser: type[UnstructuredPageParser],
    file_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    """Parsing, chunking, and indexing a file lets its chunks retrieve themselves."""
    # Arrange
    file: Path = request.getfixturevalue(file_fixture)
    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    index: FaissIndex[int, list[float]] = FaissIndex(384, MemoryStore())
    chunks: MemoryStore[int, str] = MemoryStore()
    chunker = SemanticChunker("gpt-3.5-turbo", 1000, 0)
    ingested: list[str] = []
    for unit in await parser().units(file):
        for chunk in await chunker.chunks(unit):
            await index.set(len(ingested), await embedder.embed(chunk))
            chunks.set(len(ingested), chunk)
            ingested.append(chunk)

    # Act
    assert ingested, "the sample file produced no chunks to ingest"
    target = max(ingested, key=len)
    (key,) = await index.similar(await embedder.embed(target), embedding_results=1)

    # Assert
    assert chunks.get(key) == target
