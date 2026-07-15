"""End-to-end document lifecycle: ingest, retrieve, then remove.

Exercises the full removal path across the index and the chunk store, which
the per-primitive tiers only cover in isolation: a chunk retrievable after
ingestion must disappear from results once removed, leaving the rest intact.
"""

import pytest

from rager.embedders import SentenceTransformerDenseEmbedder
from rager.indexes import FaissIndex
from rager.stores import MemoryStore

pytestmark = pytest.mark.application

DOCUMENTS = [
    "Cats purr when they are happy and content.",
    "The stock market closed higher today after a volatile session.",
    "Honeybees communicate the location of food through a waggle dance.",
]


@pytest.mark.asyncio
async def test_removed_document_drops_out_of_retrieval() -> None:
    """A removed chunk stops appearing in results while the others remain."""
    # Arrange
    embedder = SentenceTransformerDenseEmbedder("all-MiniLM-L6-v2")
    index: FaissIndex[int, list[float]] = FaissIndex(384, MemoryStore())
    chunks: MemoryStore[int, str] = MemoryStore()
    keys: dict[str, int] = {}
    for identifier, document in enumerate(DOCUMENTS):
        index[identifier] = await embedder.embed(document)
        chunks.set(identifier, document)
        keys[document] = identifier

    # Act
    query = await embedder.embed("Why do cats purr?")
    before = await index.similar(query, embedding_results=len(DOCUMENTS))
    assert chunks.get(before[0]) == DOCUMENTS[0]

    index.remove(keys[DOCUMENTS[0]])
    chunks.remove(keys[DOCUMENTS[0]])

    # Assert
    after = await index.similar(query, embedding_results=len(DOCUMENTS))
    assert keys[DOCUMENTS[0]] not in after
    assert chunks.get(keys[DOCUMENTS[0]]) is None
    assert len(after) == len(DOCUMENTS) - 1
    assert all(chunks.get(key) is not None for key in after)
