"""End-to-end dense retrieval-augmented generation with real models.

This reads top to bottom as a recipe: chunk a set of documents, embed each
chunk with a dense model, index the embeddings under a per-chunk key, then
answer a question by retrieving the nearest chunk and prompting a generator
with it as context. Copy it and swap primitives to fit your own library --
nothing is hidden behind a pipeline.
"""

import pytest

from rager.chunkers import SemanticChunker
from rager.embedders import SentenceTransformerDenseEmbedder
from rager.generators import TransformersGenerator
from rager.indexes import FaissIndex
from rager.stores import MemoryStore

pytestmark = pytest.mark.application

DOCUMENTS = [
    "Cats purr when they are happy and content.",
    "The stock market closed higher today after a volatile session.",
    "Honeybees communicate the location of food through a waggle dance.",
]


@pytest.mark.asyncio
async def test_dense_rag_retrieves_and_answers_from_context() -> None:
    """Dense retrieval surfaces the on-topic chunk and grounds the answer."""
    # Arrange
    chunker = SemanticChunker()
    embedder = SentenceTransformerDenseEmbedder("all-MiniLM-L6-v2")
    index: FaissIndex[int, list[float]] = FaissIndex(384, MemoryStore())
    chunks: MemoryStore[int, str] = MemoryStore()
    generator = TransformersGenerator(
        "HuggingFaceTB/SmolLM2-135M-Instruct", max_new_tokens=32
    )
    identifier = 0
    for document in DOCUMENTS:
        for chunk in chunker.chunks(document):
            await index.set(identifier, await embedder.embed(chunk))
            chunks.set(identifier, chunk)
            identifier += 1

    # Act
    query = "Why do cats purr?"
    (key,) = await index.similar(await embedder.embed(query), embedding_results=1)
    context = chunks.get(key)
    answer = await generator.prompt(
        f"Answer using only the context.\n{context}\n{query}"
    )

    # Assert
    assert context == DOCUMENTS[0]
    assert isinstance(answer, str)
    assert answer.strip()
