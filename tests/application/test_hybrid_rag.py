"""End-to-end hybrid retrieval-augmented generation with real models.

Chunk a set of documents, embed each chunk with both a dense and a sparse
model, index both, then answer a question by fusing the two rankings,
reranking the fused candidates, and prompting a generator with the top chunk
as context. Each stage is asserted independently so a regression in fusion,
reranking, or generation is caught on its own.

Because the dense and sparse indexes derive keys from their own embeddings,
the same chunk lands under two unrelated keys. The single chunk store is fed
both keys per chunk so every fused key resolves back to its text.
"""

import asyncio

import pytest

from rager.chunkers import SemanticChunker
from rager.embedders import SentenceTransformerDenseEmbedder, SpladeSparseEmbedder
from rager.fusers import ReciprocalRankFuser
from rager.generators import TransformersGenerator
from rager.indexes import DenseIndex, SparseIndex
from rager.scorers import CrossEncoderScorer
from rager.stores import MemoryStore

pytestmark = pytest.mark.application

DOCUMENTS = [
    "Cats purr when they are happy and content.",
    "The stock market closed higher today after a volatile session.",
    "Honeybees communicate the location of food through a waggle dance.",
]
RELEVANT = DOCUMENTS[0]


@pytest.mark.asyncio
async def test_hybrid_rag_fuses_reranks_and_answers() -> None:
    """Fusion ranks the on-topic chunk first, reranking keeps it, answer is grounded."""
    # Arrange
    chunker = SemanticChunker()
    dense_embedder = SentenceTransformerDenseEmbedder("all-MiniLM-L6-v2")
    sparse_embedder = SpladeSparseEmbedder("prithivida/Splade_PP_en_v1")
    dense_index = DenseIndex(384)
    sparse_index = SparseIndex()
    chunks: MemoryStore[int, str] = MemoryStore()
    fuser: ReciprocalRankFuser[int] = ReciprocalRankFuser()
    scorer = CrossEncoderScorer("cross-encoder/ms-marco-MiniLM-L6-v2")
    generator = TransformersGenerator(
        "HuggingFaceTB/SmolLM2-135M-Instruct", max_new_tokens=32
    )
    for document in DOCUMENTS:
        for chunk in chunker.chunks(document):
            dense_key, sparse_key = await asyncio.gather(
                dense_index.add(await dense_embedder.embed(chunk)),
                sparse_index.add(await sparse_embedder.embed(chunk)),
            )
            chunks.set(dense_key, chunk)
            chunks.set(sparse_key, chunk)

    # Act
    query = "Why do cats purr?"
    dense_ranking, sparse_ranking = await asyncio.gather(
        dense_index.similar(await dense_embedder.embed(query)),
        sparse_index.similar(await sparse_embedder.embed(query)),
    )
    candidates: list[str] = []
    for key in fuser.fuse(dense_ranking, sparse_ranking):
        text = chunks.get(key)
        if text is not None and text not in candidates:
            candidates.append(text)

    # Assert
    assert candidates[0] == RELEVANT

    # Act
    scores = await asyncio.gather(*(scorer.score(query, text) for text in candidates))
    reranked = [
        text for _, text in sorted(zip(scores, candidates, strict=True), reverse=True)
    ]

    # Assert
    assert reranked[0] == RELEVANT

    # Act
    answer = await generator.prompt(
        f"Answer using only the context.\n{reranked[0]}\n{query}"
    )

    # Assert
    assert isinstance(answer, str)
    assert answer.strip()
