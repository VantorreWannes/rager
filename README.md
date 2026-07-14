# rager

Composable RAG primitives with caching baked in.

## Install

```
uv add rager
```

## Getting started

```
uv run pytest
uv run prek install
```

For GPU (CUDA/ROCm) torch, add the matching [PyTorch index](https://pytorch.org/get-started/locally/) to your own project and install torch from it — those builds aren't on PyPI.

## Example

Wire the primitives together yourself — there is no hidden pipeline. This chunks documents, embeds and indexes each chunk, then answers a query from the nearest chunk:

```python
chunker = SemanticChunker()
embedder = SentenceTransformerDenseEmbedder()
index = MemoryDenseIndex()
chunks = MemoryStore()
generator = TransformersGenerator()

for document in documents:
    for chunk in chunker.chunk(document):
       embedding = await embedder.embed(chunk)
       key = await index.add(embedding)
       chunks.set(key, chunk)

query = "Why do cats purr?"
(key,) = await index.similar(await embedder.embed(query), results=1)
answer = await generator.prompt(f"Answer using only the context.\n{chunks.get(key)}\n{query}")
```

`tests/application/` has full dense and hybrid recipes.

## API

Every stage is a `Protocol` with concrete implementations. `async` methods batch concurrent calls; model-backed methods cache results under `.jar/`.

### Parsers — extract text units from files

- **`Parser`** — protocol: `units(file)` returns text units, `id(file)` returns the content `Hash`.
- **`UnstructuredFileParser`** — parses any file supported by [`unstructured`](https://github.com/Unstructured-IO/unstructured).
- **`PdfFileParser`**, **`MarkdownFileParser`**, **`CsvFileParser`** — aliases of `UnstructuredFileParser` for readable call sites.

### Chunkers — split units into chunks

- **`Chunker`** — protocol: `chunk(unit) -> list[str]`.
- **`SemanticChunker`** — splits on semantic boundaries with a token budget: `SemanticChunker(model_name="gpt-3.5-turbo", chunk_size=1000, overlap=0)`.

### Embedders — turn chunks into vectors

- **`Embedder[E]`** — protocol: `async embed(chunk) -> E`.
- **`SentenceTransformerDenseEmbedder`** — dense, L2-normalized `DenseEmbedding` via [SentenceTransformers](https://www.sbert.net/) (default `all-MiniLM-L6-v2`).
- **`SpladeSparseEmbedder`** — sparse `SparseEmbedding` via a SPLADE encoder (default `prithivida/Splade_PP_en_v1`).

### Indexes — store vectors and search by similarity

- **`Index[E, K]`** — protocol: `async add(embedding) -> key`, `async remove(key)`, `async similar(embedding, results=100) -> list[key]`. Ranks by inner product (equals cosine for L2-normalized vectors). Keys are derived from embedding content, so adding the same vector twice yields one entry.
- **`MemoryDenseIndex`** — flat [FAISS](https://github.com/facebookresearch/faiss) index; `MemoryDenseIndex(dimensions=None)` infers width from the first vector unless fixed.
- **`MemorySparseIndex`** — in-memory inner-product search over sparse weight maps.
- **`FileSparseIndex`** — like `MemorySparseIndex`, but seals embeddings on disk under `.jar/`, keeping only keys in memory.

### Fusers — merge ranked lists

- **`Fuser[V]`** — protocol: `fuse(*rankings) -> list[V]`.
- **`ReciprocalRankFuser`** — reciprocal rank fusion with smoothing constant `k` (default 60).
- **`BordaCountFuser`** — Borda count fusion.

### Scorers — rerank chunks against a query

- **`Scorer`** — protocol: `async score(query, chunk) -> float`.
- **`CrossEncoderScorer`** — cross-encoder reranker (default `cross-encoder/ms-marco-MiniLM-L6-v2`).

### Generators — produce an answer

- **`Generator`** — protocol: `async prompt(query) -> str`.
- **`TransformersGenerator`** — local [Transformers](https://huggingface.co/docs/transformers) text-generation model (default `HuggingFaceTB/SmolLM2-135M-Instruct`, `max_new_tokens=512`).

### Stores — map index keys back to data

- **`Store[K, V]`** — protocol: `set(key, value)`, `get(key) -> value | None`, `remove(key)`, `keys()`.
- **`MemoryStore[K, V]`** — in-memory map from key to value (chunk text, embeddings, metadata, ...).
- **`FileStore[K, V]`** — like `MemoryStore`, but seals buffer values on disk under `.jar/`, keeping only keys and digests in memory.

### Types

- **`Hash`** — a `blake3` hasher; the content ID returned by parsers.
- **`DenseEmbedding`** — `list[float]`.
- **`SparseEmbedding`** — `dict[int, float]` mapping token id to weight.
