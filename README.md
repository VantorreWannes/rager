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

Wire the primitives together yourself — there is no hidden pipeline. This chunks documents, embeds and indexes each chunk under its own key, then answers a query from the nearest chunk:

```python
chunker = SemanticChunker()
embedder = SentenceTransformerDenseEmbedder()
index = FaissIndex(384, MemoryStore())
chunks = MemoryStore()
generator = TransformersGenerator()

key = 0
for document in documents:
    for chunk in chunker.chunks(document):
        await index.set(key, await embedder.embed(chunk))
        chunks[key] = chunk
        key += 1

query = "Why do cats purr?"
(key,) = await index.similar(await embedder.embed(query), embedding_results=1)
answer = await generator.prompt(f"Answer using only the context.\n{chunks[key]}\n{query}")
```

`tests/application/` has full dense and hybrid recipes.

## API

Every stage is a `Protocol` with concrete implementations. Each module also ships an abstract `Base*` helper that derives the protocol from a few core operations, so you can drop in your own implementation. `async` methods batch concurrent calls; model-backed methods cache results under `.jar/`.

### Parsers — extract text units from files

- **`Parser`** — protocol: `units(file)` returns text units.
- **`BaseParser`** — derives a cached `units()` from `_jar`, `_batched`, and `_units`; the jar is keyed by the file's `blake3` content hash.
- **`UnstructuredFileParser`** — parses any file supported by [`unstructured`](https://github.com/Unstructured-IO/unstructured), returning its elements.
- **`UnstructuredPageParser`** — composes a file parser and returns one string per page, grouping elements by page number.
- **`PdfFileParser`**, **`MarkdownFileParser`**, **`CsvFileParser`** — aliases of `UnstructuredFileParser`; **`PdfPageParser`**, **`MarkdownPageParser`**, **`CsvPageParser`** — aliases of `UnstructuredPageParser`, for readable call sites.

### Chunkers — split units into chunks

- **`Chunker`** — protocol: `chunks(unit) -> list[str]`.
- **`BaseChunker`** — derives a cached `chunks()` from `_jar` and `_split`.
- **`SemanticChunker`** — splits on semantic boundaries with a token budget: `SemanticChunker(model_name="gpt-3.5-turbo", chunk_size=1000, overlap=0)`.

### Embedders — turn chunks into vectors

- **`Embedder[E]`** — protocol: `async embed(chunk) -> E`.
- **`BaseEmbedder`** — derives a cached `embed()` from `_jar` and `_encode`.
- **`SentenceTransformerDenseEmbedder`** — dense, L2-normalized `DenseEmbedding` via [SentenceTransformers](https://www.sbert.net/) (default `all-MiniLM-L6-v2`).
- **`SpladeSparseEmbedder`** — sparse `SparseEmbedding` via a SPLADE encoder (default `prithivida/Splade_PP_en_v1`).

### Indexes — store vectors and search by similarity

- **`Index[K, E]`** — protocol: a `Store[K, E]` of embeddings plus `async similar(embedding, embedding_results=100) -> list[key]`. Ranks by inner product (equals cosine for L2-normalized vectors).
- **`FaissIndex(dimensions, key_map)`** — flat [FAISS](https://github.com/facebookresearch/faiss) index; seals embeddings on disk under `.jar/` and records each key's FAISS id in the injected `key_map` store.
- **`SparseIndex(embedding_map, token_map)`** — inner-product search over sparse weight maps through an inverted token index; the injected stores decide whether embeddings live in memory or on disk, and search only loads the posting lists of the query's tokens.

### Fusers — merge ranked lists

- **`Fuser[V]`** — protocol: `fuse(*rankings) -> list[V]`.
- **`BaseFuser`** — derives `fuse()` from `_weight(rank, size)`.
- **`ReciprocalRankFuser`** — reciprocal rank fusion with smoothing constant `k` (default 60).
- **`BordaCountFuser`** — Borda count fusion.

### Scorers — rerank chunks against a query

- **`Scorer`** — protocol: `async score(query, chunk) -> float`.
- **`BaseScorer`** — derives a cached `score()` from `_jar` and `_predict`.
- **`CrossEncoderScorer`** — cross-encoder reranker (default `cross-encoder/ms-marco-MiniLM-L6-v2`).

### Generators — produce an answer

- **`Generator`** — protocol: `async prompt(query) -> str`.
- **`BaseGenerator`** — derives a cached `prompt()` from `_jar` and `_generate`.
- **`TransformersGenerator`** — local [Transformers](https://huggingface.co/docs/transformers) text-generation model (default `HuggingFaceTB/SmolLM2-135M-Instruct`, `max_new_tokens=512`).

### Stores — map keys to data

- **`Store[K, V]`** — protocol: mapping-style access (`store[key]`, `del store[key]`, `in`, `len()`, iteration) plus `set(key, value)`, `get(key) -> value | None`, `remove(key)`, `clear()`, and `keys()`.
- **`BaseStore`** — derives the full protocol from `keys`, `__setitem__`, `__getitem__`, and `__delitem__`.
- **`MemoryStore[K, V]`** — in-memory map from key to value (chunk text, embeddings, metadata, ...).
- **`FileStore[K, V]`** — like `MemoryStore`, but seals values on disk under `.jar/`, keeping only keys and digests in memory.

### Types

- **`Hash`** — a `blake3` hasher.
- **`DenseEmbedding`** — `list[float]`.
- **`SparseEmbedding`** — `dict[int, float]` mapping token id to weight.
