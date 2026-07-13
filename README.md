# rager

Caching based RAG primitives — composable building blocks for retrieval-augmented generation with caching baked in.

## Install

```
uv add rager
```

## Getting started

```
uv run pytest
uv run prek install
```

Add an `--extra` to pin the default device: `rocm` (AMD), `cu130` (NVIDIA), or `cpu`. With `cpu`, torch supplies automatic fallback. For example: `uv run --extra cpu pytest`.
