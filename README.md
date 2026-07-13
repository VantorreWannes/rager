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

`uv add rager` pulls the CPU torch build from PyPI. For GPU (CUDA/ROCm) torch, add the matching [PyTorch index](https://pytorch.org/get-started/locally/) to your own project and install torch from it — those builds aren't on PyPI.
