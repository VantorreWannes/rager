"""Generators for producing answers to prompts."""

from datetime import timedelta
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import belljar
import concresce
from transformers import pipeline

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from transformers import TextGenerationPipeline


class Generator(Protocol):
    """Protocol for prompting content generators."""

    async def prompt(self, query: str) -> str:
        """Generate content based on the query."""
        ...


class TransformersGenerator:
    """Generator that prompts a local transformers text-generation model."""

    def __init__(self, model_name: str, max_new_tokens: int = 512) -> None:
        """Initialize the generator with a specific model."""
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

    @cached_property
    def model(self) -> TextGenerationPipeline:
        """Load the text-generation pipeline."""
        return pipeline("text-generation", model=self.model_name)

    @cached_property
    def _generate(self) -> Callable[[str], Awaitable[str]]:
        """Coalesce concurrent calls into per-instance generation batches."""
        return concresce.batch(window=timedelta(milliseconds=100))(self._generate_batch)

    async def _generate_batch(self, query: str) -> str:
        """Generate an answer for each query in the collected batch."""
        queries = await concresce.collect(query)
        chats = [[{"role": "user", "content": query}] for query in queries]
        outputs = self.model(
            chats,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        answers = [output[0]["generated_text"][-1]["content"] for output in outputs]
        return concresce.scatter(answers)

    @belljar.store(Path(".jar/generators"))
    async def prompt(self, query: str) -> str:
        """Generate content based on the query."""
        belljar.include(self.model_name)
        belljar.include(self.max_new_tokens)
        belljar.include(query)
        belljar.check()
        return await self._generate(query)
