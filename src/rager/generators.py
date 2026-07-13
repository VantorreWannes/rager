"""Generators for producing answers to prompts."""

import logging
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import belljar
import concresce
from transformers import pipeline

if TYPE_CHECKING:
    from transformers import TextGenerationPipeline

logger = logging.getLogger(__name__)


class Generator(Protocol):
    """Protocol for prompting content generators."""

    async def prompt(self, query: str) -> str:
        """Generate content based on the query."""
        ...


class TransformersGenerator:
    """Generator that prompts a local transformers text-generation model."""

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        max_new_tokens: int = 512,
    ) -> None:
        """Initialize the generator with a specific model."""
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

    @cached_property
    def model(self) -> TextGenerationPipeline:
        """Load the text-generation pipeline."""
        logger.info("Loading text-generation pipeline for model %r", self.model_name)
        return pipeline("text-generation", model=self.model_name)

    @concresce.batch
    async def _generate(self, query: str) -> str:
        """Generate an answer for each query in the collected batch."""
        queries = await concresce.collect(query)
        logger.debug(
            "Generating answers for a batch of %d queries with %r (max_new_tokens=%d)",
            len(queries),
            self.model_name,
            self.max_new_tokens,
        )
        chats = [[{"role": "user", "content": query}] for query in queries]
        outputs = self.model(
            chats,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        answers = [output[0]["generated_text"][-1]["content"] for output in outputs]
        return cast("str", answers)

    @belljar.store(Path(".jar/generators"))
    async def prompt(self, query: str) -> str:
        """Generate content based on the query."""
        belljar.include(self.model_name)
        belljar.include(self.max_new_tokens)
        belljar.include(query)
        belljar.check()
        logger.debug(
            "Cache miss; generating answer for query of %d characters", len(query)
        )
        return await self._generate(query)
