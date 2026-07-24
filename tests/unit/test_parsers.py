"""Unit tests for parsers."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rager.parsers import BaseParser, UnstructuredFileParser, UnstructuredPageParser

pytestmark = pytest.mark.unit


def test_base_parser_is_abstract() -> None:
    """BaseParser cannot be instantiated without the jar and batch operations."""
    # Act & Assert
    with pytest.raises(TypeError):
        BaseParser()  # type: ignore[abstract]


@patch("rager.parsers.partition")
@patch("rager.parsers.blake3.blake3")
@patch("rager.parsers.Jar")
@pytest.mark.asyncio
async def test_unstructured_file_parser_units(
    jar_cls: MagicMock, blake3_fn: MagicMock, partition: MagicMock
) -> None:
    """units() folds the file's content hash into the jar and seals the units."""
    # Arrange
    parser = UnstructuredFileParser()
    element = MagicMock()
    partition.return_value = [element]
    file = MagicMock()
    file.read_bytes.return_value = b"payload"
    jar = jar_cls[tuple].return_value
    jar.get.return_value = None
    jar.set.side_effect = lambda value: value

    # Act
    result = await parser.units(file)

    # Assert
    assert result == (element,)
    partition.assert_called_once_with(str(file))
    file.read_bytes.assert_called_once_with()
    blake3_fn.assert_called_once_with(b"payload")
    jar.include.assert_any_call(parser._jar.__code__)
    jar.include.assert_any_call(parser._batched.__code__)
    jar.include.assert_any_call(parser._units.__code__)
    jar.include.assert_any_call(parser.units.__code__)
    jar.include.assert_any_call(blake3_fn.return_value.digest.return_value)
    jar.set.assert_called_once_with((element,))


@patch("rager.parsers.partition")
@patch("rager.parsers.Jar")
@pytest.mark.asyncio
async def test_unstructured_file_parser_units_cached(
    jar_cls: MagicMock, partition: MagicMock
) -> None:
    """units() returns the sealed units without parsing on a cache hit."""
    # Arrange
    parser = UnstructuredFileParser()
    file = MagicMock()
    file.read_bytes.return_value = b"payload"
    cached = (MagicMock(),)
    jar = jar_cls[tuple].return_value
    jar.get.return_value = cached

    # Act
    result = await parser.units(file)

    # Assert
    assert result is cached
    partition.assert_not_called()
    jar.set.assert_not_called()


@patch.object(UnstructuredFileParser, "_batched", new_callable=AsyncMock)
@patch("rager.parsers.Jar")
@pytest.mark.asyncio
async def test_unstructured_file_parser_batches_concurrent_units_calls(
    jar_cls: MagicMock, batched: AsyncMock
) -> None:
    """Concurrent units() calls for different files share one _batched() call."""
    # Arrange
    parser = UnstructuredFileParser()
    first_file, second_file = MagicMock(), MagicMock()
    first_file.read_bytes.return_value = b"first"
    second_file.read_bytes.return_value = b"second"
    first_elements, second_elements = (MagicMock(),), (MagicMock(),)
    batched.return_value = [first_elements, second_elements]
    jar = jar_cls[tuple].return_value
    jar.get.return_value = None
    jar.set.side_effect = lambda value: value

    # Act
    first_result, second_result = await asyncio.gather(
        parser.units(first_file), parser.units(second_file)
    )

    # Assert
    batched.assert_awaited_once_with([first_file, second_file])
    assert first_result == first_elements
    assert second_result == second_elements


@patch.object(UnstructuredFileParser, "units", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_unstructured_page_parser_batched(file_units: AsyncMock) -> None:
    """_batched() groups each file's elements by page and joins their text."""
    # Arrange
    parser = UnstructuredPageParser()
    page_two = MagicMock(text="second page")
    page_two.metadata.page_number = 2
    page_one = MagicMock(text="first page")
    page_one.metadata.page_number = 1
    unpaged = MagicMock(text="stray text")
    unpaged.metadata.page_number = None
    file_units.side_effect = [(page_two, page_one), (unpaged,)]
    first_file, second_file = MagicMock(), MagicMock()

    # Act
    result = await parser._batched([first_file, second_file])

    # Assert
    file_units.assert_any_call(first_file)
    file_units.assert_any_call(second_file)
    assert result == [("first page", "second page"), ("stray text",)]


def test_unstructured_page_parser_delegates_to_file_parser() -> None:
    """The page parser extracts its raw elements through a file parser."""
    # Act
    parser = UnstructuredPageParser()

    # Assert
    assert isinstance(parser._file_parser, UnstructuredFileParser)


@patch("rager.parsers.blake3.blake3")
@patch("rager.parsers.Jar")
@patch.object(UnstructuredPageParser, "_batched", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_unstructured_page_parser_units(
    batched: AsyncMock, jar_cls: MagicMock, blake3_fn: MagicMock
) -> None:
    """units() folds the file's content hash into the jar and seals the pages."""
    # Arrange
    parser = UnstructuredPageParser()
    batched.return_value = [("only page",)]
    file = MagicMock()
    file.read_bytes.return_value = b"payload"
    jar = jar_cls[tuple].return_value
    jar.get.return_value = None
    jar.set.side_effect = lambda value: value

    # Act
    result = await parser.units(file)

    # Assert
    assert result == ("only page",)
    batched.assert_awaited_once_with([file])
    jar.include.assert_any_call(blake3_fn.return_value.digest.return_value)
    jar.set.assert_called_once_with(("only page",))


@patch("rager.parsers.Jar")
@patch.object(UnstructuredPageParser, "_batched", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_unstructured_page_parser_units_cached(
    batched: AsyncMock, jar_cls: MagicMock
) -> None:
    """units() returns the sealed pages without parsing on a cache hit."""
    # Arrange
    parser = UnstructuredPageParser()
    file = MagicMock()
    file.read_bytes.return_value = b"payload"
    cached = ("only page",)
    jar = jar_cls[tuple].return_value
    jar.get.return_value = cached

    # Act
    result = await parser.units(file)

    # Assert
    assert result is cached
    batched.assert_not_awaited()
    jar.set.assert_not_called()
