"""Tests for ``ToolDispatcher``."""

from __future__ import annotations

import pytest

from openai_function_calling_example.dispatch import (
    ToolDispatcher,
    ToolError,
    UnknownToolError,
)


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def greet(name: str, excited: bool = False) -> str:
    """Greet someone."""
    return f"Hello, {name}{'!' if excited else '.'}"


def _dispatcher() -> ToolDispatcher:
    d = ToolDispatcher()
    d.register(add)
    d.register(greet)
    return d


def test_dispatch_calls_function_with_dict_args() -> None:
    assert _dispatcher().dispatch({"name": "add", "arguments": {"a": 2, "b": 3}}) == 5


def test_dispatch_accepts_json_string_arguments() -> None:
    call = {"name": "greet", "arguments": '{"name": "Asha", "excited": true}'}
    result = _dispatcher().dispatch(call)
    assert result == "Hello, Asha!"


def test_dispatch_applies_defaults() -> None:
    assert _dispatcher().dispatch({"name": "greet", "arguments": {"name": "Bo"}}) == "Hello, Bo."


def test_register_as_decorator_returns_function() -> None:
    d = ToolDispatcher()

    @d.register
    def square(x: int) -> int:
        """Square a number."""
        return x * x

    assert square(4) == 16
    assert "square" in d.tool_names


def test_schemas_lists_all_tools() -> None:
    names = [s["function"]["name"] for s in _dispatcher().schemas()]
    assert names == ["add", "greet"]


def test_unknown_tool_raises() -> None:
    with pytest.raises(UnknownToolError):
        _dispatcher().dispatch({"name": "nope", "arguments": {}})


def test_missing_name_raises() -> None:
    with pytest.raises(UnknownToolError):
        _dispatcher().dispatch({"arguments": {}})


def test_bad_arguments_raise_tool_error() -> None:
    with pytest.raises(ToolError):
        _dispatcher().dispatch({"name": "add", "arguments": {"a": 1}})


def test_extra_arguments_raise_tool_error() -> None:
    with pytest.raises(ToolError):
        _dispatcher().dispatch({"name": "add", "arguments": {"a": 1, "b": 2, "c": 3}})


def test_invalid_json_string_raises_tool_error() -> None:
    with pytest.raises(ToolError):
        _dispatcher().dispatch({"name": "add", "arguments": "{not json}"})


def test_non_object_arguments_raise_tool_error() -> None:
    with pytest.raises(ToolError):
        _dispatcher().dispatch({"name": "add", "arguments": "[1, 2]"})


def test_duplicate_registration_raises() -> None:
    d = ToolDispatcher()
    d.register(add)
    with pytest.raises(ValueError):
        d.register(add)
