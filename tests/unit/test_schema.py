"""Tests for ``function_to_schema``."""

from __future__ import annotations

from openai_function_calling_example.schema import function_to_schema


def sample(city: str, days: int, ratio: float = 1.0, verbose: bool = False) -> str:
    """Look up a forecast.

    A longer explanation that should be ignored by the description extractor.
    """
    return f"{city}:{days}:{ratio}:{verbose}"


def test_schema_top_level_shape() -> None:
    schema = function_to_schema(sample)
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "sample"
    assert fn["description"] == "Look up a forecast."
    assert fn["parameters"]["type"] == "object"


def test_schema_parameter_types() -> None:
    props = function_to_schema(sample)["function"]["parameters"]["properties"]
    assert props["city"]["type"] == "string"
    assert props["days"]["type"] == "integer"
    assert props["ratio"]["type"] == "number"
    assert props["verbose"]["type"] == "boolean"


def test_schema_required_excludes_defaults() -> None:
    required = function_to_schema(sample)["function"]["parameters"]["required"]
    assert required == ["city", "days"]


def test_schema_optional_annotation_unwrapped() -> None:
    def fn(limit: int | None = None) -> int:
        """Do a thing."""
        return limit or 0

    props = function_to_schema(fn)["function"]["parameters"]["properties"]
    assert props["limit"]["type"] == "integer"
    assert function_to_schema(fn)["function"]["parameters"]["required"] == []


def test_schema_container_types() -> None:
    def fn(tags: list[str], meta: dict[str, int]) -> None:
        """Containers."""

    props = function_to_schema(fn)["function"]["parameters"]["properties"]
    assert props["tags"]["type"] == "array"
    assert props["meta"]["type"] == "object"


def test_schema_name_and_description_override() -> None:
    schema = function_to_schema(sample, name="forecast", description="Custom.")
    assert schema["function"]["name"] == "forecast"
    assert schema["function"]["description"] == "Custom."
