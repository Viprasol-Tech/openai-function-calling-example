"""Generate OpenAI-style tool JSON schemas from plain Python functions.

``function_to_schema`` inspects a function's signature, type hints and
docstring and produces the ``{"type": "function", "function": {...}}`` shape
that the OpenAI Chat Completions / Responses tool-calling APIs expect. The
result is a plain ``dict`` built only from the standard library — no network,
no OpenAI client, no extra dependencies.

Supported parameter types map as follows::

    int   -> "integer"
    float -> "number"
    str   -> "string"
    bool  -> "boolean"
    list  -> "array"
    dict  -> "object"

A parameter is *required* when it has no default value. Parameters annotated
as ``Optional[X]`` (``X | None``) are treated as type ``X`` and, when they also
carry a default, become optional.

Part of OpenAI Function Calling Example by Viprasol Tech Private Limited
(https://viprasol.com).
"""

from __future__ import annotations

import inspect
import types
import typing
from collections.abc import Callable
from typing import Any, Union

#: Mapping of Python primitive types to JSON Schema ``type`` strings.
_JSON_TYPES: dict[type, str] = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _unwrap_optional(annotation: Any) -> Any:
    """Return the non-``None`` member of an ``Optional``/``X | None`` annotation.

    For a union with exactly one non-``None`` member (e.g. ``int | None`` or
    ``Optional[str]``) the inner type is returned. Any other annotation is
    returned unchanged.

    Args:
        annotation: The raw annotation taken from a function signature.

    Returns:
        The unwrapped annotation, or the original annotation when it is not a
        single-type optional.
    """
    origin = typing.get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        args = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _json_type_for(annotation: Any) -> str:
    """Map a Python annotation to a JSON Schema ``type`` string.

    Unknown or missing annotations fall back to ``"string"`` so that the
    generated schema is always valid.

    Args:
        annotation: A type annotation (possibly ``Optional``/``X | None``).

    Returns:
        The JSON Schema type name, such as ``"integer"`` or ``"string"``.
    """
    annotation = _unwrap_optional(annotation)
    if annotation is inspect.Parameter.empty:
        return "string"
    # Generic aliases like list[int] / dict[str, int] expose their base via origin.
    origin = typing.get_origin(annotation)
    base = origin if origin is not None else annotation
    if isinstance(base, type):
        for py_type, json_type in _JSON_TYPES.items():
            if base is py_type or (base is not bool and issubclass_safe(base, py_type)):
                return json_type
    return "string"


def issubclass_safe(candidate: Any, parent: type) -> bool:
    """Return ``issubclass(candidate, parent)`` without raising on non-types.

    Args:
        candidate: The object to test; may not be a class.
        parent: The parent class to test against.

    Returns:
        ``True`` if ``candidate`` is a subclass of ``parent``, else ``False``.
    """
    return isinstance(candidate, type) and issubclass(candidate, parent)


def _first_paragraph(docstring: str | None) -> str:
    """Extract the first paragraph of a docstring as a one-line description.

    Args:
        docstring: The raw ``__doc__`` of a function, or ``None``.

    Returns:
        The first non-empty paragraph collapsed to a single line, or an empty
        string when no docstring is present.
    """
    if not docstring:
        return ""
    lines: list[str] = []
    for raw in inspect.cleandoc(docstring).splitlines():
        if not raw.strip():
            break
        lines.append(raw.strip())
    return " ".join(lines)


def function_to_schema(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Build an OpenAI-style tool schema for a Python function.

    The returned dict has the shape::

        {
            "type": "function",
            "function": {
                "name": ...,
                "description": ...,
                "parameters": {
                    "type": "object",
                    "properties": {<param>: {"type": ...}, ...},
                    "required": [<param>, ...],
                },
            },
        }

    Args:
        fn: The function to describe. Its signature and type hints drive the
            ``parameters`` block; its docstring's first paragraph becomes the
            description unless one is supplied explicitly.
        name: Override for the tool name. Defaults to ``fn.__name__``.
        description: Override for the tool description. Defaults to the first
            paragraph of ``fn.__doc__``.

    Returns:
        A JSON-serialisable dict ready to pass as one element of a tools list.

    Raises:
        TypeError: If ``fn`` is not callable.
    """
    if not callable(fn):
        raise TypeError("function_to_schema expects a callable")

    signature = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:  # tolerate unresolved forward refs
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []
    for param_name, param in signature.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = hints.get(param_name, param.annotation)
        properties[param_name] = {"type": _json_type_for(annotation)}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": name or fn.__name__,
            "description": description if description is not None else _first_paragraph(fn.__doc__),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
