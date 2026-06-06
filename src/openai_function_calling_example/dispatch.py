"""Register Python functions as tools and dispatch OpenAI tool-call dicts to them.

A model returns tool calls shaped like ``{"name": "add", "arguments": {...}}``
(the ``arguments`` field is JSON text in the wire format). ``ToolDispatcher``
registers plain Python functions, exposes their generated schemas, and executes
an incoming tool call by validating its arguments against the target signature
and invoking the function.

Everything here runs offline: no OpenAI client is required to register tools,
produce schemas, or dispatch calls.

Part of OpenAI Function Calling Example by Viprasol Tech Private Limited
(https://viprasol.com).
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from typing import Any

from openai_function_calling_example.schema import function_to_schema


class ToolError(RuntimeError):
    """Raised when a tool call cannot be dispatched (bad or missing arguments)."""


class UnknownToolError(ToolError):
    """Raised when a tool call names a function that is not registered."""


class ToolDispatcher:
    """Registry of callable tools plus an executor for model tool-call dicts.

    Example:
        >>> dispatcher = ToolDispatcher()
        >>> @dispatcher.register
        ... def add(a: int, b: int) -> int:
        ...     "Add two integers."
        ...     return a + b
        >>> dispatcher.dispatch({"name": "add", "arguments": {"a": 2, "b": 3}})
        5
    """

    def __init__(self) -> None:
        """Create an empty dispatcher."""
        self._tools: dict[str, Callable[..., Any]] = {}

    def register(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
    ) -> Callable[..., Any]:
        """Register ``fn`` as a tool and return it unchanged (usable as a decorator).

        Args:
            fn: The function to expose as a tool.
            name: Optional tool name override; defaults to ``fn.__name__``.

        Returns:
            The original function, so this can be used as ``@dispatcher.register``.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        tool_name = name or fn.__name__
        if tool_name in self._tools:
            raise ValueError(f"tool {tool_name!r} is already registered")
        self._tools[tool_name] = fn
        return fn

    @property
    def tool_names(self) -> list[str]:
        """Return the registered tool names in registration order."""
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-style tool schemas for every registered tool.

        Returns:
            A list suitable for passing as the ``tools`` argument of an
            OpenAI chat/responses request.
        """
        return [function_to_schema(fn, name=name) for name, fn in self._tools.items()]

    def dispatch(self, tool_call: dict[str, Any]) -> Any:
        """Execute a single tool call and return the function's result.

        Args:
            tool_call: A dict with a ``"name"`` key and an ``"arguments"`` value.
                ``arguments`` may be a dict or a JSON-encoded string (as sent on
                the OpenAI wire format); an absent ``arguments`` means no args.

        Returns:
            Whatever the target function returns.

        Raises:
            UnknownToolError: If ``name`` is missing or not registered.
            ToolError: If ``arguments`` is not valid JSON / a mapping, or does
                not satisfy the target function's signature.
        """
        name = tool_call.get("name")
        if not name or name not in self._tools:
            raise UnknownToolError(f"unknown tool: {name!r}")

        raw_arguments = tool_call.get("arguments", {})
        if isinstance(raw_arguments, str):
            try:
                raw_arguments = json.loads(raw_arguments) if raw_arguments else {}
            except json.JSONDecodeError as exc:
                raise ToolError(f"arguments for {name!r} are not valid JSON: {exc}") from exc
        if not isinstance(raw_arguments, dict):
            kind = type(raw_arguments).__name__
            raise ToolError(f"arguments for {name!r} must be an object, got {kind}")

        fn = self._tools[name]
        signature = inspect.signature(fn)
        try:
            bound = signature.bind(**raw_arguments)
        except TypeError as exc:
            raise ToolError(f"invalid arguments for {name!r}: {exc}") from exc
        bound.apply_defaults()
        return fn(*bound.args, **bound.kwargs)
