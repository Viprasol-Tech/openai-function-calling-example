"""OpenAI Function Calling Example — by Viprasol Tech Private Limited.

Generate OpenAI-style JSON tool schemas from plain Python functions and
dispatch tool-call dicts back to those functions, fully offline.
"""

from __future__ import annotations

from openai_function_calling_example.dispatch import (
    ToolDispatcher,
    ToolError,
    UnknownToolError,
)
from openai_function_calling_example.schema import function_to_schema

__all__ = [
    "ToolDispatcher",
    "ToolError",
    "UnknownToolError",
    "__version__",
    "function_to_schema",
]

__version__ = "0.1.0"
