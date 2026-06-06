"""Command-line interface for OpenAI Function Calling Example.

``openai-function-calling-example demo`` generates a tool schema from a sample
Python function and dispatches a simulated model tool-call against it — fully
offline, with no OpenAI client or API key required.

Part of OpenAI Function Calling Example by Viprasol Tech Private Limited
(https://viprasol.com).
"""

from __future__ import annotations

import json

import typer
from rich.console import Console

from openai_function_calling_example import __version__
from openai_function_calling_example.dispatch import ToolDispatcher

app = typer.Typer(
    add_completion=False,
    help="OpenAI Function Calling Example — by Viprasol Tech.",
)
console = Console()


def get_weather(city: str, units: str = "celsius") -> dict[str, object]:
    """Return the current weather for a city.

    Args:
        city: The city to look up.
        units: Temperature units, ``celsius`` or ``fahrenheit``.

    Returns:
        A small mapping describing the (canned) weather for the city.
    """
    temperature = 21 if units == "celsius" else 70
    return {"city": city, "units": units, "temperature": temperature, "sky": "clear"}


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"openai-function-calling-example [bold cyan]{__version__}[/] - by Viprasol Tech")


@app.command()
def demo() -> None:
    """Generate a tool schema and dispatch a simulated tool call, offline."""
    dispatcher = ToolDispatcher()
    dispatcher.register(get_weather)

    console.print("[bold]Generated tool schema(s):[/]")
    console.print(json.dumps(dispatcher.schemas(), indent=2))

    # A model would emit something like this; arguments arrive as a JSON string.
    tool_call = {"name": "get_weather", "arguments": '{"city": "Mumbai", "units": "celsius"}'}
    console.print("\n[bold]Incoming tool call:[/]")
    console.print(json.dumps(tool_call, indent=2))

    result = dispatcher.dispatch(tool_call)
    console.print("\n[bold]Dispatched result:[/]")
    console.print(json.dumps(result, indent=2))
    console.print("\n[bold green]OK[/] - schema generated and tool call dispatched offline.")


if __name__ == "__main__":
    app()
