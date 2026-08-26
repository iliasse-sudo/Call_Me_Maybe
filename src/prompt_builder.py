"""
Builds the natural-language prompt sent to the LLM before constrained
decoding takes over.

Important framing: this prompt's only job is to help the model pick the
CORRECT function and extract the CORRECT argument values from the user's
request. It does NOT need to teach JSON syntax, escaping rules, or output
formatting -- constrained decoding makes malformed output structurally
impossible regardless of what the model "wants" to write. Wasting prompt
budget on format instructions just eats into a 0.6B model's limited
context usefulness; keep it focused on disambiguation.
"""
from __future__ import annotations

from typing import Any


def _format_function_signature(fn: dict[str, Any]) -> str:
    params = fn.get("parameters", {})
    param_str = ", ".join(f"{name}: {spec['type']}" for name, spec in params.items())
    return f"- {fn['name']}({param_str}): {fn['description']}"


def build_prompt(functions: list[dict[str, Any]], user_request: str) -> str:
    """Assemble the full prompt text for a given request and function catalog.

    Args:
        functions: parsed contents of functions_definition.json.
        user_request: the natural-language prompt to fulfill.

    Returns:
        The prompt string, ending right before the model needs to produce
        the function call. The caller appends the JSON priming prefix
        (e.g. '{"name": "') separately, since that's a decoding-time
        concern, not a prompt-content concern.
    """
    catalog = "\n".join(_format_function_signature(fn) for fn in functions)

    # One worked example anchors the small model on "extract values, don't
    # answer the question in prose" -- the single biggest failure mode for
    # sub-1B models on this kind of task is answering "5" instead of
    # producing a call to fn_add_numbers with a=2, b=3.
    example = (
        'Request: "What is the sum of 2 and 3?"\n'
        'Call: fn_add_numbers(a=2, b=3)\n'
    )

    return (
        "You are a function-calling assistant. Given a request and a list "
        "of available functions, choose exactly one function that fulfills "
        "the request and extract its argument values from the request "
        "text.\n\n"
        "Available functions:\n"
        f"{catalog}\n\n"
        "Example:\n"
        f"{example}\n"
        f'Request: "{user_request}"\n'
        "Call:"
    )