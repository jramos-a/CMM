"""Builds the text prompts given to the model.

Generation uses two prompts. The *selection* prompt lists every function
and is used until the function name is fixed. The *argument* prompt only
describes the chosen function: it is much shorter, and since the SDK has
no KV cache every step costs a full pass over the prompt, so the
arguments are generated about twice as fast.
"""

from __future__ import annotations

from src.schema import FunctionDefinition, ParameterSpec

SELECTION_SYSTEM = (
    "You are a function calling engine: you never answer the user, you "
    "call the one function below that fulfils the request.\n"
    "Functions:\n"
)

ARGUMENT_SYSTEM = (
    "You are a function calling engine. Extract the arguments of this "
    "function call from the user request.\n"
    "Rules: copy text values exactly as written in the request, without "
    "the quotes around them. Use literal values: write a symbol rather "
    "than its name. A regex value is a general regular expression (with "
    "character classes when the request describes a kind of character) "
    "written as a JSON string.\n"
    "Function:\n"
)


def describe_type(spec: ParameterSpec) -> str:
    """Render a parameter type in a compact, readable form."""
    if spec.enum is not None:
        return " | ".join(repr(v) for v in spec.enum)
    if spec.type == "array" and spec.items is not None:
        return f"list[{describe_type(spec.items)}]"
    if spec.type == "object" and spec.properties:
        inner = ", ".join(f"{k}: {describe_type(v)}"
                          for k, v in spec.properties.items())
        return "{" + inner + "}"
    return spec.type


def describe_function(fn: FunctionDefinition) -> str:
    """Render one function as ``name(args): description``."""
    params = ", ".join(f"{name}: {describe_type(spec)}"
                       for name, spec in fn.parameters.items())
    line = f"- {fn.name}({params})"
    return f"{line}: {fn.description}" if fn.description else line


def wrap(system: str, request: str, chat: bool, thinking_tags: bool) -> str:
    """Wrap a system text and the user request in the model's template.

    Args:
        system: System instructions.
        request: The user's natural-language request.
        chat: Use the ChatML template (``<|im_start|>`` tokens).
        thinking_tags: Add an empty ``<think>`` block (Qwen3) so the model
            answers directly instead of reasoning first.

    Returns:
        The prompt text; generation continues right after it.
    """
    if not chat:
        return f"{system}\n\nRequest: {request}\nFunction call: "
    think = "<think>\n\n</think>\n\n" if thinking_tags else ""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{request}<|im_end|>\n"
        f"<|im_start|>assistant\n{think}"
    )


def build_selection_prompt(
    request: str, functions: list[FunctionDefinition], chat: bool,
    thinking_tags: bool,
) -> str:
    """Build the prompt used to choose the function."""
    listing = "\n".join(describe_function(fn) for fn in functions)
    return wrap(SELECTION_SYSTEM + listing, request, chat, thinking_tags)


def build_argument_prompt(
    request: str, fn: FunctionDefinition, chat: bool, thinking_tags: bool,
) -> str:
    """Build the prompt used to generate the arguments of ``fn``."""
    return wrap(ARGUMENT_SYSTEM + describe_function(fn), request, chat,
                thinking_tags)
