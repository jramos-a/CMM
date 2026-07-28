# ABOUTME: Wires prompt construction and constrained decoding together to
# ABOUTME: turn a natural-language prompt into a validated function call.

from __future__ import annotations

from typing import Any, Protocol

from src.constrained_decoder import (
    constrained_choice_decode,
    constrained_number_decode,
    constrained_string_decode,
)
from src.schema import FunctionCallResult, FunctionDefinition
from src.vocab import VocabIndex


class LLM(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]: ...


def _build_selection_prompt(prompt: str, functions: list[FunctionDefinition]) -> str:
    lines = [
        "You are a function-calling assistant. Choose exactly one function "
        "that answers the user's request.",
        "Available functions:",
    ]
    for fn in functions:
        params = ", ".join(f"{name}: {spec.type}" for name, spec in fn.parameters.items())
        lines.append(f"- {fn.name}({params}): {fn.description}")
    lines.append(f'User request: "{prompt}"')
    lines.append("Function name: ")
    return "\n".join(lines)


def _build_argument_prompt(
    prompt: str, fn: FunctionDefinition, filled: dict[str, Any]
) -> str:
    filled_desc = ", ".join(f'"{k}": {v!r}' for k, v in filled.items()) or "(none yet)"
    return (
        f'User request: "{prompt}"\n'
        f"Chosen function: {fn.name}\n"
        f"Arguments filled so far: {filled_desc}\n"
        f"Next argument value: "
    )


def select_function(
    sdk: LLM, vocab_index: VocabIndex, prompt: str, functions: list[FunctionDefinition]
) -> FunctionDefinition:
    """Ask the LLM to pick a function name, restricted to the known names."""
    text = _build_selection_prompt(prompt, functions)
    ids = sdk.encode(text)
    names = [fn.name for fn in functions]
    chosen_name = constrained_choice_decode(sdk, vocab_index, ids, names)
    by_name = {fn.name: fn for fn in functions}
    return by_name[chosen_name]


def generate_arguments(
    sdk: LLM, vocab_index: VocabIndex, prompt: str, fn: FunctionDefinition
) -> dict[str, Any]:
    """Fill every declared parameter of ``fn`` with a constrained-decoded value."""
    args: dict[str, Any] = {}
    for param_name, spec in fn.parameters.items():
        text = _build_argument_prompt(prompt, fn, args)
        ids = sdk.encode(text)
        if spec.type == "number":
            raw = constrained_number_decode(sdk, vocab_index, ids)
            args[param_name] = float(raw)
        elif spec.type == "boolean":
            raw_bool = constrained_choice_decode(sdk, vocab_index, ids, ["true", "false"])
            args[param_name] = raw_bool == "true"
        else:  # "string"
            args[param_name] = constrained_string_decode(sdk, vocab_index, ids)
    return args


def process_prompt(
    sdk: LLM,
    vocab_index: VocabIndex,
    prompt: str,
    functions: list[FunctionDefinition],
) -> FunctionCallResult:
    fn = select_function(sdk, vocab_index, prompt, functions)
    args = generate_arguments(sdk, vocab_index, prompt, fn)
    return FunctionCallResult(prompt=prompt, fn_name=fn.name, args=args)
