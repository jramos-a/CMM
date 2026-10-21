"""Demonstration of tokenization and constrained decoding working together.

Run with ``uv run python -m src.demo ["some prompt"]``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.grammar import forced_text, initial_state
from src.io_utils import InputError, load_function_definitions
from src.llm import LLMClient, LLMError
from src.pipeline import FunctionCaller
from src.tokenizer import pre_tokenize
from src.visualize import render_trace


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n--- {title} " + "-" * max(0, 60 - len(title)))


def show_encoding(caller: FunctionCaller, prompt: str) -> None:
    """Show how our tokenizer encodes and decodes the prompt."""
    tok = caller.decoder.tokenizer
    section("1. Encoding with our own BPE tokenizer")
    print(f"text        : {prompt!r}")
    print(f"pre-tokens  : {pre_tokenize(prompt)}")
    ids = tok.encode(prompt)
    print(f"token ids   : {ids}")
    print(f"tokens      : {[caller.decoder.token_label(i) for i in ids]}")
    decoded = tok.decode(ids)
    print(f"decode(ids) : {decoded!r} (round trip ok: {decoded == prompt})")
    reference = getattr(caller.decoder.llm.sdk, "encode", None)
    if reference is not None:
        ref_ids = [int(i) for i in reference(prompt)[0].tolist()]
        print(f"matches the SDK reference encode(): {ref_ids == ids}")


def show_vocabulary(caller: FunctionCaller) -> None:
    """Show the vocabulary indexes built from the decoded tokens."""
    table = caller.decoder.table
    section("2. Vocabulary indexes (built once from decode)")
    print(f"tokens with a text          : {len(table.all_ids())}")
    print(f"always valid in a string    : {table.plain_string_ids.size}")
    print(f"with a quote or backslash   : {len(table.string_special_ids)}")
    print(f"multi-byte fragments        : {len(table.partial_ids)}")


def show_masking(caller: FunctionCaller, prompt: str) -> None:
    """Show one constrained step: raw top tokens vs allowed top tokens."""
    decoder = caller.decoder
    section("3. One constrained step: logits before and after the mask")
    forced, stack = forced_text(initial_state(caller.functions))
    ids = caller.selection_ids(prompt) + decoder.tokenizer.encode(forced)
    print(f"forced so far (no model call): {forced!r}")
    logits = decoder.llm.logits(ids)
    allowed, _ = decoder.candidates(stack)
    raw_top = np.argsort(-logits)[:5]
    print("unconstrained top 5 :",
          [decoder.token_label(int(i)) for i in raw_top])
    masked = np.full(logits.shape, -np.inf)
    allowed = allowed[allowed < logits.shape[0]]
    masked[allowed] = logits[allowed]
    top = [int(i) for i in np.argsort(-masked)[:5] if masked[i] > -np.inf]
    print(f"allowed tokens      : {allowed.size} of {logits.shape[0]}")
    print("constrained top 5   :", [decoder.token_label(i) for i in top])


def main(argv: list[str] | None = None) -> int:
    """Run the demonstration; return the exit code."""
    parser = argparse.ArgumentParser(prog="python -m src.demo")
    parser.add_argument("prompt", nargs="?",
                        default="What is the sum of 40 and 2?")
    parser.add_argument("--functions_definition", type=Path,
                        default=Path("data/input/functions_definition.json"))
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = parser.parse_args(argv)
    try:
        functions = load_function_definitions(args.functions_definition)
        caller = FunctionCaller.create(LLMClient.load(args.model), functions)
    except (InputError, LLMError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    show_encoding(caller, args.prompt)
    show_vocabulary(caller)
    show_masking(caller, args.prompt)
    section("4. Full constrained generation")
    caller.decoder.config.trace = True
    outcome = caller.call(args.prompt)
    render_trace(args.prompt, outcome.decode, sys.stdout)
    print(f"\nresult: {outcome.result.model_dump_json()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
