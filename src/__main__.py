# ABOUTME: CLI entry point. Loads inputs, runs the LLM through constrained
# ABOUTME: decoding for every prompt, and writes the results JSON file.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.io_utils import InputError, load_function_definitions, load_prompts, write_results
from src.pipeline import process_prompt
from src.schema import FunctionCallResult
from src.vocab import build_vocab_index, vocab_size_from_tokenizer_file


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="function-calling",
        description="Translate natural-language prompts into structured function calls.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input"),
        help="Directory containing function_calling_tests.json and the function "
        "definitions file (default: data/input).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calling_results.json"),
        help="Path to write function_calling_results.json (default: "
        "data/output/function_calling_results.json).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model identifier to load via llm_sdk (default: Qwen/Qwen3-0.6B).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        prompts = load_prompts(args.input)
        functions = load_function_definitions(args.input)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
    except ImportError as exc:
        print(f"error: could not import llm_sdk ({exc})", file=sys.stderr)
        return 1

    try:
        sdk = Small_LLM_Model(model_name=args.model)
        vocab_size = vocab_size_from_tokenizer_file(sdk)
        vocab_index = build_vocab_index(sdk, args.model, vocab_size)
    except Exception as exc:  # noqa: BLE001 - surface any load failure cleanly
        print(f"error: could not initialise the model ({exc})", file=sys.stderr)
        return 1

    results: list[FunctionCallResult] = []
    for prompt in prompts:
        try:
            results.append(process_prompt(sdk, vocab_index, prompt, functions))
        except Exception as exc:  # noqa: BLE001 - keep processing remaining prompts
            print(f"warning: failed to process prompt '{prompt}': {exc}", file=sys.stderr)

    try:
        write_results(args.output, results)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {len(results)} result(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
