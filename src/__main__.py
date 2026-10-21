"""Command line entry point: ``uv run python -m src [options]``."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.constrained_decoder import DecodingConfig
from src.io_utils import (
    InputError,
    load_function_definitions,
    load_prompts,
    write_results,
)
from src.llm import LLMClient, LLMError
from src.pipeline import FunctionCaller
from src.schema import FunctionCallResult
from src.visualize import render_trace

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Translate natural-language prompts into function "
                    "calls with constrained decoding.",
    )
    parser.add_argument(
        "--functions_definition", type=Path,
        default=Path("data/input/functions_definition.json"),
        help="JSON file with the available functions.")
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/input/function_calling_tests.json"),
        help="JSON file with the prompts to process.")
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/output/function_calling_results.json"),
        help="Where to write the results.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Hugging Face model id (default: {DEFAULT_MODEL}).")
    parser.add_argument(
        "--visualize", action="store_true",
        help="Print the token-by-token generation process.")
    return parser.parse_args(argv)


def log(message: str) -> None:
    """Print a progress message on stderr."""
    print(message, file=sys.stderr, flush=True)


def run(args: argparse.Namespace) -> int:
    """Run the whole pipeline; return the process exit code."""
    try:
        functions = load_function_definitions(args.functions_definition)
        prompts = load_prompts(args.input)
    except InputError as exc:
        log(f"error: {exc}")
        return 1

    results: list[FunctionCallResult] = []
    if not prompts:
        log("warning: no prompt to process")
    else:
        log(f"loading model {args.model} ...")
        try:
            llm = LLMClient.load(args.model)
            caller = FunctionCaller.create(
                llm, functions, DecodingConfig(trace=args.visualize))
        except LLMError as exc:
            log(f"error: {exc}")
            return 1
        start = time.perf_counter()
        try:
            for index, prompt in enumerate(prompts, 1):
                t0 = time.perf_counter()
                try:
                    outcome = caller.call(prompt)
                except Exception as exc:
                    log(f"[{index}/{len(prompts)}] warning: skipping "
                        f"{prompt!r}: {type(exc).__name__}: {exc}")
                    continue
                results.append(outcome.result)
                if args.visualize:
                    render_trace(prompt, outcome.decode)
                note = (f" (recovered: {outcome.decode.error})"
                        if outcome.decode.recovered else "")
                log(f"[{index}/{len(prompts)}] {outcome.result.name} "
                    f"{outcome.result.parameters} "
                    f"({time.perf_counter() - t0:.1f}s){note}")
        except KeyboardInterrupt:
            log("interrupted: writing the results obtained so far")
            _write(args.output, results)
            return 130
        log(f"processed {len(results)} prompt(s) in "
            f"{time.perf_counter() - start:.1f}s "
            f"({llm.calls} forward passes, {llm.cache_hits} cache hits)")
    return _write(args.output, results)


def _write(path: Path, results: list[FunctionCallResult]) -> int:
    """Write the results file and return an exit code."""
    try:
        write_results(path, results)
    except InputError as exc:
        log(f"error: {exc}")
        return 1
    log(f"results written to {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point that never lets an exception escape."""
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        log("interrupted")
        return 130
    except Exception as exc:
        log(f"error: unexpected failure: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
