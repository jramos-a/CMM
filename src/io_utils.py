"""Robust JSON loading and writing helpers for the CLI."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.schema import FunctionCallResult, FunctionDefinition, PromptEntry


class InputError(RuntimeError):
    """Raised when an input file is missing, unreadable or invalid."""


def _warn(message: str) -> None:
    """Print a warning on stderr."""
    print(f"warning: {message}", file=sys.stderr)


def read_json_file(path: Path) -> Any:
    """Read and parse a JSON file.

    Args:
        path: File to read.

    Returns:
        The decoded JSON value.

    Raises:
        InputError: If the file is missing, unreadable or not valid JSON.
    """
    if not path.exists():
        raise InputError(f"input file not found: {path}")
    if not path.is_file():
        raise InputError(f"input path is not a file: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise InputError(
            f"invalid JSON in {path} (line {exc.lineno}, column "
            f"{exc.colno}): {exc.msg}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise InputError(f"{path} is not valid UTF-8: {exc}") from exc
    except OSError as exc:
        raise InputError(f"could not read {path}: {exc}") from exc


def load_prompts(path: Path) -> list[str]:
    """Load the natural-language prompts to process.

    Entries may be ``{"prompt": "..."}`` objects or bare strings. Invalid
    entries are skipped with a warning instead of aborting the whole run.

    Args:
        path: Path to ``function_calling_tests.json``.

    Returns:
        The list of prompts, in file order.

    Raises:
        InputError: If the file cannot be used at all.
    """
    raw = read_json_file(path)
    if not isinstance(raw, list):
        raise InputError(f"{path} must contain a JSON array of prompts")
    prompts: list[str] = []
    for index, entry in enumerate(raw):
        if isinstance(entry, str):
            prompts.append(entry)
            continue
        try:
            prompts.append(PromptEntry.model_validate(entry).prompt)
        except ValidationError:
            _warn(f"skipping invalid prompt entry #{index} in {path}: "
                  f"{entry!r}")
    return prompts


def load_function_definitions(path: Path) -> list[FunctionDefinition]:
    """Load and validate the available function definitions.

    Invalid definitions and duplicated names are skipped with a warning.

    Args:
        path: Path to ``functions_definition.json``.

    Returns:
        The validated definitions.

    Raises:
        InputError: If the file is unusable or holds no valid definition.
    """
    raw = read_json_file(path)
    if not isinstance(raw, list):
        raise InputError(f"{path} must contain a JSON array of functions")
    definitions: list[FunctionDefinition] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw):
        try:
            definition = FunctionDefinition.model_validate(entry)
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                for err in exc.errors()
            )
            _warn(f"skipping invalid function #{index} in {path}: {details}")
            continue
        if definition.name in seen:
            _warn(f"skipping duplicated function name {definition.name!r}")
            continue
        seen.add(definition.name)
        definitions.append(definition)
    if not definitions:
        raise InputError(f"no valid function definition found in {path}")
    return definitions


def write_results(path: Path, results: list[FunctionCallResult]) -> None:
    """Write the results atomically as a pretty-printed JSON array.

    Args:
        path: Destination file.
        results: Results to serialise.

    Raises:
        InputError: If the file cannot be written.
    """
    payload = [result.model_dump() for result in results]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=path.parent, prefix=".tmp_", suffix=".json"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
    except OSError as exc:
        raise InputError(f"could not write {path}: {exc}") from exc
