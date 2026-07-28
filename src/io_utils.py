# ABOUTME: Robust JSON loading/writing helpers for the function-calling CLI.
# ABOUTME: Tolerates missing files, malformed JSON, and either input file naming scheme.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.schema import FunctionDefinition, FunctionCallResult


class InputError(RuntimeError):
    """Raised when an input file is missing or fails to parse/validate."""


def _read_json_file(path: Path) -> Any:
    if not path.is_file():
        raise InputError(f"input file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise InputError(f"invalid JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise InputError(f"could not read {path}: {exc}") from exc


def load_prompts(input_dir: Path) -> list[str]:
    """Load the list of natural-language prompts to process.

    Accepts either a list of plain strings or a list of {"prompt": ...} objects.
    """
    path = input_dir / "function_calling_tests.json"
    raw = _read_json_file(path)
    if not isinstance(raw, list):
        raise InputError(f"{path} must contain a JSON array")

    prompts: list[str] = []
    for entry in raw:
        if isinstance(entry, str):
            prompts.append(entry)
        elif isinstance(entry, dict) and isinstance(entry.get("prompt"), str):
            prompts.append(entry["prompt"])
        else:
            raise InputError(f"unrecognised test entry in {path}: {entry!r}")
    return prompts


def load_function_definitions(input_dir: Path) -> list[FunctionDefinition]:
    """Load available function definitions, trying both known file names."""
    candidates = [
        input_dir / "function_definitions.json",
        input_dir / "functions_definition.json",
    ]
    path = next((c for c in candidates if c.is_file()), candidates[0])
    raw = _read_json_file(path)
    if not isinstance(raw, list):
        raise InputError(f"{path} must contain a JSON array")

    definitions: list[FunctionDefinition] = []
    for entry in raw:
        try:
            definitions.append(FunctionDefinition.model_validate(entry))
        except ValidationError as exc:
            raise InputError(f"invalid function definition in {path}: {exc}") from exc
    if not definitions:
        raise InputError(f"no function definitions found in {path}")
    return definitions


def write_results(output_path: Path, results: list[FunctionCallResult]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [result.model_dump() for result in results]
    try:
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    except OSError as exc:
        raise InputError(f"could not write {output_path}: {exc}") from exc
