"""Tests for input/output handling and the command line."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import src.__main__ as cli
from src.io_utils import (
    InputError,
    load_function_definitions,
    load_prompts,
    write_results,
)
from src.llm import LLMClient
from src.schema import FunctionCallResult
from tests.conftest import RandomSDK, make_tokenizer

FUNCTIONS = [
    {"name": "fn_add_numbers", "description": "Add two numbers.",
     "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
     "returns": {"type": "number"}},
    {"name": "fn_greet", "description": "Greet someone.",
     "parameters": {"name": {"type": "string"}},
     "returns": {"type": "string"}},
]


def write(path: Path, content: Any) -> Path:
    """Write ``content`` (JSON-encoded unless a str) to ``path``."""
    text = content if isinstance(content, str) else json.dumps(content)
    path.write_text(text, encoding="utf-8")
    return path


def test_missing_file(tmp_path: Path) -> None:
    """A missing file gives a clear error."""
    with pytest.raises(InputError, match="not found"):
        load_prompts(tmp_path / "nope.json")


def test_invalid_json(tmp_path: Path) -> None:
    """Malformed JSON gives a clear error with its location."""
    path = write(tmp_path / "bad.json", '[{"prompt": "x"},]')
    with pytest.raises(InputError, match="invalid JSON"):
        load_prompts(path)


def test_not_a_list(tmp_path: Path) -> None:
    """The top-level value must be an array."""
    path = write(tmp_path / "obj.json", {"prompt": "x"})
    with pytest.raises(InputError, match="array"):
        load_prompts(path)


def test_bad_prompt_entries_are_skipped(tmp_path: Path) -> None:
    """Invalid entries are skipped, valid ones kept in order."""
    path = write(tmp_path / "p.json",
                 [{"prompt": "a"}, 3, {"other": 1}, "b", {"prompt": ""}])
    assert load_prompts(path) == ["a", "b", ""]


def test_bad_function_entries_are_skipped(tmp_path: Path) -> None:
    """Invalid or duplicated definitions are skipped."""
    path = write(tmp_path / "f.json", FUNCTIONS + [
        {"name": "", "parameters": {}},
        {"name": "fn_x", "parameters": {"a": {"type": "complex"}}},
        {"name": "fn_arr", "parameters": {"a": {"type": "array"}}},
        FUNCTIONS[0],
    ])
    names = [fn.name for fn in load_function_definitions(path)]
    assert names == ["fn_add_numbers", "fn_greet"]


def test_no_valid_function(tmp_path: Path) -> None:
    """At least one valid definition is required."""
    path = write(tmp_path / "f.json", [{"name": ""}])
    with pytest.raises(InputError, match="no valid function"):
        load_function_definitions(path)


def test_write_results(tmp_path: Path) -> None:
    """Results are written as a JSON array with exactly the 3 keys."""
    out = tmp_path / "deep" / "dir" / "out.json"
    write_results(out, [FunctionCallResult(
        prompt="p", name="fn", parameters={"a": 1.0})])
    assert json.loads(out.read_text(encoding="utf-8")) == [
        {"prompt": "p", "name": "fn", "parameters": {"a": 1.0}}]
    assert [p.name for p in out.parent.iterdir()] == ["out.json"]


def test_cli_reports_missing_input(tmp_path: Path,
                                   capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI exits with 1 and a message instead of crashing."""
    code = cli.main(["--input", str(tmp_path / "none.json"),
                     "--functions_definition", str(tmp_path / "f.json")])
    assert code == 1
    assert "error:" in capsys.readouterr().err


def test_cli_end_to_end_with_fake_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full run with a fake model: valid output for every prompt."""
    tokenizer = make_tokenizer()

    def fake_load(model_name: str) -> LLMClient:
        client = LLMClient(sdk=RandomSDK(tokenizer.vocab_size))
        monkeypatch.setattr(client.__class__, "load_tokenizer",
                            lambda self: tokenizer)
        return client

    monkeypatch.setattr(LLMClient, "load", staticmethod(fake_load))
    functions = write(tmp_path / "f.json", FUNCTIONS)
    prompts = write(tmp_path / "p.json",
                    [{"prompt": "What is 2 + 3?"}, {"prompt": "Greet bob"}])
    out = tmp_path / "out" / "results.json"
    code = cli.main(["--functions_definition", str(functions),
                     "--input", str(prompts), "--output", str(out)])
    assert code == 0
    results = json.loads(out.read_text(encoding="utf-8"))
    assert [r["prompt"] for r in results] == ["What is 2 + 3?", "Greet bob"]
    for result in results:
        assert list(result) == ["prompt", "name", "parameters"]
        assert result["name"] in ("fn_add_numbers", "fn_greet")
