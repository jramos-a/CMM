"""Tests for the schema-driven JSON automaton."""

from __future__ import annotations

import json
from typing import Optional

import pytest

from src.grammar import (
    Stack,
    advance_text,
    complete,
    forced_text,
    initial_state,
    lenient_advance,
)
from tests.conftest import SAMPLE_FUNCTIONS, matches_spec


def accepts(text: str) -> bool:
    """Return whether ``text`` is a complete accepted document."""
    end = advance_text(initial_state(SAMPLE_FUNCTIONS), text)
    return end == ()


def run(text: str) -> Optional[Stack]:
    """Feed ``text`` from the initial state."""
    return advance_text(initial_state(SAMPLE_FUNCTIONS), text)


@pytest.mark.parametrize("doc", [
    '{"name": "fn_add_numbers", "parameters": {"a": 2, "b": -3.5e+2}}',
    '{"name": "fn_add_numbers", "parameters": {"a": 0, "b": 0.25}}',
    '{"name": "fn_greet", "parameters": {"name": "a \\"q\\" \\\\ \\u00e9"}}',
    '{"name": "fn_flags", "parameters": {"verbose": true, "count": -7, '
    '"mode": "fast2"}}',
    '{"name": "fn_flags", "parameters": {"verbose": false, "count": 0, '
    '"mode": "fast"}}',
    '{"name": "fn_nested", "parameters": {"user": {"name": "x", "tags": []}, '
    '"points": [{"x": 1, "y": 2}, {"x": 3, "y": 4}], "matrix": [[1, 2], []], '
    '"nothing": null}}',
    '{"name": "fn_noargs", "parameters": {}}',
])
def test_accepts_valid_documents(doc: str) -> None:
    """Schema-valid canonical documents are accepted."""
    assert accepts(doc)
    json.loads(doc)


@pytest.mark.parametrize("doc", [
    '{"name": "fn_unknown", "parameters": {}}',
    '{"name": "fn_add_numbers", "parameters": {"a": 01, "b": 2}}',
    '{"name": "fn_add_numbers", "parameters": {"a": "2", "b": 2}}',
    '{"name": "fn_add_numbers", "parameters": {"a": 2}}',
    '{"name": "fn_add_numbers", "parameters": {"b": 2, "a": 2}}',
    '{"name": "fn_add_numbers", "parameters": {"a": 2, "b": 2, "c": 1}}',
    '{"name": "fn_add_numbers", "parameters": {"a": 2., "b": 2}}',
    '{"name": "fn_greet", "parameters": {"name": "bad \\x"}}',
    '{"name": "fn_greet", "parameters": {"name": "new\nline"}}',
    '{"name": "fn_flags", "parameters": {"verbose": yes, "count": 1, '
    '"mode": "fast"}}',
    '{"name": "fn_flags", "parameters": {"verbose": true, "count": 1.5, '
    '"mode": "fast"}}',
    '{"name": "fn_flags", "parameters": {"verbose": true, "count": 1, '
    '"mode": "medium"}}',
    '{"name": "fn_noargs", "parameters": {}} ',
])
def test_rejects_invalid_documents(doc: str) -> None:
    """Syntax or schema violations are refused."""
    assert not accepts(doc)


def test_forced_text_stops_on_shared_prefix() -> None:
    """``fn_`` is shared by several names, so it is not forced."""
    text, _ = forced_text(initial_state(SAMPLE_FUNCTIONS))
    assert text == '{"name": "'


def test_forced_text_finishes_unique_name() -> None:
    """Once the name is unambiguous, everything up to a value is forced."""
    state = run('{"name": "fn_ad')
    assert state is not None
    text, _ = forced_text(state)
    assert text == 'd_numbers", "parameters": {"a": '


def test_enum_prefix_options_need_a_decision() -> None:
    """``fast`` is a prefix of ``fast2``: the closing quote is not forced."""
    state = run('{"name": "fn_flags", "parameters": {"verbose": true, '
                '"count": 1, "mode": "fast')
    assert state is not None
    assert forced_text(state)[0] == ""


@pytest.mark.parametrize("prefix", [
    "", '{"name": "fn_', '{"name": "fn_nested", "parameters": {"user": {"na',
    '{"name": "fn_nested", "parameters": {"user": {"name": "ab\\',
    '{"name": "fn_nested", "parameters": {"user": {"name": "ab\\u00',
    '{"name": "fn_nested", "parameters": {"user": {"name": "a", "tags": '
    '["x", "y',
    '{"name": "fn_nested", "parameters": {"user": {"name": "a", "tags": []}'
    ', "points": [{"x": -',
    '{"name": "fn_nested", "parameters": {"user": {"name": "a", "tags": []}'
    ', "points": [], "matrix": [[1, 2], [3',
    '{"name": "fn_add_numbers", "parameters": {"a": 1.5e',
    '{"name": "fn_flags", "parameters": {"verbose": fa',
])
def test_complete_always_yields_valid_json(prefix: str) -> None:
    """Error recovery can close any partial document into a valid one."""
    state = run(prefix)
    assert state is not None
    doc = prefix + complete(state)
    data = json.loads(doc)
    fn = next(f for f in SAMPLE_FUNCTIONS if f.name == data["name"])
    assert matches_spec(data["parameters"], fn.as_object_spec())


def test_lenient_advance_repairs_invalid_escapes() -> None:
    """``\\d`` written by the model becomes the valid JSON ``\\\\d``."""
    state = run('{"name": "fn_greet", "parameters": {"name": "')
    assert state is not None
    fed = lenient_advance(state, "\\d+")
    assert fed is not None
    assert fed[1] == "\\\\d+"
    doc = '{"name": "fn_greet", "parameters": {"name": "' + fed[1] + '"}}'
    assert json.loads(doc)["parameters"]["name"] == "\\d+"


def test_lenient_advance_keeps_valid_escapes() -> None:
    """Valid escapes such as ``\\n`` are left untouched."""
    state = run('{"name": "fn_greet", "parameters": {"name": "')
    assert state is not None
    fed = lenient_advance(state, "a\\nb")
    assert fed is not None and fed[1] == "a\\nb"
