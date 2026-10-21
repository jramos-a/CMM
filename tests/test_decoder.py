"""Tests for the constrained decoder with fake language models."""

from __future__ import annotations

import json

import pytest

from src.grammar import initial_state
from src.pipeline import FunctionCaller
from src.tokenizer import BPETokenizer
from tests.conftest import (
    SAMPLE_FUNCTIONS,
    FailingSDK,
    OracleSDK,
    RandomSDK,
    make_decoder,
    matches_spec,
)


def check_document(text: str) -> None:
    """Assert ``text`` is valid JSON matching one sample function."""
    data = json.loads(text)
    assert list(data) == ["name", "parameters"]
    fn = next(f for f in SAMPLE_FUNCTIONS if f.name == data["name"])
    assert matches_spec(data["parameters"], fn.as_object_spec())


@pytest.mark.parametrize("seed", range(40))
def test_random_logits_always_give_valid_json(
    tokenizer: BPETokenizer, seed: int
) -> None:
    """Whatever the model scores are, the output is schema-valid JSON."""
    decoder = make_decoder(RandomSDK(tokenizer.vocab_size, seed), tokenizer,
                           max_string_tokens=12, max_number_tokens=6)
    result = decoder.generate(tokenizer.encode("prompt"),
                              initial_state(SAMPLE_FUNCTIONS))
    assert not result.recovered
    check_document(result.text)


@pytest.mark.parametrize("target", [
    '{"name": "fn_add_numbers", "parameters": {"a": 40, "b": -2.5}}',
    '{"name": "fn_greet", "parameters": {"name": "José \\"Pepe\\""}}',
    '{"name": "fn_greet", "parameters": {"name": "日本 \U0001F600"}}',
    '{"name": "fn_flags", "parameters": {"verbose": true, "count": 12, '
    '"mode": "fast"}}',
    '{"name": "fn_nested", "parameters": {"user": {"name": "a", "tags": '
    '["x", "y"]}, "points": [{"x": 1, "y": 2}], "matrix": [[1], [2, 3]], '
    '"nothing": null}}',
    '{"name": "fn_noargs", "parameters": {}}',
])
def test_oracle_reproduces_target(tokenizer: BPETokenizer,
                                  target: str) -> None:
    """A model that 'wants' the target gets exactly the target."""
    decoder = make_decoder(OracleSDK(tokenizer, target), tokenizer,
                           trace=True)
    result = decoder.generate(tokenizer.encode("prompt"),
                              initial_state(SAMPLE_FUNCTIONS))
    assert result.text == target
    assert "".join(step.text for step in result.trace
                   if step.kind == "forced")
    assert any(step.kind == "model" for step in result.trace)


def test_model_failure_is_recovered(tokenizer: BPETokenizer) -> None:
    """A crashing model still yields a valid (completed) document."""
    decoder = make_decoder(FailingSDK(tokenizer.vocab_size, ok_calls=3),
                           tokenizer)
    result = decoder.generate(tokenizer.encode("prompt"),
                              initial_state(SAMPLE_FUNCTIONS))
    assert result.recovered
    assert result.error and "simulated" in result.error
    check_document(result.text)


def test_step_limit_is_recovered(tokenizer: BPETokenizer) -> None:
    """Hitting the step budget closes the document instead of failing."""
    decoder = make_decoder(RandomSDK(tokenizer.vocab_size, 3), tokenizer,
                           max_steps=2)
    result = decoder.generate(tokenizer.encode("prompt"),
                              initial_state(SAMPLE_FUNCTIONS))
    assert result.recovered
    check_document(result.text)


def test_string_length_is_capped(tokenizer: BPETokenizer) -> None:
    """A model that never closes a string is cut off."""
    target = '{"name": "fn_greet", "parameters": {"name": "' + "a" * 500
    decoder = make_decoder(OracleSDK(tokenizer, target), tokenizer,
                           max_string_tokens=20)
    result = decoder.generate(tokenizer.encode("prompt"),
                              initial_state(SAMPLE_FUNCTIONS))
    data = json.loads(result.text)
    assert data["parameters"]["name"] == "a" * 20


def test_logits_cache_avoids_recomputation(tokenizer: BPETokenizer) -> None:
    """Identical inputs are served from the cache."""
    decoder = make_decoder(RandomSDK(tokenizer.vocab_size), tokenizer)
    decoder.llm.logits([1, 2, 3])
    decoder.llm.logits([1, 2, 3])
    assert decoder.llm.calls == 1 and decoder.llm.cache_hits == 1


def test_pipeline_coerces_types(tokenizer: BPETokenizer) -> None:
    """``number`` values are floats in the result, integers stay ints."""
    target = ('{"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}')
    caller = FunctionCaller(
        decoder=make_decoder(OracleSDK(tokenizer, target), tokenizer),
        functions=SAMPLE_FUNCTIONS)
    outcome = caller.call("What is the sum of 2 and 3?")
    assert outcome.result.model_dump() == {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {"a": 2.0, "b": 3.0},
    }
    assert isinstance(outcome.result.parameters["a"], float)
