# ABOUTME: Token-by-token constrained decoding primitives used to force the
# ABOUTME: LLM's output into a valid, schema-conforming JSON shape.

from __future__ import annotations

import math
import re
from typing import Protocol

from src.vocab import VocabIndex

_NUMBER_CONTINUE_RE = re.compile(r"^-?[0-9]*\.?[0-9]*(?:[eE][+-]?[0-9]*)?$")
_NUMBER_COMPLETE_RE = re.compile(r"^-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$")

MAX_NUMBER_TOKENS = 12
MAX_STRING_TOKENS = 64


class LogitsProvider(Protocol):
    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]: ...


def _argmax_among(logits: list[float], candidate_ids: list[int]) -> int:
    """Return the candidate id with the highest logit. Ties broken by id order."""
    best_id = candidate_ids[0]
    best_score = -math.inf
    for token_id in candidate_ids:
        score = logits[token_id]
        if score > best_score:
            best_score = score
            best_id = token_id
    return best_id


def constrained_choice_decode(
    sdk: LogitsProvider,
    vocab_index: VocabIndex,
    prompt_ids: list[int],
    choices: list[str],
) -> str:
    """Generate text token-by-token, restricted at every step to tokens that
    keep the output a prefix of at least one entry of ``choices``. Stops as
    soon as the output exactly equals one (and only one) remaining choice,
    or when only one choice remains possible."""
    if not choices:
        raise ValueError("choices must not be empty")

    ids = list(prompt_ids)
    produced = ""
    active = list(choices)

    while True:
        active = [c for c in active if c.startswith(produced)]
        if not active:
            raise RuntimeError("constrained decoding reached a dead end")
        if produced in active and len(active) == 1:
            return produced

        candidate_ids: set[int] = set()
        for choice in active:
            remaining = choice[len(produced):]
            if not remaining:
                continue
            for token_id, _text in vocab_index.candidates_for_prefix_budget(remaining):
                candidate_ids.add(token_id)

        if produced and produced in choices and not candidate_ids:
            return produced
        if not candidate_ids:
            raise RuntimeError("no valid continuation tokens for constrained choice")

        logits = sdk.get_logits_from_input_ids(ids)
        chosen_id = _argmax_among(logits, sorted(candidate_ids))
        chosen_text = vocab_index.id_to_str.get(chosen_id, "")
        ids.append(chosen_id)
        produced += chosen_text

        if produced in choices:
            remaining_active = [c for c in active if c.startswith(produced) and c != produced]
            if not remaining_active:
                return produced


def _filtered_candidates(
    vocab_index: VocabIndex, pattern: re.Pattern[str], allow_empty: bool = False
) -> list[tuple[int, str]]:
    results = []
    for token_id, text in vocab_index.id_to_str.items():
        if text == "" and not allow_empty:
            continue
        if pattern.match(text):
            results.append((token_id, text))
    return results


def constrained_number_decode(
    sdk: LogitsProvider, vocab_index: VocabIndex, prompt_ids: list[int]
) -> str:
    """Generate a JSON number literal, token by token, only ever allowing
    tokens that keep the buffer a valid (possibly incomplete) number."""
    ids = list(prompt_ids)
    buffer = ""
    for _ in range(MAX_NUMBER_TOKENS):
        candidates = [
            (tid, text)
            for tid, text in _filtered_candidates(vocab_index, _NUMBER_CONTINUE_RE)
            if _NUMBER_CONTINUE_RE.match(buffer + text)
        ]
        if _NUMBER_COMPLETE_RE.match(buffer):
            # Model may legally choose "nothing more fits" implicitly by there
            # being no valid extending token; but if extensions exist we still
            # let the highest-scoring valid token decide whether to continue.
            if not candidates:
                return buffer
        if not candidates:
            if _NUMBER_COMPLETE_RE.match(buffer):
                return buffer
            raise RuntimeError(f"could not complete a valid number from '{buffer}'")

        logits = sdk.get_logits_from_input_ids(ids)
        candidate_ids = sorted({tid for tid, _ in candidates})
        chosen_id = _argmax_among(logits, candidate_ids)
        chosen_text = vocab_index.id_to_str.get(chosen_id, "")
        ids.append(chosen_id)
        buffer += chosen_text

        if _NUMBER_COMPLETE_RE.match(buffer):
            extendable = any(
                _NUMBER_CONTINUE_RE.match(buffer + text) and text
                for _, text in vocab_index.id_to_str.items()
            )
            if not extendable:
                return buffer

    if _NUMBER_COMPLETE_RE.match(buffer):
        return buffer
    raise RuntimeError(f"number generation exceeded token budget: '{buffer}'")


def constrained_string_decode(
    sdk: LogitsProvider,
    vocab_index: VocabIndex,
    prompt_ids: list[int],
    stop_token_str: str = '"',
) -> str:
    """Generate a JSON string's *content* (no surrounding quotes), token by
    token, stopping as soon as the model selects a token equal to the closing
    quote. Tokens containing an unescaped quote or a raw newline are excluded
    so the result always stays JSON-safe."""
    ids = list(prompt_ids)
    buffer = ""
    for _ in range(MAX_STRING_TOKENS):
        candidates = []
        for token_id, text in vocab_index.id_to_str.items():
            if text == "":
                continue
            if text == stop_token_str:
                candidates.append((token_id, text))
                continue
            if '"' in text or "\\" in text or "\n" in text:
                continue
            candidates.append((token_id, text))

        if not candidates:
            return buffer

        logits = sdk.get_logits_from_input_ids(ids)
        candidate_ids = sorted({tid for tid, _ in candidates})
        chosen_id = _argmax_among(logits, candidate_ids)
        chosen_text = vocab_index.id_to_str.get(chosen_id, "")
        ids.append(chosen_id)

        if chosen_text == stop_token_str:
            return buffer

        buffer += chosen_text

    return buffer
