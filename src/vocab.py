"""Pre-computed views of the vocabulary used by the constrained decoder."""

from __future__ import annotations

import codecs

import numpy as np
from pydantic import BaseModel, ConfigDict

from src.tokenizer import BPETokenizer

REPLACEMENT_CHAR = "�"


def utf8_feed(data: bytes) -> tuple[str, bytes]:
    """Decode the complete characters of ``data``.

    Returns:
        The decoded text and the trailing bytes of an unfinished
        character.

    Raises:
        UnicodeDecodeError: If ``data`` is not a valid UTF-8 prefix.
    """
    decoder = codecs.getincrementaldecoder("utf-8")()
    text = decoder.decode(data, final=False)
    return text, decoder.getstate()[0]


class TokenTable(BaseModel):
    """Token id -> text map plus indexes that make masking fast.

    Attributes:
        texts: Decoded text of every id, ``None`` for special tokens and
            for tokens that are not valid UTF-8 on their own.
        first_char_index: First character -> ids of tokens starting with it.
        plain_string_ids: Tokens that are always valid inside a JSON
            string (no quote, no backslash, no control character).
        string_special_ids: Tokens with a quote or a backslash; they need
            to be checked against the automaton one by one.
        partial_ids: Tokens holding a fragment of a multi-byte character
            (only usable inside strings, combined with their neighbours).
        partial_bytes: Raw bytes of each ``partial_ids`` token.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    texts: list[str | None]
    first_char_index: dict[str, list[int]]
    plain_string_ids: np.ndarray
    string_special_ids: list[int]
    partial_ids: list[int]
    partial_bytes: list[bytes]

    @classmethod
    def build(cls, tokenizer: BPETokenizer) -> TokenTable:
        """Decode every token once and build the indexes."""
        texts: list[str | None] = []
        index: dict[str, list[int]] = {}
        plain: list[int] = []
        special: list[int] = []
        partial: list[int] = []
        partial_raw: list[bytes] = []
        for token_id in range(tokenizer.vocab_size):
            raw = None if tokenizer.is_special(token_id) \
                else tokenizer.token_bytes(token_id)
            text: str | None = None
            if raw:
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    partial.append(token_id)
                    partial_raw.append(raw)
            if text is not None and REPLACEMENT_CHAR in text:
                text = None
            texts.append(text)
            if text is None:
                continue
            index.setdefault(text[0], []).append(token_id)
            if '"' in text or "\\" in text:
                special.append(token_id)
            elif all(ord(c) >= 0x20 for c in text):
                plain.append(token_id)
        return cls(
            texts=texts,
            first_char_index=index,
            plain_string_ids=np.asarray(plain, dtype=np.int64),
            string_special_ids=special,
            partial_ids=partial,
            partial_bytes=partial_raw,
        )

    def ids_starting_with(self, chars: set[str]) -> list[int]:
        """Return the ids of all tokens whose first character is in chars."""
        out: list[int] = []
        for char in chars:
            out.extend(self.first_char_index.get(char, ()))
        return out

    def all_ids(self) -> list[int]:
        """Return every id that has a usable text."""
        return [i for i, text in enumerate(self.texts) if text is not None]
