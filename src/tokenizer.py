"""Byte-level BPE tokenizer rebuilt from the model's vocabulary files.

This replaces ``Small_LLM_Model.encode``/``decode``: it only needs the
files returned by the public ``get_path_to_*_file`` methods of the SDK.
The pre-tokenizer reproduces the Qwen2/Qwen3 split regex by hand, so the
standard library is enough (no ``regex`` / ``tokenizers`` dependency).
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, PrivateAttr

_CONTRACTIONS_2 = ("re", "ve", "ll")
_CONTRACTIONS_1 = ("s", "t", "m", "d")


def bytes_to_unicode() -> dict[int, str]:
    """Return the GPT-2 byte -> printable unicode character table."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codes = printable[:]
    shift = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            codes.append(256 + shift)
            shift += 1
    return {b: chr(c) for b, c in zip(printable, codes)}


def _is_letter(char: str) -> bool:
    """Return whether ``char`` is in the unicode ``L`` category."""
    return unicodedata.category(char)[0] == "L"


def _is_number(char: str) -> bool:
    """Return whether ``char`` is in the unicode ``N`` category."""
    return unicodedata.category(char)[0] == "N"


def _is_other(char: str) -> bool:
    """Return whether ``char`` is neither space, letter nor number."""
    return not (char.isspace() or _is_letter(char) or _is_number(char))


def _match_at(text: str, i: int) -> int:
    """Return the end of the pre-token starting at ``i``.

    Hand-written equivalent of the Qwen2 split pattern::

        (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}
        | ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+

    Alternatives are tried in order, exactly like the regex engine does.
    """
    n = len(text)
    char = text[i]
    if char == "'":
        two = text[i + 1:i + 3].lower()
        if two in _CONTRACTIONS_2:
            return i + 3
        if text[i + 1:i + 2].lower() in _CONTRACTIONS_1:
            return i + 2
    if _is_letter(char) or (
        char not in "\r\n" and not _is_number(char)
        and i + 1 < n and _is_letter(text[i + 1])
    ):
        j = i + 1
        while j < n and _is_letter(text[j]):
            j += 1
        return j
    if _is_number(char):
        return i + 1
    start = i + 1 if char == " " and i + 1 < n and _is_other(text[i + 1]) \
        else i
    if _is_other(text[start]):
        j = start
        while j < n and _is_other(text[j]):
            j += 1
        while j < n and text[j] in "\r\n":
            j += 1
        return j
    j = i
    last_newline = -1
    while j < n and text[j].isspace():
        if text[j] in "\r\n":
            last_newline = j
        j += 1
    if last_newline >= 0:
        return last_newline + 1
    if j == n or j - 1 == i:
        return j
    return j - 1


def pre_tokenize(text: str) -> list[str]:
    """Split ``text`` into pre-tokens the way the Qwen tokenizer does."""
    pieces: list[str] = []
    i = 0
    while i < len(text):
        end = _match_at(text, i)
        pieces.append(text[i:end])
        i = end
    return pieces


class BPETokenizer(BaseModel):
    """Byte-level BPE tokenizer with public ``encode``/``decode``.

    Attributes:
        encoder: Token string (byte-level alphabet) -> id.
        merge_ranks: Merge pair -> priority (lower merges first).
        special_tokens: Added/special token content -> id.
    """

    encoder: dict[str, int]
    merge_ranks: dict[tuple[str, str], int]
    special_tokens: dict[str, int] = {}

    _decoder: dict[int, str] = PrivateAttr(default_factory=dict)
    _special_by_id: dict[int, str] = PrivateAttr(default_factory=dict)
    _byte_encoder: dict[int, str] = PrivateAttr(default_factory=dict)
    _byte_decoder: dict[str, int] = PrivateAttr(default_factory=dict)
    _cache: dict[str, list[int]] = PrivateAttr(default_factory=dict)
    _special_re: Optional[re.Pattern[str]] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Build the reverse tables once the fields are validated."""
        self._decoder = {i: s for s, i in self.encoder.items()}
        self._special_by_id = {i: s for s, i in self.special_tokens.items()}
        self._byte_encoder = bytes_to_unicode()
        self._byte_decoder = {c: b for b, c in self._byte_encoder.items()}
        if self.special_tokens:
            ordered = sorted(self.special_tokens, key=len, reverse=True)
            self._special_re = re.compile(
                "(" + "|".join(re.escape(tok) for tok in ordered) + ")"
            )

    @classmethod
    def from_files(
        cls,
        vocab_path: str | Path,
        merges_path: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
    ) -> BPETokenizer:
        """Build a tokenizer from ``vocab.json`` and friends.

        Args:
            vocab_path: ``vocab.json`` (token -> id).
            merges_path: ``merges.txt``; if missing, the merges are read
                from ``tokenizer.json``.
            tokenizer_path: ``tokenizer.json``, used for added tokens.

        Returns:
            A ready to use tokenizer.
        """
        with Path(vocab_path).open("r", encoding="utf-8") as handle:
            encoder: dict[str, int] = json.load(handle)
        tok_data: dict[str, Any] = {}
        if tokenizer_path is not None:
            with Path(tokenizer_path).open("r", encoding="utf-8") as handle:
                tok_data = json.load(handle)
        merges: list[tuple[str, str]] = []
        if merges_path is not None:
            with Path(merges_path).open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.rstrip("\n")
                    if not line or line.startswith("#version"):
                        continue
                    left, _, right = line.partition(" ")
                    merges.append((left, right))
        else:
            for merge in tok_data.get("model", {}).get("merges", []):
                if isinstance(merge, str):
                    left, _, right = merge.partition(" ")
                    merges.append((left, right))
                else:
                    merges.append((merge[0], merge[1]))
        special = {
            entry["content"]: int(entry["id"])
            for entry in tok_data.get("added_tokens", [])
        }
        return cls(
            encoder=encoder,
            merge_ranks={pair: rank for rank, pair in enumerate(merges)},
            special_tokens=special,
        )

    @property
    def vocab_size(self) -> int:
        """Number of ids covered by the vocabulary and added tokens."""
        top = max(self._decoder) if self._decoder else -1
        if self._special_by_id:
            top = max(top, max(self._special_by_id))
        return top + 1

    def _bpe(self, piece: str) -> list[int]:
        """Apply the BPE merges to one pre-token and return its ids."""
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        word = [self._byte_encoder[b] for b in piece.encode("utf-8")]
        ranks = self.merge_ranks
        while len(word) > 1:
            best_rank = -1
            best_pos = -1
            for pos in range(len(word) - 1):
                rank = ranks.get((word[pos], word[pos + 1]))
                if rank is not None and (best_rank < 0 or rank < best_rank):
                    best_rank = rank
                    best_pos = pos
            if best_pos < 0:
                break
            pair = (word[best_pos], word[best_pos + 1])
            merged: list[str] = []
            pos = 0
            while pos < len(word):
                if pos < len(word) - 1 and (word[pos], word[pos + 1]) == pair:
                    merged.append(word[pos] + word[pos + 1])
                    pos += 2
                else:
                    merged.append(word[pos])
                    pos += 1
            word = merged
        ids = [self.encoder[symbol] for symbol in word]
        self._cache[piece] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode ``text`` into token ids (no special tokens are added).

        Added tokens such as ``<|im_start|>`` written in ``text`` are kept
        atomic, like the reference tokenizer does.
        """
        ids: list[int] = []
        chunks = self._special_re.split(text) if self._special_re else [text]
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.special_tokens:
                ids.append(self.special_tokens[chunk])
                continue
            for piece in pre_tokenize(unicodedata.normalize("NFC", chunk)):
                ids.extend(self._bpe(piece))
        return ids

    def token_bytes(self, token_id: int) -> Optional[bytes]:
        """Return the raw bytes of a regular token, or ``None``."""
        symbol = self._decoder.get(token_id)
        if symbol is None:
            return None
        return bytes(self._byte_decoder[c] for c in symbol)

    def is_special(self, token_id: int) -> bool:
        """Return whether ``token_id`` is an added/special token."""
        return token_id in self._special_by_id

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode token ids back into text.

        Args:
            ids: Token ids to decode.
            skip_special: Drop added/special tokens from the output.

        Returns:
            The decoded text (invalid UTF-8 is replaced by U+FFFD).
        """
        out = bytearray()
        for token_id in ids:
            special = self._special_by_id.get(token_id)
            if special is not None:
                if not skip_special:
                    out.extend(special.encode("utf-8"))
                continue
            raw = self.token_bytes(token_id)
            if raw is not None:
                out.extend(raw)
        return out.decode("utf-8", errors="replace")
