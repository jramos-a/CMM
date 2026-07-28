# ABOUTME: Builds and caches the token-id <-> token-string index used to
# ABOUTME: restrict candidate tokens during constrained decoding.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class DecodesTokens(Protocol):
    """Structural type for the subset of the SDK we depend on here."""

    def decode(self, ids: list[int]) -> str: ...

    def get_path_to_tokenizer_file(self) -> str: ...


@dataclass
class TrieNode:
    """A single node of the token-string prefix trie.

    ``token_ids`` holds every vocabulary token whose decoded string equals the
    path from the root to this node (usually 0 or 1, occasionally more for
    tokenizers with duplicate pieces).
    """

    children: dict[str, "TrieNode"] = field(default_factory=dict)
    token_ids: list[int] = field(default_factory=list)


class VocabIndex:
    """Maps token ids to their decoded string form and exposes a prefix trie
    so that, given a partial output string, we can cheaply enumerate which
    vocabulary tokens are legal continuations."""

    def __init__(self, id_to_str: dict[int, str]) -> None:
        self.id_to_str = id_to_str
        self.root = TrieNode()
        for token_id, text in id_to_str.items():
            if text == "":
                continue
            node = self.root
            for char in text:
                node = node.children.setdefault(char, TrieNode())
            node.token_ids.append(token_id)

    def candidates_for_prefix_budget(self, allowed_prefix: str) -> list[tuple[int, str]]:
        """Return every (token_id, token_str) whose decoded string is a
        non-empty prefix-match walk starting at the trie root that stays
        inside ``allowed_prefix`` (i.e. the token string is itself a prefix
        of ``allowed_prefix``, possibly equal to it)."""
        results: list[tuple[int, str]] = []
        node = self.root
        text_so_far = ""
        for char in allowed_prefix:
            if char not in node.children:
                break
            node = node.children[char]
            text_so_far += char
            for token_id in node.token_ids:
                results.append((token_id, text_so_far))
        return results


def _cache_path(model_name: str) -> Path:
    safe_name = model_name.replace("/", "__")
    cache_dir = Path.home() / ".cache" / "function_calling_vocab"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{safe_name}.json"


def build_vocab_index(sdk: DecodesTokens, model_name: str, vocab_size: int) -> VocabIndex:
    """Build (or load from disk cache) the id->string map for the model.

    We rely purely on ``sdk.decode`` (a public, non-underscored method) so we
    never touch any private llm_sdk internals, per the project rules.
    """
    cache_file = _cache_path(model_name)
    if cache_file.is_file():
        try:
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
            if raw.get("vocab_size") == vocab_size:
                id_to_str = {int(k): v for k, v in raw["id_to_str"].items()}
                return VocabIndex(id_to_str)
        except (json.JSONDecodeError, KeyError, ValueError):
            pass  # fall through and rebuild

    built_id_to_str: dict[int, str] = {}
    for token_id in range(vocab_size):
        try:
            built_id_to_str[token_id] = sdk.decode([token_id])
        except Exception:  # noqa: BLE001 - a handful of ids may be invalid
            continue

    cache_file.write_text(
        json.dumps({"vocab_size": vocab_size, "id_to_str": built_id_to_str}),
        encoding="utf-8",
    )
    return VocabIndex(built_id_to_str)


def vocab_size_from_tokenizer_file(sdk: DecodesTokens) -> int:
    """Read vocab size out of tokenizer.json without touching private attrs."""
    tokenizer_path = Path(sdk.get_path_to_tokenizer_file())
    data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    vocab = data.get("model", {}).get("vocab", {})
    size = (max(vocab.values()) + 1) if vocab else 0
    for added in data.get("added_tokens", []):
        size = max(size, added.get("id", 0) + 1)
    return size
