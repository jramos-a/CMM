"""Thin wrapper around ``llm_sdk`` with logits caching and retries."""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from src.tokenizer import BPETokenizer


class LLMError(RuntimeError):
    """Raised when the language model cannot produce logits."""


class LLMClient(BaseModel):
    """Logits provider backed by ``Small_LLM_Model``.

    Only the public SDK API is used. Each call recomputes the full forward
    pass (the SDK exposes no KV cache), so identical inputs are memoised in
    a small LRU cache.

    Attributes:
        sdk: Object exposing ``get_logits_from_input_ids``.
        cache_size: Maximum number of memoised logits vectors.
        retries: Extra attempts after a failed forward pass.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sdk: Any
    cache_size: int = 64
    retries: int = 1

    calls: int = 0
    cache_hits: int = 0
    seconds: float = 0.0

    _cache: OrderedDict[tuple[int, ...], np.ndarray] = PrivateAttr(
        default_factory=OrderedDict)

    @classmethod
    def load(cls, model_name: str) -> LLMClient:
        """Load ``model_name`` through ``llm_sdk``.

        Raises:
            LLMError: If the SDK or the model cannot be loaded.
        """
        try:
            from llm_sdk import Small_LLM_Model
        except ImportError as exc:
            raise LLMError(f"could not import llm_sdk: {exc}") from exc
        try:
            sdk = Small_LLM_Model(model_name=model_name)
        except Exception as exc:
            raise LLMError(
                f"could not load model {model_name!r}: {exc}") from exc
        return cls(sdk=sdk)

    def load_tokenizer(self) -> BPETokenizer:
        """Build our own tokenizer from the model's vocabulary files.

        Raises:
            LLMError: If the vocabulary files cannot be fetched or parsed.
        """
        try:
            vocab = self.sdk.get_path_to_vocab_file()
        except Exception as exc:
            raise LLMError(f"could not locate vocab file: {exc}") from exc
        merges = self._optional_path("get_path_to_merges_file")
        tok_file = self._optional_path("get_path_to_tokenizer_file")
        try:
            return BPETokenizer.from_files(vocab, merges, tok_file)
        except (OSError, ValueError, KeyError) as exc:
            raise LLMError(f"could not build the tokenizer: {exc}") from exc

    def _optional_path(self, method: str) -> str | None:
        """Call an optional ``get_path_to_*`` SDK method if it exists."""
        getter = getattr(self.sdk, method, None)
        if getter is None:
            return None
        try:
            return str(getter())
        except Exception:
            return None

    def logits(self, input_ids: list[int]) -> np.ndarray:
        """Return the next-token logits for ``input_ids``.

        Raises:
            LLMError: If the forward pass keeps failing after retries.
        """
        key = tuple(input_ids)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self.cache_hits += 1
            return cached
        last_error: Exception | None = None
        for _ in range(self.retries + 1):
            start = time.perf_counter()
            try:
                raw = self.sdk.get_logits_from_input_ids(list(input_ids))
                values = np.asarray(raw, dtype=np.float32).reshape(-1)
            except Exception as exc:
                last_error = exc
                continue
            finally:
                self.seconds += time.perf_counter() - start
            self.calls += 1
            self._cache[key] = values
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return values
        raise LLMError(f"forward pass failed: {last_error}")
