"""Token-by-token constrained decoding driven by the grammar automaton.

At every step the decoder:

1. emits, without calling the model, any text that the schema fully
   determines (braces, keys, separators, the end of a function name...);
2. otherwise computes the set of vocabulary tokens whose characters are
   all accepted by the automaton;
3. sets the logits of every other token to ``-inf`` and picks the best
   remaining one (greedy decoding).

Because every emitted token keeps the automaton alive, and the automaton
only accepts schema-valid documents, the final text always parses.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict

from src.grammar import (
    NUMBER_CHARS,
    Stack,
    advance,
    advance_text,
    allowed_chars,
    complete,
    default_char,
    forced_text,
    has_function_name,
    is_in_free_string,
    lenient_advance,
    top_kind,
)
from src.llm import LLMClient, LLMError
from src.tokenizer import BPETokenizer
from src.vocab import TokenTable, utf8_feed

# Effect of a token: (next state, emitted text, pending UTF-8 bytes).
Move = tuple[Stack, str, bytes]


class DecodingError(RuntimeError):
    """Raised when constrained decoding cannot continue normally."""


class DecodingConfig(BaseModel):
    """Limits and options of the decoder.

    Attributes:
        max_string_tokens: Tokens allowed inside a single string value.
        max_number_tokens: Tokens allowed inside a single number value.
        max_steps: Model-driven steps allowed for one call.
        trace: Record a step-by-step trace (for visualisation).
        top_k: Alternatives kept per step in the trace.
    """

    max_string_tokens: int = 96
    max_number_tokens: int = 24
    max_steps: int = 400
    trace: bool = False
    top_k: int = 3


class StepTrace(BaseModel):
    """One generation step, as shown by the visualiser.

    Attributes:
        kind: ``forced`` (no model call), ``single`` (one valid token),
            ``model`` (chosen from the logits) or ``recovery``.
        text: Text appended at this step.
        candidates: Number of tokens that were allowed.
        probability: Probability of the chosen token among the allowed.
        alternatives: Best other allowed tokens with their probability.
    """

    kind: Literal["forced", "single", "model", "recovery"]
    text: str
    candidates: int = 1
    probability: Optional[float] = None
    alternatives: list[tuple[str, float]] = []


class DecodeResult(BaseModel):
    """Output of one constrained generation.

    Attributes:
        text: The generated JSON document.
        trace: Step-by-step trace (empty unless enabled).
        model_steps: Number of steps where the model was queried.
        recovered: Whether error recovery had to close the document.
        error: The error that triggered the recovery, if any.
    """

    text: str
    trace: list[StepTrace] = []
    model_steps: int = 0
    recovered: bool = False
    error: Optional[str] = None


class ConstrainedDecoder(BaseModel):
    """Generates schema-valid JSON with the LLM.

    Attributes:
        llm: Logits provider.
        tokenizer: Tokenizer used for forced text.
        table: Pre-computed vocabulary indexes.
        config: Decoding limits and options.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: LLMClient
    tokenizer: BPETokenizer
    table: TokenTable
    config: DecodingConfig = DecodingConfig()

    def token_label(self, token_id: int) -> str:
        """Return a printable form of a token (bytes for fragments)."""
        text = self.table.texts[token_id]
        if text is not None:
            return text
        raw = self.tokenizer.token_bytes(token_id) or b""
        return "".join(f"\\x{b:02x}" for b in raw)

    def _string_candidates(
        self, stack: Stack, pending: bytes
    ) -> tuple[np.ndarray, dict[int, Move]]:
        """Allowed tokens inside free string content.

        Fragments of multi-byte characters are buffered in ``pending``
        until the character is complete.
        """
        moves: dict[int, Move] = {}
        for token_id, raw in zip(self.table.partial_ids,
                                 self.table.partial_bytes):
            try:
                text, rest = utf8_feed(pending + raw)
            except UnicodeDecodeError:
                continue
            fed = lenient_advance(stack, text) if text else (stack, "")
            if fed is not None and is_in_free_string(fed[0]):
                moves[token_id] = (fed[0], fed[1], rest)
        if pending:
            return np.fromiter(moves, dtype=np.int64,
                               count=len(moves)), moves
        texts = self.table.texts
        for token_id in self.table.string_special_ids:
            fed = lenient_advance(stack, texts[token_id] or "")
            if fed is not None:
                moves[token_id] = (fed[0], fed[1], b"")
        extra = np.fromiter(moves, dtype=np.int64, count=len(moves))
        return np.concatenate([self.table.plain_string_ids, extra]), moves

    def candidates(
        self, stack: Stack, pending: bytes = b""
    ) -> tuple[np.ndarray, dict[int, Move]]:
        """Return the allowed token ids and what each of them does.

        Args:
            stack: Current automaton state.
            pending: Bytes of an unfinished multi-byte character.

        Returns:
            The allowed ids and, for each id present in the mapping, its
            ``Move``. Ids absent from the mapping are plain string tokens:
            the state is unchanged and their text is emitted as is.
        """
        if is_in_free_string(stack):
            return self._string_candidates(stack, pending)
        texts = self.table.texts
        moves: dict[int, Move] = {}
        chars = allowed_chars(stack)
        pool = self.table.all_ids() if chars is None \
            else self.table.ids_starting_with(chars)
        for token_id in pool:
            fed = lenient_advance(stack, texts[token_id] or "")
            if fed is not None:
                moves[token_id] = (fed[0], fed[1], b"")
        return np.fromiter(moves, dtype=np.int64, count=len(moves)), moves

    def _forced_step(self, stack: Stack) -> tuple[str, list[int], Stack, bool]:
        """Compute the text forced by the grammar from ``stack``.

        Token healing: if the forced text ends with a whitespace-only
        token (``"b": `` ends with a lone space), that token is held back
        so the model can merge it with what follows (`` -`` for a negative
        number), as the natural tokenization would.

        Returns:
            The text to emit, its token ids, the state after it, and
            whether a held-back token must now be chosen by the model.
        """
        forced, after = forced_text(stack)
        if not forced:
            return "", [], stack, False
        forced_ids = self.tokenizer.encode(forced)
        last = self.table.texts[forced_ids[-1]]
        if after and last and last.isspace():
            kept = forced[:len(forced) - len(last)]
            healed = advance_text(stack, kept)
            if healed is not None:
                return kept, forced_ids[:-1], healed, True
        return forced, forced_ids, after, False

    def _limit_number(self, cands: np.ndarray) -> np.ndarray:
        """Keep only tokens that end the current number, when possible."""
        texts = self.table.texts
        closing = [int(t) for t in cands
                   if (texts[int(t)] or "0")[0] not in NUMBER_CHARS]
        return np.asarray(closing, dtype=np.int64) if closing else cands

    def _pick(
        self, ids: list[int], cands: np.ndarray, trace: bool
    ) -> tuple[int, Optional[StepTrace]]:
        """Set the logits outside ``cands`` to -inf and pick the best."""
        logits = self.llm.logits(ids)
        cands = cands[cands < logits.shape[0]]
        if cands.size == 0:
            raise DecodingError("no allowed token fits the model vocabulary")
        masked = np.full(logits.shape, -np.inf, dtype=np.float32)
        masked[cands] = logits[cands]
        best = int(np.argmax(masked))
        if not trace:
            return best, None
        scores = logits[cands].astype(np.float64)
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        order = np.argsort(-probs)[: self.config.top_k + 1]
        alternatives = [(self.token_label(int(cands[i])), float(probs[i]))
                        for i in order if int(cands[i]) != best]
        return best, StepTrace(
            kind="model", text=self.token_label(best),
            candidates=int(cands.size),
            probability=float(probs[cands == best][0]),
            alternatives=alternatives[: self.config.top_k])

    def generate(
        self,
        prompt_ids: list[int],
        stack: Stack,
        reprompt: Optional[Callable[[str], list[int]]] = None,
    ) -> DecodeResult:
        """Generate a complete document starting from ``stack``.

        Args:
            prompt_ids: Encoded prompt fed to the model.
            stack: Initial automaton state.
            reprompt: Called once with the text generated so far as soon
                as the function name is fixed; returns a new (shorter)
                prompt used for the rest of the generation.

        Returns:
            The generated text and statistics. If anything goes wrong
            (model failure, step limit...), the partial output is closed
            with the shortest valid completion and ``recovered`` is set.
        """
        cfg = self.config
        ids = list(prompt_ids)
        generated_from = len(ids)
        parts: list[str] = []
        trace: list[StepTrace] = []
        pending = b""
        steps = 0
        run_kind = ""
        run_len = 0
        try:
            while stack:
                if reprompt is not None and not has_function_name(stack):
                    new_prompt = reprompt("".join(parts))
                    ids = new_prompt + ids[generated_from:]
                    generated_from = len(new_prompt)
                    reprompt = None
                forced, forced_ids, after, heal = self._forced_step(stack)
                if forced:
                    ids.extend(forced_ids)
                    parts.append(forced)
                    stack = after
                    run_kind, run_len = "", 0
                    if cfg.trace:
                        trace.append(StepTrace(kind="forced", text=forced))
                    if not heal:
                        continue
                kind = top_kind(stack)
                run_len = run_len + 1 if kind == run_kind else 1
                run_kind = kind
                if kind == "str" and run_len > cfg.max_string_tokens:
                    pending = b""
                    closing, stack = self._close_string(stack)
                    ids.extend(self.tokenizer.encode(closing))
                    parts.append(closing)
                    if cfg.trace:
                        trace.append(StepTrace(kind="recovery", text=closing))
                    continue
                cands, moves = self.candidates(stack, pending)
                if kind == "num" and run_len > cfg.max_number_tokens:
                    cands = self._limit_number(cands)
                if cands.size == 0:
                    raise DecodingError("no token satisfies the grammar")
                if cands.size == 1:
                    chosen = int(cands[0])
                    step = StepTrace(kind="single",
                                     text=self.token_label(chosen))
                else:
                    steps += 1
                    if steps > cfg.max_steps:
                        raise DecodingError("maximum number of steps reached")
                    chosen, model_step = self._pick(ids, cands, cfg.trace)
                    step = model_step or StepTrace(kind="model", text="")
                if cfg.trace:
                    trace.append(step)
                ids.append(chosen)
                stack, emitted, pending = moves.get(
                    chosen, (stack, self.table.texts[chosen] or "", b""))
                parts.append(emitted)
        except (DecodingError, LLMError) as exc:
            tail = complete(stack)
            parts.append(tail)
            if cfg.trace:
                trace.append(StepTrace(kind="recovery", text=tail))
            return DecodeResult(text="".join(parts), trace=trace,
                                model_steps=steps, recovered=True,
                                error=str(exc))
        return DecodeResult(text="".join(parts), trace=trace,
                            model_steps=steps)

    @staticmethod
    def _close_string(stack: Stack) -> tuple[str, Stack]:
        """Return the text that ends an over-long string value."""
        closing: list[str] = []
        while top_kind(stack) == "str":
            char = default_char(stack)
            nxt = advance(stack, char)
            if nxt is None:
                raise DecodingError("could not close the string value")
            closing.append(char)
            stack = nxt
        return "".join(closing), stack
