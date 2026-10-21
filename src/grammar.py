"""Character-level pushdown automaton for schema-constrained JSON.

The automaton accepts exactly the JSON documents of the form::

    {"name": "<one function name>", "parameters": {<that function's args>}}

with a canonical layout (``", "`` and ``": "`` separators, no free
whitespace). Its state is an immutable stack of frames (plain tuples, so
copying a state while testing thousands of candidate tokens is cheap):

* ``("lit", text, pos)``          -- fixed text that must be emitted.
* ``("str", esc)``                -- free JSON string content.
* ``("enum", options, typed, quoted)`` -- one of a closed set of values.
* ``("fname", options, typed, functions)`` -- the function name; once it
  is closed the frame is replaced by that function's parameter schema.
* ``("num", dfa_state, integer)`` -- a JSON number (or integer).
* ``("arr", item_spec, count, max_items)`` -- a JSON array.

``advance`` feeds one character and returns the next state, or ``None``
when the character would break the syntax or the schema.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from src.schema import FunctionDefinition, ParameterSpec

Frame = tuple[Any, ...]
Stack = tuple[Frame, ...]

DIGITS = frozenset("0123456789")
HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
ESCAPABLE = frozenset('"\\/bfnrtu')
NUMBER_CHARS = frozenset("0123456789.eE+-")
NUM_ACCEPTING = frozenset({"zero", "int", "frac", "exp_digits"})
DEFAULT_MAX_ITEMS = 16


def _num_next(state: str, char: str, integer: bool) -> Optional[str]:
    """Transition function of the JSON number DFA."""
    digit = char in DIGITS
    if state in ("start", "minus"):
        if char == "-" and state == "start":
            return "minus"
        if char == "0":
            return "zero"
        return "int" if digit else None
    if state in ("zero", "int"):
        if digit and state == "int":
            return "int"
        if integer:
            return None
        if char == ".":
            return "dot"
        return "exp" if char in "eE" else None
    if state in ("dot", "frac"):
        if digit:
            return "frac"
        return "exp" if state == "frac" and char in "eE" else None
    if state == "exp" and char in "+-":
        return "exp_sign"
    return "exp_digits" if digit else None


def _num_chars(state: str, integer: bool) -> set[str]:
    """Characters accepted by the number DFA from ``state``."""
    return {c for c in NUMBER_CHARS if _num_next(state, c, integer)}


def _escape(text: str) -> str:
    """Return ``text`` JSON-escaped, without the surrounding quotes."""
    return json.dumps(text, ensure_ascii=False)[1:-1]


def expand(spec: ParameterSpec) -> list[Frame]:
    """Return the frames (bottom to top) that generate one value of spec."""
    if spec.enum is not None:
        if spec.type == "string":
            options = tuple(_escape(str(v)) for v in spec.enum)
            return [("enum", options, "", True), ("lit", '"', 0)]
        literals = tuple(json.dumps(v) for v in spec.enum)
        return [("enum", literals, "", False)]
    if spec.type == "string":
        return [("str", 0), ("lit", '"', 0)]
    if spec.type in ("number", "integer"):
        return [("num", "start", spec.type == "integer")]
    if spec.type == "boolean":
        return [("enum", ("true", "false"), "", False)]
    if spec.type == "null":
        return [("lit", "null", 0)]
    if spec.type == "array":
        return [("arr", spec.items, 0, DEFAULT_MAX_ITEMS), ("lit", "[", 0)]
    properties = spec.properties or {}
    if not properties:
        return [("lit", "{}", 0)]
    sequence: list[list[Frame]] = []
    for index, (key, sub_spec) in enumerate(properties.items()):
        prefix = "{" if index == 0 else ", "
        sequence.append([("lit", f'{prefix}"{_escape(key)}": ', 0)])
        sequence.append(expand(sub_spec))
    sequence.append([("lit", "}", 0)])
    frames: list[Frame] = []
    for item in reversed(sequence):
        frames.extend(item)
    return frames


def initial_state(functions: list[FunctionDefinition]) -> Stack:
    """Return the start state for a call to any of ``functions``."""
    by_name = {_escape(fn.name): fn for fn in functions}
    return (
        ("fname", tuple(by_name), "", by_name),
        ("lit", '{"name": "', 0),
    )


def _close_function_name(rest: Stack, fn: FunctionDefinition) -> Stack:
    """Replace the closed name frame by the function's parameters."""
    return (
        rest
        + (("lit", "}", 0),)
        + tuple(expand(fn.as_object_spec()))
        + (("lit", ', "parameters": ', 0),)
    )


def advance(stack: Stack, char: str) -> Optional[Stack]:
    """Feed one character to the automaton.

    Args:
        stack: Current state.
        char: Next output character.

    Returns:
        The new state, or ``None`` if ``char`` is not allowed here.
    """
    while stack:
        top = stack[-1]
        rest = stack[:-1]
        kind = top[0]
        if kind == "lit":
            text, pos = top[1], top[2]
            if text[pos] != char:
                return None
            if pos + 1 == len(text):
                return rest
            return rest + (("lit", text, pos + 1),)
        if kind == "str":
            esc = top[1]
            if esc == 0:
                if char == '"':
                    return rest
                if char == "\\":
                    return rest + (("str", -1),)
                return None if ord(char) < 0x20 else stack
            if esc == -1:
                if char not in ESCAPABLE:
                    return None
                return rest + (("str", 4 if char == "u" else 0),)
            if char not in HEX_DIGITS:
                return None
            return rest + (("str", esc - 1),)
        if kind in ("enum", "fname"):
            options, typed = top[1], top[2]
            quoted = kind == "fname" or top[3]
            if quoted and char == '"' and typed in options:
                if kind == "fname":
                    return _close_function_name(rest, top[3][typed])
                return rest
            extended = typed + char
            if any(opt.startswith(extended) for opt in options):
                return rest + ((kind, options, extended, top[3]),)
            if not quoted and typed in options:
                stack = rest
                continue
            return None
        if kind == "num":
            state, integer = top[1], top[2]
            nxt = _num_next(state, char, integer)
            if nxt is not None:
                return rest + (("num", nxt, integer),)
            if state in NUM_ACCEPTING:
                stack = rest
                continue
            return None
        if kind == "arr":
            item, count, max_items = top[1], top[2], top[3]
            if char == "]":
                return rest
            if count == 0:
                stack = rest + (("arr", item, 1, max_items),) + tuple(
                    expand(item))
                continue
            if char == "," and count < max_items:
                return (rest + (("arr", item, count + 1, max_items),)
                        + tuple(expand(item)) + (("lit", " ", 0),))
            return None
        raise ValueError(f"unknown frame kind: {kind!r}")
    return None


def advance_text(stack: Stack, text: str) -> Optional[Stack]:
    """Feed a whole string; return ``None`` as soon as a char is refused."""
    state: Optional[Stack] = stack
    for char in text:
        if state is None:
            return None
        state = advance(state, char)
    return state


def lenient_advance(stack: Stack, text: str) -> Optional[tuple[Stack, str]]:
    """Feed model text, repairing invalid escapes inside strings.

    Small models often write regexes like ``\\d+`` without doubling the
    backslash, which is not valid JSON. Instead of forbidding the token,
    the backslash is kept literally: ``\\d`` is emitted as ``\\\\d``.

    Returns:
        The new state and the (possibly repaired) emitted text, or
        ``None`` if the text cannot be accepted.
    """
    out: list[str] = []
    for char in text:
        nxt = advance(stack, char)
        if nxt is None and stack and stack[-1] == ("str", -1):
            repaired = advance(stack, "\\")
            nxt = advance(repaired, char) if repaired is not None else None
            out.append("\\")
        if nxt is None:
            return None
        out.append(char)
        stack = nxt
    return stack, "".join(out)


def allowed_chars(stack: Stack) -> Optional[set[str]]:
    """Return the set of characters accepted next.

    Returns:
        The allowed characters, or ``None`` when almost any character is
        allowed (inside free string content).
    """
    if not stack:
        return set()
    top = stack[-1]
    rest = stack[:-1]
    kind = top[0]
    if kind == "lit":
        return {top[1][top[2]]}
    if kind == "str":
        esc = top[1]
        if esc == 0:
            return None
        return set(ESCAPABLE) if esc == -1 else set(HEX_DIGITS)
    if kind in ("enum", "fname"):
        options, typed = top[1], top[2]
        quoted = kind == "fname" or top[3]
        chars = {opt[len(typed)] for opt in options
                 if opt.startswith(typed) and len(opt) > len(typed)}
        if typed in options:
            if quoted:
                chars.add('"')
            else:
                parent = allowed_chars(rest)
                if parent is None:
                    return None
                chars |= parent
        return chars
    if kind == "num":
        chars = _num_chars(top[1], top[2])
        if top[1] in NUM_ACCEPTING:
            parent = allowed_chars(rest)
            if parent is None:
                return None
            chars |= parent
        return chars
    if kind == "arr":
        item, count, max_items = top[1], top[2], top[3]
        if count == 0:
            first = allowed_chars(rest + (("arr", item, 1, max_items),)
                                  + tuple(expand(item)))
            return None if first is None else first | {"]"}
        return {"]", ","} if count < max_items else {"]"}
    raise ValueError(f"unknown frame kind: {kind!r}")


def forced_text(stack: Stack) -> tuple[str, Stack]:
    """Return the text that is fully determined from ``stack`` onwards.

    A common prefix shared by several enum options (such as ``fn_``) is
    *not* forced: forcing it would split it into unnatural tokens and
    mislead the model. Only the end of the last remaining option is.

    Returns:
        The forced characters and the state reached after them.
    """
    out: list[str] = []
    while stack:
        top = stack[-1]
        if top[0] in ("enum", "fname") and sum(
                opt.startswith(top[2]) for opt in top[1]) > 1:
            break
        chars = allowed_chars(stack)
        if chars is None or len(chars) != 1:
            break
        char = next(iter(chars))
        nxt = advance(stack, char)
        if nxt is None:
            break
        out.append(char)
        stack = nxt
    return "".join(out), stack


def default_char(stack: Stack) -> str:
    """Pick the character that leads to the shortest valid completion."""
    top = stack[-1]
    kind = top[0]
    if kind == "lit":
        return str(top[1][top[2]])
    if kind == "str":
        return '"' if top[1] == 0 else ("n" if top[1] == -1 else "0")
    if kind in ("enum", "fname"):
        options, typed = top[1], top[2]
        quoted = kind == "fname" or top[3]
        if typed in options:
            return '"' if quoted else default_char(stack[:-1])
        for opt in options:
            if opt.startswith(typed):
                return str(opt[len(typed)])
    if kind == "num":
        if top[1] in NUM_ACCEPTING:
            return default_char(stack[:-1])
        return "0"
    return "]"


def complete(stack: Stack) -> str:
    """Return a short text that brings ``stack`` to a complete document.

    Used for error recovery: whatever happened during generation, the
    partial output can always be closed into schema-valid JSON.
    """
    out: list[str] = []
    while stack:
        char = default_char(stack)
        nxt = advance(stack, char)
        if nxt is None:
            raise RuntimeError("grammar completion reached a dead end")
        out.append(char)
        stack = nxt
    return "".join(out)


def is_in_free_string(stack: Stack) -> bool:
    """Return whether the next char is plain JSON string content."""
    return bool(stack) and stack[-1][0] == "str" and stack[-1][1] == 0


def has_function_name(stack: Stack) -> bool:
    """Return whether the function name is still to be generated."""
    return any(frame[0] == "fname" for frame in stack)


def top_kind(stack: Stack) -> str:
    """Return the kind of the top frame (``""`` when complete)."""
    return str(stack[-1][0]) if stack else ""
