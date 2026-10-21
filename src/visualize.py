"""Terminal visualisation of the constrained generation process."""

from __future__ import annotations

import sys
from typing import TextIO

from src.constrained_decoder import DecodeResult

_COLORS = {
    "forced": "\033[2m",
    "single": "\033[36m",
    "model": "\033[32m",
    "recovery": "\033[31m",
}
_RESET = "\033[0m"
_LABELS = {
    "forced": "forced ",
    "single": "single ",
    "model": "model  ",
    "recovery": "recover",
}


def _paint(text: str, kind: str, color: bool) -> str:
    """Wrap ``text`` in the ANSI color of ``kind`` when enabled."""
    return f"{_COLORS[kind]}{text}{_RESET}" if color else text


def render_trace(
    prompt: str, decoded: DecodeResult, stream: TextIO = sys.stderr
) -> None:
    """Print every generation step and the colored final output.

    Legend: dim = text forced by the schema (no model call), cyan = only
    one token was allowed, green = chosen by the model among the allowed
    tokens (with its probability and the best alternatives), red = error
    recovery.
    """
    color = stream.isatty()
    print(f"\n=== {prompt}", file=stream)
    for index, step in enumerate(decoded.trace):
        shown = repr(step.text)
        line = f"  {index:3d} {_LABELS[step.kind]} {shown:<28}"
        if step.kind == "model":
            alts = ", ".join(f"{t!r} {p:.0%}" for t, p in step.alternatives)
            line += (f" p={step.probability or 0:.0%} among "
                     f"{step.candidates} allowed | next best: {alts}")
        print(_paint(line, step.kind, color), file=stream)
    painted = "".join(_paint(s.text, s.kind, color) for s in decoded.trace)
    print(f"  => {painted}", file=stream)
    print(f"  model steps: {decoded.model_steps}"
          + (f" | recovered: {decoded.error}" if decoded.recovered else ""),
          file=stream)
