# ABOUTME: Pydantic models describing function definitions and call results.
# ABOUTME: Also holds the tiny type-mapping helpers shared across the pipeline.

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

JsonType = Literal["number", "string", "boolean"]


class ParameterSpec(BaseModel):
    """A single parameter's declared type."""

    type: JsonType


class ReturnSpec(BaseModel):
    """Declared return type of a function (unused for generation, kept for parity)."""

    type: JsonType


class FunctionDefinition(BaseModel):
    """One entry of function_definitions.json."""

    name: str
    description: str = ""
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    returns: ReturnSpec | None = None

    @field_validator("name")
    @classmethod
    def _name_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("function name must not be empty")
        return value


class FunctionCallingTest(BaseModel):
    """One entry of function_calling_tests.json. Accepts either a bare string
    or an object with a 'prompt' key, since both shapes appear in the wild."""

    prompt: str


class FunctionCallResult(BaseModel):
    """One entry written to data/output/function_calling_results.json."""

    prompt: str
    fn_name: str
    args: dict[str, Any]
