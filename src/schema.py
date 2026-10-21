"""Pydantic models for function definitions, prompts and call results."""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic import model_validator

JsonType = Literal[
    "number", "integer", "string", "boolean", "null", "array", "object"
]
EnumValue = Union[str, int, float, bool]


class ParameterSpec(BaseModel):
    """Declared type of one parameter (possibly nested).

    Attributes:
        type: JSON type of the value.
        description: Optional human readable description.
        enum: Optional closed list of allowed values.
        items: Element spec, required when ``type`` is ``array``.
        properties: Member specs, required when ``type`` is ``object``.
    """

    model_config = ConfigDict(extra="ignore")

    type: JsonType
    description: Optional[str] = None
    enum: Optional[list[EnumValue]] = None
    items: Optional[ParameterSpec] = None
    properties: Optional[dict[str, ParameterSpec]] = None

    @model_validator(mode="after")
    def _check_composite(self) -> ParameterSpec:
        """Ensure composite types carry the information they need."""
        if self.type == "array" and self.items is None:
            raise ValueError("an 'array' parameter needs an 'items' spec")
        if self.type == "object" and self.properties is None:
            self.properties = {}
        if self.enum is not None and not self.enum:
            raise ValueError("'enum' must not be empty")
        return self


class ReturnSpec(BaseModel):
    """Declared return type of a function (informative only)."""

    model_config = ConfigDict(extra="ignore")

    type: str


class FunctionDefinition(BaseModel):
    """One entry of ``functions_definition.json``."""

    model_config = ConfigDict(extra="ignore")

    name: str
    description: str = ""
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    returns: Optional[ReturnSpec] = None

    @field_validator("name")
    @classmethod
    def _name_not_empty(cls, value: str) -> str:
        """Reject empty or blank function names."""
        if not value.strip():
            raise ValueError("function name must not be empty")
        return value

    def as_object_spec(self) -> ParameterSpec:
        """Return the parameters as a single ``object`` spec."""
        return ParameterSpec(type="object", properties=self.parameters)


class PromptEntry(BaseModel):
    """One entry of ``function_calling_tests.json``."""

    model_config = ConfigDict(extra="ignore")

    prompt: str


class FunctionCallResult(BaseModel):
    """One entry written to the output file."""

    prompt: str
    name: str
    parameters: dict[str, Any]
