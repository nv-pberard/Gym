# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe mappings from a Gym run row to sandboxed NOOA method arguments."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel


if TYPE_CHECKING:
    from responses_api_agents.nooa_sandboxed_agent.config import NOOAArgumentBinding


Transform = Callable[[Any], Any]
_FORBIDDEN_ROOTS = frozenset(
    {
        "agent_ref",
        "expected_answer",
        "gold",
        "ng_agent_observations",
        "ng_trajectory",
        "patch",
        "reference_answer",
        "response",
        "reward",
        "test_patch",
        "verifier_metadata",
    }
)


def validate_source_path(source: str) -> None:
    parts = source.split(".")
    if not source or any(not part or not (part.isidentifier() or part.isdigit()) for part in parts):
        raise ValueError("source must be a non-empty dotted path of identifiers or sequence indexes")
    if parts[0] in _FORBIDDEN_ROOTS or any(part.startswith("_") for part in parts):
        raise ValueError(f"mapping from Gym-internal or verifier field {parts[0]!r} is not allowed")


def resolve_source(row: Any, source: str) -> Any:
    validate_source_path(source)
    value = row
    for part in source.split("."):
        if isinstance(value, BaseModel):
            if not hasattr(value, part):
                raise ValueError(f"source {source!r} does not exist at {part!r}")
            value = getattr(value, part)
        elif isinstance(value, Mapping):
            if part not in value:
                raise ValueError(f"source {source!r} does not exist at {part!r}")
            value = value[part]
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and part.isdigit():
            try:
                value = value[int(part)]
            except IndexError as error:
                raise ValueError(f"source {source!r} has no index {part}") from error
        else:
            raise ValueError(f"source {source!r} cannot traverse {part!r}")
    return value


def identity(value: Any) -> Any:
    return value


def normalize_responses_input(value: Any) -> str | list[Any]:
    if isinstance(value, str):
        return value
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise ValueError("Responses input must be a string or a sequence")
    return [item.model_dump(exclude_none=True) if isinstance(item, BaseModel) else item for item in value]


def latest_user_text(value: Any) -> str:
    normalized = normalize_responses_input(value)
    if isinstance(normalized, str):
        return normalized
    for item in reversed(normalized):
        if not isinstance(item, Mapping) or item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
            parts = [
                part["text"]
                for part in content
                if isinstance(part, Mapping)
                and part.get("type") in {"input_text", "output_text"}
                and isinstance(part.get("text"), str)
            ]
            if parts:
                return "\n".join(parts)
        raise ValueError("latest user message does not contain text")
    raise ValueError("Responses input does not contain a user message")


TRANSFORMS: Mapping[str, Transform] = {
    "identity": identity,
    "latest_user_text": latest_user_text,
    "normalize_responses_input": normalize_responses_input,
}


def get_transform(name: str) -> Transform:
    try:
        return TRANSFORMS[name]
    except KeyError as error:
        raise ValueError(f"unknown transform {name!r}; available transforms: {sorted(TRANSFORMS)}") from error


def materialize_arguments(row: Any, bindings: Mapping[str, NOOAArgumentBinding]) -> dict[str, Any]:
    return {
        name: get_transform(binding.transform)(resolve_source(row, binding.source))
        for name, binding in bindings.items()
    }
