# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the sandbox-only NOOA agent server."""

from __future__ import annotations

import keyword
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef


_RESERVED_AGENT_ATTRIBUTES = frozenset({"event_manager", "llm", "runtime"})


class NOOAArgumentBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    transform: str = "identity"

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        from responses_api_agents.nooa_sandboxed_agent.mapping import validate_source_path

        validate_source_path(value)
        return value

    @field_validator("transform")
    @classmethod
    def validate_transform(cls, value: str) -> str:
        from responses_api_agents.nooa_sandboxed_agent.mapping import get_transform

        get_transform(value)
        return value


class NOOAAgentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_class: str
    entrypoint: str
    init_kwargs: dict[str, Any] = Field(default_factory=dict)
    arguments: dict[str, NOOAArgumentBinding]
    tool_namespace: str = "resources"
    allowed_tools: list[str] = Field(default_factory=list)
    model_aliases: dict[str, str] = Field(default_factory=dict)
    source_archive: str | None = None
    pythonpath: list[str] = Field(default_factory=list)

    @field_validator("agent_class")
    @classmethod
    def validate_agent_class(cls, value: str) -> str:
        module, separator, name = value.partition(":")
        if not separator or not module or not name or "." in name:
            raise ValueError("agent_class must use the format 'module.path:ClassName'")
        return value

    @field_validator("entrypoint", "tool_namespace")
    @classmethod
    def validate_public_identifier(cls, value: str) -> str:
        if not value.isidentifier() or keyword.iskeyword(value) or value.startswith("_"):
            raise ValueError("value must be a public Python identifier")
        return value

    @model_validator(mode="after")
    def validate_names(self) -> "NOOAAgentSpec":
        invalid_arguments = sorted(
            name
            for name in self.arguments
            if not name.isidentifier() or keyword.iskeyword(name) or name.startswith("_") or name == "self"
        )
        invalid_aliases = sorted(
            name
            for name in self.model_aliases
            if not name.isidentifier() or keyword.iskeyword(name) or name.startswith("_")
        )
        invalid_tools = sorted(
            name
            for name in self.allowed_tools
            if not name.isidentifier() or keyword.iskeyword(name) or name.startswith("_")
        )
        if invalid_arguments:
            raise ValueError(f"argument names must be public Python identifiers: {invalid_arguments}")
        if invalid_aliases:
            raise ValueError(f"model_aliases keys must be public Python identifiers: {invalid_aliases}")
        if invalid_tools or len(self.allowed_tools) != len(set(self.allowed_tools)):
            raise ValueError(f"allowed_tools must be unique public Python identifiers: {invalid_tools}")
        if "llm" in self.init_kwargs:
            raise ValueError("init_kwargs.llm is reserved for Gym")
        reserved = sorted(
            ({self.tool_namespace, *self.model_aliases} & _RESERVED_AGENT_ATTRIBUTES)
            | ({self.tool_namespace} & set(self.model_aliases))
        )
        if reserved:
            raise ValueError(f"tool_namespace and model_aliases contain reserved or conflicting names: {reserved}")
        return self


class NOOARuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["baked", "archive"] = "baked"
    archive_path: str | None = None
    python: str = "/opt/nooa/bin/python"
    extract_dir: str = "/tmp/nemo-gym-nooa-runtime"
    workdir: str | None = None
    expected_nooa_version: str | None = "0.0.10"

    @field_validator("python", "extract_dir", "workdir")
    @classmethod
    def validate_absolute_sandbox_path(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.startswith("/") or value in {"/", "/tmp"}:
            raise ValueError("sandbox runtime paths must be absolute, non-root paths")
        return value

    @field_validator("expected_nooa_version")
    @classmethod
    def validate_expected_nooa_version(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("runtime.expected_nooa_version must be non-empty or null")
        return value

    @model_validator(mode="after")
    def validate_source(self) -> "NOOARuntimeConfig":
        if self.source == "archive" and not self.archive_path:
            raise ValueError("runtime.archive_path is required when runtime.source=archive")
        if self.source == "baked" and self.archive_path is not None:
            raise ValueError("runtime.archive_path is only valid when runtime.source=archive")
        return self


class NOOASandboxedAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    sandbox_provider: str | dict[str, Any] = "sandbox"
    sandbox_spec: dict[str, Any] = Field(default_factory=dict)
    sandbox_model_base_url: str | None = None
    sandbox_model_base_urls: dict[str, str] = Field(default_factory=dict)
    sandbox_resources_base_url: str | None = None
    seed_session_overrides: dict[str, Any] = Field(default_factory=lambda: {"create_pty": False})
    runtime: NOOARuntimeConfig = Field(default_factory=NOOARuntimeConfig)
    agent: NOOAAgentSpec
    max_model_calls: int = Field(default=10, gt=0)
    timeout_s: float = Field(default=2100, gt=0)
    concurrency: int = Field(default=8, gt=0)
    max_artifact_bytes: int = Field(default=64 * 1024 * 1024, gt=0)
    results_dir: str | None = "responses_api_agents/nooa_sandboxed_agent/results"
