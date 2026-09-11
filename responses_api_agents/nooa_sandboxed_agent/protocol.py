# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned host/sandbox artifact contracts for the NOOA runner."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ARTIFACT_SCHEMA_VERSION: Literal["1.0"] = "1.0"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NOOAMCPConfig(StrictModel):
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    allowed_tools: list[str] = Field(default_factory=list)


class NOOAModelEndpoint(StrictModel):
    url: str
    model: str = "gym-policy-model"
    server_name: str
    headers: dict[str, str] = Field(default_factory=dict)


class NOOASandboxRequest(StrictModel):
    schema_version: Literal["1.0"] = ARTIFACT_SCHEMA_VERSION
    rollout_id: str
    task_id: str
    agent_class: str
    entrypoint: str
    init_kwargs: dict[str, Any] = Field(default_factory=dict)
    arguments: dict[str, Any] = Field(default_factory=dict)
    tool_namespace: str = "resources"
    model: NOOAModelEndpoint
    model_aliases: dict[str, NOOAModelEndpoint] = Field(default_factory=dict)
    mcp: NOOAMCPConfig | None = None
    max_model_calls: int = Field(default=10, gt=0)
    response_defaults: dict[str, Any] = Field(default_factory=dict)
    artifacts_dir: str


class NOOAInvocationArtifact(StrictModel):
    invocation_id: str
    parent_invocation_id: str | None = None
    method_name: str | None = None
    status: Literal["completed", "failed", "incomplete", "unknown"] = "unknown"
    started_at: float | None = None
    completed_at: float | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    error_type: str | None = None
    model_response_ids: list[str] = Field(default_factory=list)


class NOOAToolArtifact(StrictModel):
    kind: Literal["resource", "code", "nooa"] = "resource"
    invocation_id: str
    tool_call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    output: Any = None
    status: Literal["completed", "failed", "timeout", "cancelled", "incomplete", "unknown"] = "unknown"
    started_at: float | None = None
    completed_at: float | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    error_type: str | None = None


class NOOAModelCallArtifact(StrictModel):
    invocation_id: str
    model_server_name: str
    request: dict[str, Any]
    response: dict[str, Any] | None = None
    started_at: float
    completed_at: float | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    error_type: str | None = None


class NOOASandboxResult(StrictModel):
    schema_version: Literal["1.0"] = ARTIFACT_SCHEMA_VERSION
    status: Literal[
        "completed",
        "failed",
        "invalid_policy_output",
        "model_budget_exceeded",
        "cancelled",
    ]
    return_value: Any = None
    return_type: str | None = None
    error: str | None = None
    model_calls: list[NOOAModelCallArtifact] = Field(default_factory=list)
    tool_calls: list[NOOAToolArtifact] = Field(default_factory=list)
    invocations: list[NOOAInvocationArtifact] = Field(default_factory=list)
    observation_gaps: list[str] = Field(default_factory=list)


class NOOAEvent(StrictModel):
    schema_version: Literal["1.0"] = ARTIFACT_SCHEMA_VERSION
    type: str
    timestamp: float
    payload: dict[str, Any] = Field(default_factory=dict)
