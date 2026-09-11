# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import SandboxObservation
from responses_api_agents.nooa_sandboxed_agent.projection import project_result
from responses_api_agents.nooa_sandboxed_agent.protocol import (
    NOOAInvocationArtifact,
    NOOAModelCallArtifact,
    NOOASandboxResult,
    NOOAToolArtifact,
)


def test_host_can_import_runner_without_importing_nooa() -> None:
    before = set(sys.modules)
    importlib.import_module("responses_api_agents.nooa_sandboxed_agent.app")
    importlib.import_module("responses_api_agents.nooa_sandboxed_agent.sandbox_runner")
    imported = set(sys.modules) - before

    assert "nooa" not in imported
    assert not any(name.startswith("nooa.") for name in imported)


def test_projection_preserves_model_and_resource_tool_evidence() -> None:
    model_response = {
        "id": "resp_model_1",
        "created_at": 1,
        "model": "policy",
        "object": "response",
        "output": [
            {
                "type": "function_call",
                "id": "code_1",
                "call_id": "code_1",
                "name": "execute_python",
                "arguments": '{"code":"return_value = 3"}',
                "status": "completed",
            },
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "done", "annotations": []}],
            },
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }
    result = NOOASandboxResult(
        status="completed",
        return_value="done",
        return_type="builtins:str",
        invocations=[
            NOOAInvocationArtifact(
                invocation_id="inv_1",
                status="completed",
                started_at=1.0,
                completed_at=2.0,
                duration_ms=1000,
            )
        ],
        model_calls=[
            NOOAModelCallArtifact(
                invocation_id="inv_1",
                model_server_name="policy_model",
                request={"input": "task"},
                response=model_response,
                started_at=1.1,
                completed_at=1.5,
                duration_ms=400,
            )
        ],
        tool_calls=[
            NOOAToolArtifact(
                kind="code",
                invocation_id="inv_1",
                tool_call_id="code_1",
                name="execute_python",
                arguments={"code": "return_value = 3"},
                output=3,
                status="completed",
                started_at=1.3,
                completed_at=1.4,
                duration_ms=100,
            ),
            NOOAToolArtifact(
                invocation_id="inv_1",
                tool_call_id="mcp_1",
                name="lookup",
                arguments={"key": "x"},
                output={"value": 3},
                status="completed",
                started_at=1.5,
                completed_at=1.8,
                duration_ms=300,
            ),
        ],
    )
    params = NeMoGymResponseCreateParamsNonStreaming(input="task", model="policy", tools=[])
    sandbox = SandboxObservation(
        role="agent",
        provider="fake",
        sandbox_id="box-1",
        outcome="completed",
        exit_code=0,
        wall_time_s=2.0,
    )

    response, observations, trajectory = project_result(
        result,
        responses_create_params=params,
        rollout_id="rollout-1",
        task_id="task-1",
        sandbox_observation=sandbox,
    )

    assert sum(getattr(item, "call_id", None) == "code_1" for item in response.output) == 2
    assert not any(getattr(item, "call_id", None) == "mcp_1" for item in response.output)
    assert any(getattr(item, "id", None) == "msg_1" for item in response.output)
    assert observations.records[-1] == sandbox
    assert trajectory.model_calls[0].response_metadata.response_id == "resp_model_1"
    assert {tool.tool_name: tool.output for tool in trajectory.tool_calls} == {
        "execute_python": 3,
        "lookup": {"value": 3},
    }
    assert trajectory.turns[0].model_calls[0].model_ref.name == "policy_model"


def test_projection_exposes_nooa_failure_on_response() -> None:
    result = NOOASandboxResult(status="model_budget_exceeded", error="NOOA rollout exceeded 2 model calls")
    params = NeMoGymResponseCreateParamsNonStreaming(input="task", model="policy")
    sandbox = SandboxObservation(
        role="agent",
        provider="fake",
        sandbox_id="box-1",
        outcome="failed",
        exit_code=2,
        wall_time_s=2.0,
    )

    response, observations, _ = project_result(
        result,
        responses_create_params=params,
        rollout_id="rollout-1",
        task_id="task-1",
        sandbox_observation=sandbox,
    )

    assert response.status == "failed"
    assert response.error is not None
    assert response.error.message == "NOOA rollout exceeded 2 model calls"
    assert response.metadata == {
        "nooa_status": "model_budget_exceeded",
        "nooa_error": "NOOA rollout exceeded 2 model calls",
    }
    assert response.output == []
    assert any(gap.code == "native_invocation_unavailable" for gap in observations.gaps)
