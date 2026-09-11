# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.nooa_sandboxed_agent.app import (
    NOOASandboxedAgent,
    NOOASandboxedRunRequest,
    _sandbox_outcome,
)
from responses_api_agents.nooa_sandboxed_agent.config import NOOASandboxedAgentConfig
from responses_api_agents.nooa_sandboxed_agent.protocol import NOOASandboxResult
from responses_api_agents.nooa_sandboxed_agent.runtime import (
    AcquiredSandbox,
    SandboxExecution,
    _sandbox_spec,
    release_sandbox,
)


def _config(**updates) -> NOOASandboxedAgentConfig:
    values = {
        "name": "nooa_sandboxed_agent",
        "host": "0.0.0.0",
        "port": 8000,
        "entrypoint": "app.py",
        "resources_server": {"type": "resources_servers", "name": "resources"},
        "model_server": {"type": "responses_api_models", "name": "policy_model"},
        "sandbox_provider": {"docker": {}},
        "sandbox_spec": {"image": "nemo-gym-nooa:latest"},
        "sandbox_model_base_url": "http://model.internal:8000",
        "sandbox_model_base_urls": {"reviewer_model": "http://reviewer.internal:8002"},
        "sandbox_resources_base_url": "http://resources.internal:8001",
        "agent": {
            "agent_class": "example.agent:ExampleAgent",
            "entrypoint": "solve",
            "arguments": {
                "task": {
                    "source": "responses_create_params.input",
                    "transform": "latest_user_text",
                }
            },
            "allowed_tools": ["lookup"],
        },
    }
    values.update(updates)
    return NOOASandboxedAgentConfig.model_validate(values)


def test_sandbox_request_uses_signed_mcp_metadata_and_explicit_argument_mapping() -> None:
    config = _config()
    server = NOOASandboxedAgent.model_construct(
        config=config,
        server_client=SimpleNamespace(global_config_dict={}),
    )
    body = NOOASandboxedRunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "look this up"}],
            model="policy",
        ),
        task_id="task-1",
    )

    request = server._sandbox_request(
        body,
        {
            "mcp": {
                "server_name": "resources",
                "url_path": "/mcp",
                "headers": {"X-NeMo-Gym-Session-Token": "signed"},
            }
        },
        rollout_id="rollout-1",
        task_id="task-1",
    )

    assert request.arguments == {"task": "look this up"}
    assert request.model.url == "http://model.internal:8000/v1/responses"
    assert request.mcp is not None
    assert request.mcp.url == "http://resources.internal:8001/mcp"
    assert request.mcp.headers == {"X-NeMo-Gym-Session-Token": "signed"}
    assert request.mcp.allowed_tools == ["lookup"]


def test_model_alias_uses_its_sandbox_reachable_override() -> None:
    config = _config(
        agent={
            "agent_class": "example.agent:ExampleAgent",
            "entrypoint": "solve",
            "arguments": {"task": {"source": "responses_create_params.input"}},
            "model_aliases": {"reviewer_llm": "reviewer_model"},
        }
    )
    server = NOOASandboxedAgent.model_construct(
        config=config,
        server_client=SimpleNamespace(global_config_dict={}),
    )
    body = NOOASandboxedRunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="review this", model="policy")
    )

    request = server._sandbox_request(body, {}, rollout_id="rollout-1", task_id="task-1")

    assert request.model_aliases["reviewer_llm"].url == "http://reviewer.internal:8002/v1/responses"


def test_sandbox_request_uses_external_harness_model_placeholder() -> None:
    config = _config(
        agent={
            "agent_class": "example.agent:ExampleAgent",
            "entrypoint": "solve",
            "arguments": {"task": {"source": "responses_create_params.input"}},
        }
    )
    server = NOOASandboxedAgent.model_construct(
        config=config,
        server_client=SimpleNamespace(global_config_dict={}),
    )
    body = NOOASandboxedRunRequest(responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="solve"))

    request = server._sandbox_request(body, {}, rollout_id="rollout-1", task_id="task-1")

    assert request.model.model == "gym-policy-model"


def test_sandbox_spec_rejects_unknown_fields() -> None:
    config = _config(sandbox_spec={"image": "nemo-gym-nooa:latest", "typo": True})

    with pytest.raises(ValueError, match="Unsupported sandbox_spec fields"):
        _sandbox_spec(config, {})


@pytest.mark.parametrize(("owned", "called"), [(True, "stop"), (False, "detach")])
async def test_release_sandbox_respects_ownership(owned: bool, called: str) -> None:
    sandbox = SimpleNamespace(stop=AsyncMock(), detach=AsyncMock())

    await release_sandbox(AcquiredSandbox(sandbox=sandbox, owned=owned))

    getattr(sandbox, called).assert_awaited_once_with()
    getattr(sandbox, "detach" if called == "stop" else "stop").assert_not_awaited()


def test_agent_failure_is_not_reported_as_sandbox_failure() -> None:
    execution = SandboxExecution(
        result=NOOASandboxResult(status="model_budget_exceeded", error="budget exhausted"),
        return_code=2,
        error_type=None,
        stdout="",
        stderr="",
        wall_time_s=1.0,
    )

    assert _sandbox_outcome(execution) == "failed"
