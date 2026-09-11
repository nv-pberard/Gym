# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import responses_api_agents.nooa_sandboxed_agent.app as app_module
import responses_api_agents.nooa_sandboxed_agent.runtime as runtime_module
from nemo_gym.base_resources_server import AggregateMetricsRequest
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import SandboxObservation
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.server_utils import ServerClient
from responses_api_agents.nooa_sandboxed_agent.app import (
    NOOASandboxedAgent,
    NOOASandboxedRunRequest,
    _is_scored,
    _sandbox_outcome,
)
from responses_api_agents.nooa_sandboxed_agent.config import NOOASandboxedAgentConfig
from responses_api_agents.nooa_sandboxed_agent.projection import project_result
from responses_api_agents.nooa_sandboxed_agent.protocol import NOOAModelEndpoint, NOOASandboxRequest, NOOASandboxResult
from responses_api_agents.nooa_sandboxed_agent.runtime import (
    AcquiredSandbox,
    SandboxExecution,
    _sandbox_spec,
    acquire_sandbox,
    execute_in_sandbox,
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
    assert request.expected_nooa_version == "0.0.10"


async def test_acquire_sandbox_prefers_full_descriptor_and_preserves_cleanup(monkeypatch) -> None:
    connected = SimpleNamespace()
    connect = AsyncMock(return_value=connected)
    monkeypatch.setattr(runtime_module, "create_provider", lambda _: "provider")
    monkeypatch.setattr(runtime_module.AsyncSandbox, "connect", connect)

    acquired = await acquire_sandbox(
        _config(),
        seed_response={
            "sandbox_descriptor": {"sandbox_id": "box", "workdir": "/app"},
            "sandbox_handle": "legacy-box",
            "cleanup_url_path": "/close_session",
        },
        resolved_provider={"fake": {}},
        default_metadata={},
    )

    connect.assert_awaited_once_with({"sandbox_id": "box", "workdir": "/app"}, provider="provider")
    assert acquired.sandbox is connected
    assert acquired.owned is False
    assert acquired.cleanup_url_path == "/close_session"


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


async def test_agent_release_calls_resources_cleanup_and_detaches(monkeypatch) -> None:
    cleanup_response = SimpleNamespace()
    server_client = MagicMock(spec=ServerClient)
    server_client.post = AsyncMock(return_value=cleanup_response)
    server = NOOASandboxedAgent(config=_config(), server_client=server_client)
    sandbox = SimpleNamespace(sandbox_id="box", stop=AsyncMock(), detach=AsyncMock())
    raise_for_status = AsyncMock()
    monkeypatch.setattr(app_module, "raise_for_status", raise_for_status)

    await server._release(
        AcquiredSandbox(sandbox=sandbox, owned=False, cleanup_url_path="/close_session"),
        {"session": "seeded"},
    )

    server_client.post.assert_awaited_once_with(
        server_name="resources",
        url_path="/close_session",
        json={},
        cookies={"session": "seeded"},
    )
    raise_for_status.assert_awaited_once_with(cleanup_response)
    sandbox.detach.assert_awaited_once_with()
    sandbox.stop.assert_not_awaited()


async def test_cleanup_failure_does_not_prevent_local_release() -> None:
    server_client = MagicMock(spec=ServerClient)
    server_client.post = AsyncMock(side_effect=RuntimeError("cleanup unavailable"))
    server = NOOASandboxedAgent(config=_config(), server_client=server_client)
    sandbox = SimpleNamespace(sandbox_id="box", stop=AsyncMock(), detach=AsyncMock())

    await server._release(
        AcquiredSandbox(sandbox=sandbox, owned=False, cleanup_url_path="/close_session"),
        {},
    )

    sandbox.detach.assert_awaited_once_with()


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


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"reward": 1.0, "nooa_status": "completed", "nooa_finished": True}, True),
        ({"reward": 0.0, "nooa_status": "failed", "nooa_finished": False}, False),
        ({"reward": 0.0, "score_valid": False}, False),
        ({"reward": 0.0, "mask_sample": True}, False),
        ({"reward": None}, False),
    ],
)
def test_is_scored_filters_harness_failures(row: dict, expected: bool) -> None:
    assert _is_scored(row) is expected


async def test_aggregate_metrics_filters_failures_and_reports_coverage(monkeypatch) -> None:
    server_client = MagicMock(spec=ServerClient)
    server_client.post = AsyncMock(return_value=SimpleNamespace())
    server = NOOASandboxedAgent(config=_config(), server_client=server_client)
    monkeypatch.setattr(app_module, "raise_for_status", AsyncMock())
    monkeypatch.setattr(
        app_module,
        "get_response_json",
        AsyncMock(return_value={"group_level_metrics": [], "agent_metrics": {}, "key_metrics": {}}),
    )
    body = AggregateMetricsRequest(
        verify_responses=[
            {"reward": 1.0, "nooa_status": "completed", "nooa_finished": True},
            {"reward": 0.0, "nooa_status": "failed", "nooa_finished": False},
        ]
    )

    metrics = await server.aggregate_metrics(body)

    forwarded = server_client.post.call_args.kwargs["json"]
    assert forwarded.verify_responses == [body.verify_responses[0]]
    assert metrics.agent_metrics == {
        "nooa/attempted": 2,
        "nooa/scored": 1,
        "nooa/excluded": 1,
        "nooa/coverage": 0.5,
    }


async def test_run_excludes_failed_nooa_attempt_from_scoring(monkeypatch) -> None:
    server_client = MagicMock(spec=ServerClient)
    server = NOOASandboxedAgent(config=_config(), server_client=server_client)
    body = NOOASandboxedRunRequest(responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="solve"))
    nooa_result = NOOASandboxResult(status="model_budget_exceeded", error="budget exhausted")
    sandbox_observation = SandboxObservation(
        role="agent",
        provider="fake",
        sandbox_id="box",
        outcome="failed",
        exit_code=2,
        wall_time_s=1.0,
    )
    agent_response, observations, trajectory = project_result(
        nooa_result,
        responses_create_params=body.responses_create_params,
        rollout_id="rollout",
        task_id="task",
        sandbox_observation=sandbox_observation,
    )
    execution = SandboxExecution(
        result=nooa_result,
        return_code=2,
        error_type=None,
        stdout="",
        stderr="",
        wall_time_s=1.0,
        artifacts_path="/results/run",
    )
    acquired = AcquiredSandbox(
        sandbox=SimpleNamespace(sandbox_id="box"),
        owned=False,
        cleanup_url_path="/close_session",
    )
    monkeypatch.setattr(
        NOOASandboxedAgent,
        "_execute",
        AsyncMock(return_value=(agent_response, observations, trajectory, execution, acquired, {"session": "seeded"})),
    )
    release = AsyncMock()
    monkeypatch.setattr(NOOASandboxedAgent, "_release", release)
    verified_data = body.model_dump(mode="json") | {
        "response": agent_response.model_dump(mode="json"),
        "reward": 0.0,
        "evaluation_completed": True,
        "resolved": False,
    }
    server_client.post = AsyncMock(return_value=SimpleNamespace())
    monkeypatch.setattr(app_module, "raise_for_status", AsyncMock())
    monkeypatch.setattr(app_module, "get_response_json", AsyncMock(return_value=verified_data))

    result = await server.run(SimpleNamespace(cookies={}), body)

    wire = result.model_dump(mode="json")
    assert result.nooa_finished is False
    assert result.score_valid is False
    assert result.failure_kind == "harness_error"
    assert result.verifier_reward == 0.0
    assert result.mask_sample is True
    assert wire["_ng_failure_class"] == "agent_run_error"
    assert "reward" not in wire
    assert "response" not in wire
    release.assert_awaited_once_with(acquired, {"session": "seeded"})


async def test_execute_uses_workdir_unique_root_and_persists_failure_artifacts(tmp_path: Path) -> None:
    sandbox = SimpleNamespace(
        sandbox_id="box",
        upload=AsyncMock(),
        download=AsyncMock(),
        exec=AsyncMock(
            side_effect=[
                SandboxExecResult(return_code=0, stdout="", stderr=""),
                SandboxExecResult(return_code=0, stdout="", stderr=""),
                SandboxExecResult(return_code=0, stdout="", stderr=""),
                SandboxExecResult(return_code=2, stdout="", stderr="runner failed"),
                SandboxExecResult(return_code=1, stdout="", stderr=""),
                SandboxExecResult(return_code=1, stdout="", stderr=""),
                SandboxExecResult(return_code=1, stdout="", stderr=""),
                SandboxExecResult(return_code=0, stdout="", stderr=""),
            ]
        ),
    )
    config = _config(
        runtime={"python": "/opt/nooa/bin/python", "workdir": "/app"},
        results_dir=str(tmp_path),
    )
    request = NOOASandboxRequest(
        rollout_id="rollout",
        task_id="task",
        agent_class="example.agent:ExampleAgent",
        entrypoint="solve",
        model=NOOAModelEndpoint(url="http://model/v1/responses", server_name="policy"),
        artifacts_dir="unused",
    )

    execution = await execute_in_sandbox(AcquiredSandbox(sandbox=sandbox, owned=True), request, config)

    runner_call = sandbox.exec.call_args_list[3]
    assert runner_call.kwargs["cwd"] == "/app"
    assert runner_call.args[0].startswith("/opt/nooa/bin/python /tmp/nemo-gym-nooa-")
    assert execution.result.status == "failed"
    assert execution.artifacts_path is not None
    artifacts = Path(execution.artifacts_path)
    assert (artifacts / "result.json").is_file()
    assert (artifacts / "execution.json").is_file()
