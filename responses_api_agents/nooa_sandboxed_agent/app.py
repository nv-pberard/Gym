# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run configured NOOA agents entirely inside a Gym sandbox."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from fastapi import Body, Request, Response
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    NEMO_GYM_MCP_METADATA_KEY,
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body
from nemo_gym.rollout_observability import (
    AgentObservationBundle,
    ObservationGap,
    SandboxObservation,
    TrajectoryRecord,
)
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.nooa_sandboxed_agent.config import NOOASandboxedAgentConfig
from responses_api_agents.nooa_sandboxed_agent.mapping import materialize_arguments
from responses_api_agents.nooa_sandboxed_agent.projection import project_result
from responses_api_agents.nooa_sandboxed_agent.protocol import (
    NOOAMCPConfig,
    NOOAModelEndpoint,
    NOOASandboxRequest,
)
from responses_api_agents.nooa_sandboxed_agent.runtime import (
    AcquiredSandbox,
    SandboxExecution,
    acquire_sandbox,
    execute_in_sandbox,
    release_sandbox,
)


LOG = logging.getLogger(__name__)


class NOOASandboxedRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class NOOASandboxedVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class NOOASandboxedVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    reward: float | None = Field(default=None, exclude_if=lambda value: value is None)
    response: NeMoGymResponse | None = Field(default=None, exclude_if=lambda value: value is None)
    nooa_status: str
    nooa_finished: bool
    nooa_results_path: str | None = None
    nooa_run_stdout: str = ""
    nooa_run_stderr: str = ""
    score_valid: bool
    verifier_reward: float | None = None
    failure_kind: str | None = None
    mask_sample: bool = False
    ng_agent_observations: AgentObservationBundle | None = Field(default=None)
    ng_trajectory: TrajectoryRecord | None = Field(default=None)


def _cookie_dict(cookies: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in cookies.items():
        result[str(key)] = str(getattr(value, "value", value))
    return result


def _task_id(body: BaseRunRequest) -> str:
    extra = body.model_extra or {}
    for key in ("task_id", "problem_id", "instance_id", "_ng_task_index"):
        if extra.get(key) is not None:
            return str(extra[key])
    return "unknown"


def _tool_names(params: NeMoGymResponseCreateParamsNonStreaming) -> set[str]:
    names: set[str] = set()
    for tool in params.tools or []:
        if isinstance(tool, Mapping):
            name = tool.get("name")
        else:
            name = getattr(tool, "name", None)
        if isinstance(name, str):
            names.add(name)
    return names


def _sandbox_outcome(execution: SandboxExecution) -> str:
    error = (execution.error_type or "").lower()
    if "timeout" in error:
        return "timeout"
    if "cancel" in error or execution.result.status == "cancelled":
        return "cancelled"
    if execution.error_type:
        return "sandbox_error"
    if execution.result.status == "completed":
        return "completed" if execution.return_code == 0 else "sandbox_error"
    if "result_artifact_unavailable" in execution.result.observation_gaps:
        return "sandbox_error"
    return "failed"


def _is_scored(row: Mapping[str, Any]) -> bool:
    reward = row.get("reward")
    return (
        not row.get("_ng_failure_class")
        and row.get("score_valid") is not False
        and row.get("mask_sample") is not True
        and row.get("nooa_finished") is not False
        and row.get("nooa_status") in {None, "completed"}
        and isinstance(reward, (int, float))
        and not isinstance(reward, bool)
    )


class NOOASandboxedAgent(SimpleResponsesAPIAgent):
    """Provider-neutral host orchestrator; it deliberately never imports NOOA."""

    config: NOOASandboxedAgentConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._semaphore = asyncio.Semaphore(self.config.concurrency)

    def _server_base_url(self, server_name: str) -> str:
        server_config = get_first_server_config_dict(self.server_client.global_config_dict, server_name)
        return self.server_client._build_server_base_url(server_config)

    def _model_endpoint(
        self,
        server_name: str,
        *,
        body: BaseRunRequest,
        model: str,
    ) -> NOOAModelEndpoint:
        if server_name in self.config.sandbox_model_base_urls:
            base_url = self.config.sandbox_model_base_urls[server_name]
        elif server_name == self.config.model_server.name and self.config.sandbox_model_base_url:
            base_url = self.config.sandbox_model_base_url
        else:
            base_url = self._server_base_url(server_name)
        return NOOAModelEndpoint(
            url=f"{self.base_url_for_run(base_url.rstrip('/'), body)}/v1/responses",
            model=model,
            server_name=server_name,
        )

    def _mcp_config(
        self,
        seed_response: dict[str, Any],
        params: NeMoGymResponseCreateParamsNonStreaming,
    ) -> NOOAMCPConfig | None:
        metadata = seed_response.get(NEMO_GYM_MCP_METADATA_KEY)
        if not isinstance(metadata, Mapping):
            if self.config.agent.allowed_tools:
                raise ValueError(
                    "The resources server did not expose signed MCP metadata, but agent.allowed_tools is non-empty"
                )
            return None

        declared = _tool_names(params)
        configured = set(self.config.agent.allowed_tools)
        allowed = configured & declared if declared else configured
        base_url = self.config.sandbox_resources_base_url or self._server_base_url(self.config.resources_server.name)
        url_path = str(metadata.get("url_path") or "/mcp")
        headers = metadata.get("headers") or {}
        if not isinstance(headers, Mapping):
            raise ValueError("MCP metadata headers must be an object")
        return NOOAMCPConfig(
            url=f"{base_url.rstrip('/')}/{url_path.lstrip('/')}",
            headers={str(key): str(value) for key, value in headers.items()},
            allowed_tools=sorted(allowed),
        )

    async def _cleanup_resources_session(self, cleanup_url_path: str | None, cookies: dict[str, str]) -> None:
        if not cleanup_url_path:
            return
        try:
            cleanup_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path=cleanup_url_path,
                json={},
                cookies=cookies,
            )
            await raise_for_status(cleanup_response)
        except Exception:
            LOG.exception("Failed to clean up NOOA Resources-server session")

    async def _release(self, acquired: AcquiredSandbox, cookies: dict[str, str]) -> None:
        await self._cleanup_resources_session(acquired.cleanup_url_path, cookies)
        try:
            await release_sandbox(acquired)
        except Exception:
            LOG.exception("Failed to release NOOA sandbox %s", acquired.sandbox.sandbox_id)

    def _sandbox_request(
        self,
        body: NOOASandboxedRunRequest,
        seed_response: dict[str, Any],
        *,
        rollout_id: str,
        task_id: str,
    ) -> NOOASandboxRequest:
        params = body.responses_create_params
        defaults = params.model_dump(mode="json", exclude_none=True)
        for key in ("input", "model", "tools"):
            defaults.pop(key, None)
        model_name = params.model or "gym-policy-model"
        aliases = {
            alias: self._model_endpoint(server_name, body=body, model=model_name)
            for alias, server_name in self.config.agent.model_aliases.items()
        }
        return NOOASandboxRequest(
            rollout_id=rollout_id,
            task_id=task_id,
            agent_class=self.config.agent.agent_class,
            entrypoint=self.config.agent.entrypoint,
            init_kwargs=self.config.agent.init_kwargs,
            arguments=materialize_arguments(body, self.config.agent.arguments),
            tool_namespace=self.config.agent.tool_namespace,
            model=self._model_endpoint(self.config.model_server.name, body=body, model=model_name),
            model_aliases=aliases,
            mcp=self._mcp_config(seed_response, params),
            max_model_calls=self.config.max_model_calls,
            expected_nooa_version=self.config.runtime.expected_nooa_version,
            response_defaults=defaults,
            artifacts_dir="/tmp/nemo-gym-nooa/artifacts",
        )

    async def _execute(
        self,
        body: NOOASandboxedRunRequest,
        *,
        cookies: dict[str, str],
    ) -> tuple[
        NeMoGymResponse, AgentObservationBundle, TrajectoryRecord, SandboxExecution, AcquiredSandbox, dict[str, str]
    ]:
        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(mode="json") | self.config.seed_session_overrides,
            cookies=cookies,
        )
        await raise_for_status(seed_response)
        cookies |= _cookie_dict(seed_response.cookies)
        seed_json = await get_response_json(seed_response)

        provider = resolve_provider_config(self.config.sandbox_provider, self.server_client.global_config_dict)
        metadata = resolve_provider_metadata(self.config.sandbox_provider, self.server_client.global_config_dict)
        try:
            acquired = await acquire_sandbox(
                self.config,
                seed_response=seed_json,
                resolved_provider=provider,
                default_metadata=metadata,
            )
        except BaseException:
            cleanup_url_path = seed_json.get("cleanup_url_path")
            if isinstance(cleanup_url_path, str) and cleanup_url_path.startswith("/"):
                await self._cleanup_resources_session(cleanup_url_path, cookies)
            raise
        try:
            rollout_id = maybe_rollout_id_from_run_body(body) or uuid4().hex
            task_id = _task_id(body)
            execution = await execute_in_sandbox(
                acquired,
                self._sandbox_request(body, seed_json, rollout_id=rollout_id, task_id=task_id),
                self.config,
            )
            sandbox_observation = SandboxObservation(
                role="agent",
                provider=acquired.sandbox.telemetry_provider_name,
                sandbox_id=acquired.sandbox.sandbox_id,
                outcome=_sandbox_outcome(execution),
                exit_code=execution.return_code,
                wall_time_s=execution.wall_time_s,
                error_type=execution.error_type,
            )
            response, observations, trajectory = project_result(
                execution.result,
                responses_create_params=body.responses_create_params,
                rollout_id=rollout_id,
                task_id=task_id,
                sandbox_observation=sandbox_observation,
            )
            return response, observations, trajectory, execution, acquired, cookies
        except BaseException:
            await self._release(acquired, cookies)
            raise

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        wrapper = NOOASandboxedRunRequest(responses_create_params=body)
        async with self._semaphore:
            agent_response, _, _, _, acquired, cookies = await self._execute(
                wrapper,
                cookies=dict(request.cookies),
            )
            try:
                for key, value in cookies.items():
                    response.set_cookie(key, value)
                return agent_response
            finally:
                await self._release(acquired, cookies)

    async def run(self, request: Request, body: NOOASandboxedRunRequest) -> NOOASandboxedVerifyResponse:
        async with self._semaphore:
            agent_response, observations, trajectory, execution, acquired, cookies = await self._execute(
                body,
                cookies=dict(request.cookies),
            )
            try:
                agent_json = agent_response.model_dump(mode="json")
                if self.config.skip_verification:
                    result = body.model_dump(mode="json") | {
                        "response": agent_json,
                        "reward": float(self.config.skip_verification_reward),
                        "verification_skipped": True,
                    }
                else:
                    verify_request = NOOASandboxedVerifyRequest.model_validate(
                        body.model_dump(mode="json") | {"response": agent_json}
                    )
                    verify_response = await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path="/verify",
                        json=verify_request.model_dump(mode="json"),
                        cookies=cookies,
                    )
                    await raise_for_status(verify_response)
                    result = await get_response_json(verify_response)

                nooa_finished = (
                    execution.result.status == "completed"
                    and agent_response.status == "completed"
                    and execution.return_code == 0
                    and execution.error_type is None
                )
                failure_kind = None
                failure_reason = None
                if not nooa_finished:
                    failure_kind = "harness_incomplete" if execution.result.status == "cancelled" else "harness_error"
                    failure_reason = (
                        execution.result.error
                        or execution.error_type
                        or f"NOOA response status: {agent_response.status}; return code: {execution.return_code}"
                    )
                elif result.get("evaluation_completed") is False:
                    failure_kind = "verification_error"
                    failure_reason = result.get("error") or "Verification did not complete"

                if failure_kind is not None:
                    verifier_reward = result.get("reward")
                    result.update(
                        reward=None,
                        response=None,
                        verifier_reward=verifier_reward if isinstance(verifier_reward, (int, float)) else None,
                        score_valid=False,
                        failure_kind=failure_kind,
                        failure_reason=failure_reason,
                        mask_sample=True,
                        _ng_failure_class="agent_run_error",
                        _ng_failure_message=failure_reason,
                    )
                else:
                    result.update(score_valid=True, mask_sample=False)

                resolved = result.get("resolved")
                if isinstance(resolved, bool) and trajectory.turns:
                    trajectory.turns[-1].resolved = resolved
                else:
                    trajectory.gaps.append(
                        ObservationGap(
                            code="resolution_unavailable",
                            invocation_id=trajectory.invocations[0].invocation_id,
                        )
                    )
                result |= {
                    "nooa_status": execution.result.status,
                    "nooa_finished": nooa_finished,
                    "nooa_results_path": execution.artifacts_path,
                    "nooa_run_stdout": execution.stdout,
                    "nooa_run_stderr": execution.stderr,
                    "ng_agent_observations": observations.model_dump(mode="json"),
                    "ng_trajectory": trajectory.model_dump(mode="json"),
                }
                return NOOASandboxedVerifyResponse.model_validate(result)
            finally:
                await self._release(acquired, cookies)

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        if self.config.skip_verification:
            return await super().aggregate_metrics(body)
        scored = [row for row in body.verify_responses if _is_scored(row)]
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body.model_copy(update={"verify_responses": scored}),
        )
        await raise_for_status(response)
        metrics = AggregateMetrics.model_validate(await get_response_json(response))
        attempted = len(body.verify_responses)
        metrics.agent_metrics.update(
            {
                "nooa/attempted": attempted,
                "nooa/scored": len(scored),
                "nooa/excluded": attempted - len(scored),
                "nooa/coverage": len(scored) / attempted if attempted else 0.0,
            }
        )
        return metrics


if __name__ == "__main__":
    NOOASandboxedAgent.run_webserver()
