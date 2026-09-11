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

    nooa_status: str
    nooa_run_stdout: str = ""
    nooa_run_stderr: str = ""
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
            json=body.model_dump(mode="json"),
            cookies=cookies,
        )
        await raise_for_status(seed_response)
        cookies |= _cookie_dict(seed_response.cookies)
        seed_json = await get_response_json(seed_response)

        provider = resolve_provider_config(self.config.sandbox_provider, self.server_client.global_config_dict)
        metadata = resolve_provider_metadata(self.config.sandbox_provider, self.server_client.global_config_dict)
        acquired = await acquire_sandbox(
            self.config,
            seed_response=seed_json,
            resolved_provider=provider,
            default_metadata=metadata,
        )
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
            try:
                await release_sandbox(acquired)
            except Exception:
                LOG.exception("Failed to release NOOA sandbox after an execution error")
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
                await release_sandbox(acquired)

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
                    "nooa_run_stdout": execution.stdout,
                    "nooa_run_stderr": execution.stderr,
                    "ng_agent_observations": observations.model_dump(mode="json"),
                    "ng_trajectory": trajectory.model_dump(mode="json"),
                }
                return NOOASandboxedVerifyResponse.model_validate(result)
            finally:
                try:
                    await release_sandbox(acquired)
                except Exception:
                    LOG.exception("Failed to release NOOA sandbox %s", acquired.sandbox.sandbox_id)

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        if self.config.skip_verification:
            return await super().aggregate_metrics(body)
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    NOOASandboxedAgent.run_webserver()
