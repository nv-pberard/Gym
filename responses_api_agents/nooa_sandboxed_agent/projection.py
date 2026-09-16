# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project sandbox artifacts into Gym Responses and observability contracts."""

from __future__ import annotations

import json
import time
from typing import Any
from uuid import uuid4

from openai.types.responses import ResponseError

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    accumulate_response_usage,
)
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    ObservationGap,
    SandboxObservation,
    ToolCallObservation,
    TrajectoryModelCall,
    TrajectoryRecord,
    TrajectoryResponseMetadata,
    TrajectoryTokenStats,
    TrajectoryToolCall,
    TrajectoryTurn,
)
from responses_api_agents.nooa_sandboxed_agent.protocol import NOOASandboxResult


def _return_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _message_text(item: Any) -> str | None:
    if getattr(item, "type", None) != "message" or getattr(item, "role", None) != "assistant":
        return None
    return "\n".join(part.text for part in item.content if getattr(part, "type", None) == "output_text")


def _tool_output(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def project_result(
    result: NOOASandboxResult,
    *,
    responses_create_params: Any,
    rollout_id: str,
    task_id: str,
    sandbox_observation: SandboxObservation,
) -> tuple[NeMoGymResponse, AgentObservationBundle, TrajectoryRecord]:
    gaps = [ObservationGap(code=code) for code in result.observation_gaps]
    valid_model_responses: list[tuple[Any, NeMoGymResponse]] = []
    for call in result.model_calls:
        if call.response is None:
            continue
        try:
            valid_model_responses.append((call, NeMoGymResponse.model_validate(call.response)))
        except Exception as error:
            gaps.append(ObservationGap(code="model_response_invalid", detail=str(error)))

    timeline: list[tuple[float, int, list[Any]]] = []
    emitted_tool_call_ids: set[str] = set()
    usage = None
    for index, (call, response) in enumerate(valid_model_responses):
        usage = accumulate_response_usage(usage, response.usage)
        emitted_tool_call_ids.update(
            item.call_id for item in response.output if getattr(item, "type", None) == "function_call"
        )
        timeline.append((call.completed_at or call.started_at, index, list(response.output)))
    offset = len(timeline)
    for index, tool in enumerate(result.tool_calls):
        if tool.tool_call_id not in emitted_tool_call_ids:
            continue
        status = "completed" if tool.status == "completed" else "incomplete"
        items = [
            NeMoGymFunctionCallOutput(
                call_id=tool.tool_call_id,
                output=_tool_output(tool.output),
                status=status,
            )
        ]
        timeline.append(
            (
                tool.completed_at or tool.started_at or 0.0,
                offset + index,
                items,
            )
        )
    output = [item for _, _, items in sorted(timeline) for item in items]

    if result.status == "completed":
        terminal_text = _return_text(result.return_value)
        last_text = next((text for item in reversed(output) if (text := _message_text(item)) is not None), None)
        if last_text != terminal_text:
            output.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg_nooa_terminal_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(type="output_text", text=terminal_text, annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )
            gaps.append(ObservationGap(code="non_trainable_terminal_output"))
    if not output and result.status == "completed":
        output.append(
            NeMoGymResponseOutputMessage(
                id=f"msg_nooa_empty_{uuid4().hex}",
                content=[NeMoGymResponseOutputText(type="output_text", text="", annotations=[])],
                role="assistant",
                status="completed" if result.status == "completed" else "incomplete",
                type="message",
            )
        )

    budget_stop = result.status == "model_budget_exceeded" and result.budget_exhausted
    response_error = (
        ResponseError(code="server_error", message=result.error[:2000])
        if result.error is not None and not budget_stop
        else None
    )
    response = NeMoGymResponse(
        id=f"resp_nooa_{uuid4().hex}",
        created_at=int(time.time()),
        model=responses_create_params.model or "nooa",
        object="response",
        output=output,
        status="completed"
        if result.status == "completed"
        else "cancelled"
        if result.status == "cancelled"
        else "incomplete"
        if budget_stop
        else "failed",
        error=response_error,
        metadata={
            "nooa_status": result.status,
            "budget_exhausted": str(budget_stop).lower(),
            **({"stop_reason": result.stop_reason} if result.stop_reason is not None else {}),
            **({"nooa_error": result.error[:2000]} if result.error is not None else {}),
        },
        parallel_tool_calls=responses_create_params.parallel_tool_calls,
        tool_choice=responses_create_params.tool_choice,
        tools=responses_create_params.tools,
        usage=usage,
    )

    model_refs_by_invocation: dict[str, list[ModelCallRef]] = {}
    trajectory_model_calls: list[TrajectoryModelCall] = []
    turns: list[TrajectoryTurn] = []
    turn_counts: dict[str, int] = {}
    for call, model_response in valid_model_responses:
        model_ref = ModelServerRef(type="responses_api_models", name=call.model_server_name)
        reference = ModelCallRef(model_ref=model_ref, response_id=model_response.id)
        model_refs_by_invocation.setdefault(call.invocation_id, []).append(reference)
        response_usage = model_response.usage
        trajectory_model_calls.append(
            TrajectoryModelCall(
                model_call_id=model_response.id,
                started_at=call.started_at,
                completed_at=call.completed_at,
                duration_ms=call.duration_ms,
                request=call.request,
                response=call.response,
                response_metadata=TrajectoryResponseMetadata(
                    response_id=model_response.id,
                    model_ref=model_ref,
                    model=model_response.model,
                    response_status=getattr(model_response, "status", None),
                    error_category=call.error_type,
                ),
                token_stats=TrajectoryTokenStats(
                    prompt_tokens=response_usage.input_tokens if response_usage else None,
                    completion_tokens=response_usage.output_tokens if response_usage else None,
                    reasoning_tokens=(
                        response_usage.output_tokens_details.reasoning_tokens
                        if response_usage and response_usage.output_tokens_details
                        else None
                    ),
                    total_tokens=response_usage.total_tokens if response_usage else None,
                    cached_tokens=(
                        response_usage.input_tokens_details.cached_tokens
                        if response_usage and response_usage.input_tokens_details
                        else None
                    ),
                ),
            )
        )
        turn_no = turn_counts.get(call.invocation_id, 0) + 1
        turn_counts[call.invocation_id] = turn_no
        turns.append(
            TrajectoryTurn(
                invocation_id=call.invocation_id,
                task_id=task_id,
                rollout_id=rollout_id,
                turn_no=turn_no,
                timestamp=call.completed_at or call.started_at,
                answer=[item.model_dump(mode="json", exclude_none=True) for item in model_response.output],
                step_count=turn_no,
                model_calls=[reference],
            )
        )

    invocations = [
        AgentInvocation(
            invocation_id=invocation.invocation_id,
            parent_invocation_id=invocation.parent_invocation_id,
            status=invocation.status,
            duration_ms=invocation.duration_ms,
            error_type=invocation.error_type,
            model_calls=model_refs_by_invocation.get(invocation.invocation_id, []),
        )
        for invocation in result.invocations
    ]
    if not invocations:
        invocations = [
            AgentInvocation(
                invocation_id=rollout_id,
                status="completed" if result.status == "completed" else "failed",
                model_calls=model_refs_by_invocation.get("root", []),
            )
        ]
        gaps.append(ObservationGap(code="native_invocation_unavailable"))

    tool_observations = [
        ToolCallObservation(
            invocation_id=tool.invocation_id,
            tool_call_id=tool.tool_call_id,
            sandbox_id=sandbox_observation.sandbox_id,
            tool_name=tool.name,
            started_at=tool.started_at,
            completed_at=tool.completed_at,
            duration_ms=tool.duration_ms,
            timing_source="artifact",
            status=tool.status,
            error_type=tool.error_type,
        )
        for tool in result.tool_calls
    ]
    bundle = AgentObservationBundle(
        source="nooa_sandboxed",
        records=[*invocations, *tool_observations, sandbox_observation],
        gaps=gaps,
    )
    trajectory_tools = [
        TrajectoryToolCall(**observation.model_dump(exclude={"kind"}), output=tool.output)
        for observation, tool in zip(tool_observations, result.tool_calls, strict=True)
    ]
    trajectory = TrajectoryRecord(
        task_id=task_id,
        rollout_id=rollout_id,
        invocations=invocations,
        turns=turns,
        model_calls=trajectory_model_calls,
        tool_calls=trajectory_tools,
        gaps=gaps,
    )
    return response, bundle, trajectory
