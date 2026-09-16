# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint staged into a sandbox to execute one NOOA rollout.

This module intentionally imports NOOA only inside ``run``. The host agent
server stages the file as an executable artifact and never imports NOOA.
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import importlib.metadata
import inspect
import json
import os
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"


class ModelCallBudgetExceeded(RuntimeError):
    pass


class InvalidPolicyOutput(RuntimeError):
    pass


def _json_value(value: Any) -> Any:
    from pydantic import BaseModel

    if isinstance(value, BaseModel):
        return _json_value(value.model_dump(mode="python"))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_value(dataclasses.asdict(value))
    if isinstance(value, tuple):
        return [_artifact_value(item) for item in value]
    if isinstance(value, list):
        return [_artifact_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _artifact_value(item) for key, item in value.items()}
    json.dumps(value)
    return value


def _artifact_value(value: Any) -> Any:
    try:
        return _json_value(value)
    except Exception:
        try:
            rendered = repr(value)[:2000]
        except Exception:
            rendered = "<repr unavailable>"
        return {
            "unserializable_type": _qualified_type(value),
            "repr": rendered,
        }


def _qualified_type(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}:{value_type.__qualname__}"


class ArtifactRecorder:
    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir
        self.events_path = artifacts_dir / "events.jsonl"
        self.invocations: dict[str, dict[str, Any]] = {}
        self.model_calls: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.current_invocation: ContextVar[str] = ContextVar("nooa_sandbox_invocation", default="root")

    def event(self, event_type: str, **payload: Any) -> None:
        record = {
            "schema_version": SCHEMA_VERSION,
            "type": event_type,
            "timestamp": time.time(),
            "payload": _artifact_value(payload),
        }
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")

    def before_agent_call(
        self,
        agent: Any,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        call_id: str,
        parent_call_id: str | None,
        **extra_kwargs: Any,
    ) -> dict[str, Any]:
        del agent, args, kwargs, extra_kwargs
        started_at = time.time()
        context = {
            "invocation_id": call_id,
            "started_at": started_at,
            "started_monotonic": time.perf_counter(),
            "token": self.current_invocation.set(call_id),
        }
        self.invocations[call_id] = {
            "invocation_id": call_id,
            "parent_invocation_id": parent_call_id,
            "method_name": method_name,
            "status": "unknown",
            "started_at": started_at,
            "model_response_ids": [],
        }
        self.event(
            "agent_call_started",
            invocation_id=call_id,
            parent_invocation_id=parent_call_id,
            method_name=method_name,
        )
        return context

    def after_agent_call(
        self,
        agent: Any,
        method_name: str,
        result: Any,
        exception: BaseException | None,
        context: Any,
        **kwargs: Any,
    ) -> None:
        del agent, method_name, result, kwargs
        if not isinstance(context, dict):
            return
        self.current_invocation.reset(context["token"])
        completed_at = time.time()
        invocation_id = context["invocation_id"]
        record = self.invocations[invocation_id]
        record.update(
            {
                "status": "failed" if exception is not None else "completed",
                "completed_at": completed_at,
                "duration_ms": max(0.0, (time.perf_counter() - context["started_monotonic"]) * 1000),
                "error_type": type(exception).__name__ if exception is not None else None,
            }
        )
        self.event(
            "agent_call_finished",
            invocation_id=invocation_id,
            status=record["status"],
            error_type=record["error_type"],
        )

    def before_generation(self, **kwargs: Any) -> dict[str, Any]:
        generation_id = str(kwargs.get("generation_id") or uuid.uuid4().hex)
        context = {"generation_id": generation_id, "started_monotonic": time.perf_counter()}
        self.event(
            "generation_started",
            invocation_id=self.current_invocation.get(),
            generation_id=generation_id,
            parent_generation_id=kwargs.get("parent_generation_id"),
            strategy=kwargs.get("strategy"),
        )
        return context

    def after_generation(self, context: Any, exception: BaseException | None, **kwargs: Any) -> None:
        generation_id = context.get("generation_id") if isinstance(context, dict) else kwargs.get("generation_id")
        self.event(
            "generation_finished",
            invocation_id=self.current_invocation.get(),
            generation_id=generation_id,
            status="failed" if exception is not None else "completed",
            error_type=type(exception).__name__ if exception is not None else None,
        )

    def _before_execution(self, kind: str, execution_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        context = {
            "kind": kind,
            "execution_id": execution_id,
            "name": name,
            "arguments": _artifact_value(arguments),
            "invocation_id": self.current_invocation.get(),
            "started_at": time.time(),
            "started_monotonic": time.perf_counter(),
        }
        self.event(
            f"{kind}_started",
            invocation_id=context["invocation_id"],
            execution_id=execution_id,
            name=name,
        )
        return context

    def _after_execution(self, context: Any, result: Any, exception: BaseException | None) -> None:
        if not isinstance(context, dict):
            return
        completed_at = time.time()
        status = "failed" if exception is not None else "completed"
        record = {
            "kind": "code" if context["kind"] == "code_execution" else "nooa",
            "invocation_id": context["invocation_id"],
            "tool_call_id": context["execution_id"],
            "name": context["name"],
            "arguments": context["arguments"],
            "output": _artifact_value(result) if exception is None else None,
            "status": status,
            "started_at": context["started_at"],
            "completed_at": completed_at,
            "duration_ms": max(0.0, (time.perf_counter() - context["started_monotonic"]) * 1000),
            "error_type": type(exception).__name__ if exception is not None else None,
        }
        self.tool_calls.append(record)
        self.event(
            f"{context['kind']}_finished",
            invocation_id=context["invocation_id"],
            execution_id=context["execution_id"],
            status=status,
            error_type=record["error_type"],
        )

    def before_code_execution(
        self,
        agent: Any,
        code: str,
        execution_id: str,
        generation_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del agent
        return self._before_execution(
            "code_execution",
            str(kwargs.get("tool_call_id") or execution_id),
            "execute_python",
            {"code": code, "generation_id": generation_id},
        )

    def after_code_execution(
        self,
        agent: Any,
        code: str,
        result: Any,
        exception: BaseException | None,
        context: Any,
        execution_id: str,
        **kwargs: Any,
    ) -> None:
        del agent, code, execution_id, kwargs
        self._after_execution(context, result, exception)

    def before_tool_execution(
        self,
        agent: Any,
        tool_name: str,
        arguments: dict[str, Any],
        execution_id: str,
        generation_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del agent, kwargs
        return self._before_execution(
            "tool_execution",
            execution_id,
            tool_name,
            {**arguments, "_generation_id": generation_id},
        )

    def after_tool_execution(
        self,
        agent: Any,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        exception: BaseException | None,
        context: Any,
        execution_id: str,
        **kwargs: Any,
    ) -> None:
        del agent, tool_name, arguments, execution_id, kwargs
        self._after_execution(context, result, exception)

    def before_method_invocation(self, **kwargs: Any) -> dict[str, Any]:
        context = {"started_monotonic": time.perf_counter()}
        self.event(
            "method_invocation_started",
            invocation_id=self.current_invocation.get(),
            method_name=kwargs.get("method_name"),
            method_invocation_id=kwargs.get("invocation_id"),
        )
        return context

    def after_method_invocation(self, context: Any, exception: BaseException | None, **kwargs: Any) -> None:
        del context
        self.event(
            "method_invocation_finished",
            invocation_id=self.current_invocation.get(),
            method_invocation_id=kwargs.get("invocation_id"),
            status="failed" if exception is not None else "completed",
            error_type=type(exception).__name__ if exception is not None else None,
        )

    def on_messages_built(self, **kwargs: Any) -> None:
        self.event(
            "messages_built",
            invocation_id=self.current_invocation.get(),
            generation_id=kwargs.get("generation_id"),
        )

    def record_model_call(self, record: dict[str, Any]) -> None:
        self.model_calls.append(record)
        response = record.get("response") or {}
        response_id = response.get("id")
        invocation_id = record["invocation_id"]
        if response_id and invocation_id in self.invocations:
            self.invocations[invocation_id]["model_response_ids"].append(response_id)
        self.event(
            "model_call_finished",
            invocation_id=invocation_id,
            response_id=response_id,
            error_type=record.get("error_type"),
        )


def _parse_http_payload(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {}
    if stripped.startswith("data:") or "\ndata:" in stripped:
        data_lines = [line[5:].strip() for line in stripped.splitlines() if line.startswith("data:")]
        stripped = data_lines[-1] if data_lines else "{}"
    payload = json.loads(stripped)
    if isinstance(payload, list):
        payload = next((item for item in payload if isinstance(item, dict) and "result" in item), {})
    if not isinstance(payload, dict):
        raise ValueError("HTTP endpoint returned a non-object JSON payload")
    return payload


class MCPClient:
    def __init__(self, session: Any, url: str, headers: dict[str, str]) -> None:
        self._session = session
        self._url = url
        self._headers = {
            "accept": "application/json, text/event-stream",
            "content-type": "application/json",
            **headers,
        }
        self._session_id: str | None = None
        self._next_id = 0
        self._lock = asyncio.Lock()

    async def _request(self, method: str, params: dict[str, Any] | None = None, *, notification: bool = False) -> Any:
        self._next_id += 1
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if not notification:
            payload["id"] = self._next_id
        if params is not None:
            payload["params"] = params
        headers = dict(self._headers)
        if self._session_id:
            headers["mcp-session-id"] = self._session_id
        async with self._session.post(self._url, json=payload, headers=headers) as response:
            text = await response.text(errors="replace")
            if response.status >= 400:
                raise RuntimeError(f"MCP HTTP {response.status}: {text[:1000]}")
            if response.headers.get("mcp-session-id"):
                self._session_id = response.headers["mcp-session-id"]
        if notification or not text.strip():
            return None
        decoded = _parse_http_payload(text)
        if decoded.get("error"):
            raise RuntimeError(f"MCP {method} failed: {decoded['error']}")
        return decoded.get("result")

    async def initialize(self) -> list[dict[str, Any]]:
        await self._request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "nemo-gym-nooa", "version": SCHEMA_VERSION},
            },
        )
        await self._request("notifications/initialized", notification=True)
        result = await self._request("tools/list", {})
        tools = (result or {}).get("tools", [])
        return [tool for tool in tools if isinstance(tool, dict)]

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        async with self._lock:
            result = await self._request("tools/call", {"name": name, "arguments": arguments})
        if not isinstance(result, dict):
            return result
        if result.get("isError"):
            raise RuntimeError(f"MCP tool {name!r} failed: {result.get('content')}")
        if result.get("structuredContent") is not None:
            return result["structuredContent"]
        content = result.get("content") or []
        texts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
        if len(texts) == 1:
            try:
                return json.loads(texts[0])
            except json.JSONDecodeError:
                return texts[0]
        return "".join(texts)


class RolloutBudget:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.calls = 0

    def charge(self) -> None:
        if self.calls >= self.limit:
            raise ModelCallBudgetExceeded(f"NOOA rollout exceeded {self.limit} model calls")
        self.calls += 1


def _dump(value: Any) -> Any:
    from pydantic import BaseModel

    return value.model_dump(mode="json", exclude_none=True) if isinstance(value, BaseModel) else value


def _responses_input(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
    instructions: list[str] = []
    output: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "system":
            if message.get("content"):
                instructions.append(str(message["content"]))
        elif isinstance(message.get("_batch"), list):
            output.extend(_dump(item) for item in message["_batch"])
        elif "type" in message:
            output.append(_dump(message))
        elif message.get("role") == "tool":
            output.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message.get("content", ""),
                }
            )
        elif message.get("role") == "assistant" and message.get("tool_calls"):
            if message.get("content"):
                output.append({"role": "assistant", "content": message["content"]})
            for call in message["tool_calls"]:
                function = call.get("function", {})
                output.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": function.get("name", ""),
                        "arguments": function.get("arguments", ""),
                    }
                )
        else:
            output.append(_dump(message))
    return output, "\n\n".join(instructions) or None


def _output_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "\n".join(parts)


def _tool_schema(tool: Any) -> dict[str, Any]:
    schema = tool.get_parameter_schema()
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": schema,
        "strict": set(schema.get("required", [])) == set(schema.get("properties", {})),
    }


def _make_llm_class() -> type:
    from nooa.unifiedllm import LLMResponse, ToolCall, UnifiedLLM

    class GymResponsesLLM(UnifiedLLM):
        def __init__(
            self,
            *,
            session: Any,
            endpoint: dict[str, Any],
            budget: RolloutBudget,
            recorder: ArtifactRecorder,
        ) -> None:
            super().__init__(model=endpoint["model"])
            self._session = session
            self._endpoint = endpoint
            self._budget = budget
            self._recorder = recorder

        def call(self, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise RuntimeError("sandboxed NOOA supports async entrypoints only")

        async def acall(
            self,
            messages: list[dict[str, Any]],
            tools: list[Any] | None = None,
            output_model: type | None = None,
            **kwargs: Any,
        ) -> Any:
            self._budget.charge()
            input_items, instructions = _responses_input(messages)
            request = {
                **self._endpoint.get("response_defaults", {}),
                "input": input_items,
                "instructions": instructions,
                "model": self._endpoint["model"],
                "parallel_tool_calls": False,
                "tools": [_tool_schema(tool) for tool in tools or []],
            }
            if output_model is not None:
                schema_factory = getattr(output_model, "model_json_schema")
                request["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": output_model.__name__,
                        "schema": schema_factory(),
                        "strict": True,
                    }
                }
            aliases = {"max_tokens": "max_output_tokens"}
            for key, value in kwargs.items():
                destination = aliases.get(key, key)
                if (
                    destination
                    in {
                        "temperature",
                        "top_p",
                        "max_output_tokens",
                        "reasoning",
                        "text",
                        "tool_choice",
                    }
                    and value is not None
                ):
                    request[destination] = _dump(value)

            started_at = time.time()
            started_monotonic = time.perf_counter()
            record = {
                "invocation_id": self._recorder.current_invocation.get(),
                "model_server_name": self._endpoint["server_name"],
                "request": request,
                "response": None,
                "started_at": started_at,
            }
            try:
                async with self._session.post(
                    self._endpoint["url"],
                    json=request,
                    headers=self._endpoint.get("headers", {}),
                ) as http_response:
                    text = await http_response.text(errors="replace")
                    if http_response.status >= 400:
                        raise RuntimeError(f"model HTTP {http_response.status}: {text[:2000]}")
                response = json.loads(text)
                if not isinstance(response, dict):
                    raise ValueError("model endpoint returned non-object JSON")
                record["response"] = response
            except BaseException as error:
                record["error_type"] = type(error).__name__
                raise
            finally:
                record["completed_at"] = time.time()
                record["duration_ms"] = max(0.0, (time.perf_counter() - started_monotonic) * 1000)
                self._recorder.record_model_call(record)

            dumped_output = [dict(item) for item in response.get("output", []) if isinstance(item, dict)]
            function_calls = [item for item in dumped_output if item.get("type") == "function_call"]
            usage = response.get("usage")
            if function_calls:
                return LLMResponse(
                    raw_response=response,
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=str(item.get("call_id") or item.get("id")),
                            name=str(item.get("name") or ""),
                            arguments=str(item.get("arguments") or ""),
                        )
                        for item in function_calls
                    ],
                    finish_reason="tool_calls",
                    assistant_message={"_batch": dumped_output},
                    usage=usage,
                )

            content: Any = _output_text(response)
            if output_model is not None:
                try:
                    content = getattr(output_model, "model_validate")(json.loads(content))
                except (json.JSONDecodeError, ValueError, TypeError) as error:
                    raise InvalidPolicyOutput(f"model returned invalid {output_model.__name__} JSON") from error
            reasoning = [item for item in dumped_output if item.get("type") == "reasoning"]
            return LLMResponse(
                raw_response=response,
                content=content,
                tool_calls=[],
                finish_reason="length" if response.get("incomplete_details") else "stop",
                assistant_message={"role": "assistant", "content": _output_text(response)},
                reasoning=json.dumps(reasoning) if reasoning else None,
                usage=usage,
            )

    return GymResponsesLLM


def _annotation(schema: dict[str, Any]) -> Any:
    kind = schema.get("type")
    if not isinstance(kind, str):
        return Any
    return {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }.get(kind, Any)


def _build_tool_namespace(
    client: MCPClient,
    tools: list[dict[str, Any]],
    allowed_tools: frozenset[str],
    recorder: ArtifactRecorder,
) -> Any:
    from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

    methods: dict[str, Any] = {}
    for tool in tools:
        name = tool.get("name")
        if not isinstance(name, str) or name not in allowed_tools:
            continue
        schema = tool.get("inputSchema") or {"type": "object", "properties": {}}
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)

        def make_method(
            tool_name: str,
            tool_schema: dict[str, Any],
            tool_validator: Any,
            description: str | None,
        ) -> Any:
            async def invoke(namespace: Any, *args: Any, **kwargs: Any) -> Any:
                bound = inspect.signature(invoke).bind(namespace, *args, **kwargs)
                arguments = dict(bound.arguments)
                arguments.pop("self", None)
                tool_validator.validate(arguments)
                started_at = time.time()
                started_monotonic = time.perf_counter()
                call_id = f"mcp_{uuid.uuid4().hex}"
                output: Any = None
                status = "completed"
                error_type = None
                try:
                    output = await client.call(tool_name, arguments)
                    return output
                except asyncio.CancelledError:
                    status = "cancelled"
                    error_type = "CancelledError"
                    raise
                except Exception as error:
                    status = "failed"
                    error_type = type(error).__name__
                    raise
                finally:
                    completed_at = time.time()
                    record = {
                        "kind": "resource",
                        "invocation_id": recorder.current_invocation.get(),
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "arguments": arguments,
                        "output": _artifact_value(output),
                        "status": status,
                        "started_at": started_at,
                        "completed_at": completed_at,
                        "duration_ms": max(0.0, (time.perf_counter() - started_monotonic) * 1000),
                        "error_type": error_type,
                    }
                    recorder.tool_calls.append(record)
                    recorder.event("mcp_tool_finished", **record)

            invoke.__name__ = tool_name
            invoke.__qualname__ = tool_name
            invoke.__doc__ = description or f"Call the {tool_name} resource tool."
            properties = tool_schema.get("properties", {})
            required = set(tool_schema.get("required", []))
            ordered = [key for key in properties if key in required] + [
                key for key in properties if key not in required
            ]
            parameters = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
            for parameter_name in ordered:
                parameter_schema = properties[parameter_name]
                default = inspect.Parameter.empty if parameter_name in required else parameter_schema.get("default")
                parameters.append(
                    inspect.Parameter(
                        parameter_name,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=default,
                        annotation=_annotation(parameter_schema),
                    )
                )
            setattr(invoke, "__signature__", inspect.Signature(parameters, return_annotation=Any))
            return invoke

        methods[name] = make_method(name, schema, validator, tool.get("description"))

    namespace_type = type("GymResourceTools", (), methods)
    return namespace_type()


async def run(request_path: Path) -> int:
    # NOOA currently imports LiteLLM even though this runner replaces its HTTP
    # client. Keep that import offline and route actual inference through aiohttp.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import aiohttp

    request = json.loads(request_path.read_text(encoding="utf-8"))
    request_path.unlink(missing_ok=True)
    artifacts_dir = Path(request["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    recorder = ArtifactRecorder(artifacts_dir)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "return_value": None,
        "return_type": None,
        "error": None,
        "model_calls": [],
        "tool_calls": [],
        "invocations": [],
        "observation_gaps": [],
        "runtime": {
            "python": sys.version,
            "cwd": os.getcwd(),
            "agent_class": request.get("agent_class"),
        },
    }

    timeout = aiohttp.ClientTimeout(total=None, connect=60, sock_read=900)
    try:
        import nooa
        from nooa import Agent
        from nooa.runtime import hooks as nooa_hooks  # type: ignore[import-untyped]

        nooa_version = importlib.metadata.version("nooa")
        result["runtime"].update(
            {
                "nooa_version": nooa_version,
                "nooa_path": str(Path(nooa.__file__).resolve()) if nooa.__file__ else None,
            }
        )
        expected_version = request.get("expected_nooa_version")
        if expected_version is not None and nooa_version != expected_version:
            raise RuntimeError(f"NOOA runtime version {nooa_version!r} does not match expected {expected_version!r}")

        try:
            hooks_scope = nooa_hooks.hooks_scope
        except AttributeError:

            @contextmanager
            def hooks_scope(hooks: Any) -> Any:
                previous = nooa_hooks.get_hooks()
                nooa_hooks.set_hooks(hooks)
                try:
                    yield hooks
                finally:
                    nooa_hooks.set_hooks(previous)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            llm_class = _make_llm_class()
            budget = RolloutBudget(int(request["max_model_calls"]))
            primary_endpoint = dict(request["model"])
            primary_endpoint["response_defaults"] = request.get("response_defaults", {})
            llm = llm_class(
                session=session,
                endpoint=primary_endpoint,
                budget=budget,
                recorder=recorder,
            )
            alias_clients = {
                alias: llm_class(
                    session=session,
                    endpoint={**endpoint, "response_defaults": request.get("response_defaults", {})},
                    budget=budget,
                    recorder=recorder,
                )
                for alias, endpoint in request.get("model_aliases", {}).items()
            }

            module_name, _, class_name = request["agent_class"].partition(":")
            module = importlib.import_module(module_name)
            result["runtime"]["agent_module_path"] = str(Path(module.__file__).resolve()) if module.__file__ else None
            configured_class = getattr(module, class_name)
            if not inspect.isclass(configured_class) or not issubclass(configured_class, Agent):
                raise TypeError(f"{request['agent_class']!r} is not a nooa.Agent subclass")

            class_attributes: dict[str, Any] = {}
            mcp_config = request.get("mcp")
            if mcp_config and mcp_config.get("allowed_tools"):
                mcp_client = MCPClient(session, mcp_config["url"], mcp_config.get("headers", {}))
                advertised_tools = await mcp_client.initialize()
                allowed_tools = frozenset(mcp_config["allowed_tools"])
                advertised_names = {tool.get("name") for tool in advertised_tools if isinstance(tool.get("name"), str)}
                missing_tools = sorted(allowed_tools - advertised_names)
                if missing_tools:
                    raise ValueError(f"configured resource tools were not advertised over MCP: {missing_tools}")
                namespace = _build_tool_namespace(
                    mcp_client,
                    advertised_tools,
                    allowed_tools,
                    recorder,
                )
                class_attributes["__annotations__"] = {request["tool_namespace"]: type(namespace)}
            else:
                namespace = None
            rollout_class = type(configured_class.__name__, (configured_class,), class_attributes)
            agent = rollout_class(llm=llm, **request.get("init_kwargs", {}))
            if namespace is not None:
                if inspect.getattr_static(agent, request["tool_namespace"], None) is not None:
                    raise ValueError(f"tool namespace {request['tool_namespace']!r} conflicts with an agent attribute")
                setattr(agent, request["tool_namespace"], namespace)
            if alias_clients:
                from nooa.agentdoc import spec

                for alias, client in alias_clients.items():
                    if inspect.getattr_static(agent, alias, None) is not None:
                        raise ValueError(f"model alias {alias!r} conflicts with an agent attribute")
                    setattr(agent, alias, client)
                    spec(agent, alias, hidden=True)
            entrypoint = getattr(agent, request["entrypoint"])
            if not inspect.iscoroutinefunction(entrypoint):
                raise TypeError(f"NOOA entrypoint {request['entrypoint']!r} must be async")

            try:
                from nooa.tracing import session_scope  # type: ignore[import-untyped]
            except ImportError:
                from contextlib import nullcontext

                trace_context = nullcontext()
            else:
                trace_context = session_scope(f"{request['rollout_id']}-{uuid.uuid4().hex[:8]}")

            with trace_context, hooks_scope(recorder):
                return_value = await entrypoint(**request.get("arguments", {}))
            result.update(
                status="completed",
                return_value=_artifact_value(return_value),
                return_type=_qualified_type(return_value),
            )
    except ModelCallBudgetExceeded as error:
        result.update(status="model_budget_exceeded", error=str(error))
    except InvalidPolicyOutput as error:
        result.update(status="invalid_policy_output", error=str(error))
    except asyncio.CancelledError:
        result.update(status="cancelled", error="sandbox runner was cancelled")
        raise
    except BaseException as error:
        causes: list[BaseException] = []
        current: BaseException | None = error
        while current is not None and all(current is not seen for seen in causes):
            causes.append(current)
            current = current.__cause__ or current.__context__
        if any(isinstance(cause, ModelCallBudgetExceeded) for cause in causes):
            status = "model_budget_exceeded"
        elif any(isinstance(cause, InvalidPolicyOutput) for cause in causes):
            status = "invalid_policy_output"
        else:
            status = "failed"
        result.update(status=status, error=f"{type(error).__name__}: {error}")
        (artifacts_dir / "traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        result["model_calls"] = recorder.model_calls
        result["tool_calls"] = recorder.tool_calls
        result["invocations"] = list(recorder.invocations.values())
        temporary = artifacts_dir / "result.json.tmp"
        temporary.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
        temporary.replace(artifacts_dir / "result.json")
    return 0 if result["status"] == "completed" else 2


def main() -> None:
    request_path = Path(os.environ.get("NOOA_SANDBOX_REQUEST", "/tmp/nemo-gym-nooa/request.json"))
    sys.exit(asyncio.run(run(request_path)))


if __name__ == "__main__":
    main()
