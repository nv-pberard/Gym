# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import time
from pathlib import Path

import pytest
from aiohttp import web

from responses_api_agents.nooa_sandboxed_agent.protocol import NOOAModelEndpoint, NOOASandboxRequest
from responses_api_agents.nooa_sandboxed_agent.sandbox_runner import run


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("nooa") is None, reason="NOOA is installed in the sandbox image"
)


async def test_runner_executes_generated_nooa_method_through_gym_model_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_requests: list[dict] = []

    async def model(request: web.Request) -> web.Response:
        body = await request.json()
        model_requests.append(body)
        return web.json_response(
            {
                "id": "resp_runner_smoke",
                "object": "response",
                "created_at": int(time.time()),
                "model": body["model"],
                "output": [
                    {
                        "id": "msg_runner_smoke",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps({"value": "sandbox-model-ok"}),
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
            }
        )

    app = web.Application()
    app.router.add_post("/v1/responses", model)
    web_runner = web.AppRunner(app)
    await web_runner.setup()
    site = web.TCPSite(web_runner, "127.0.0.1", 0)
    await site.start()
    assert site._server is not None
    port = site._server.sockets[0].getsockname()[1]
    artifacts = tmp_path / "artifacts"
    request_path = tmp_path / "request.json"
    sandbox_request = NOOASandboxRequest(
        rollout_id="rollout-smoke",
        task_id="task-smoke",
        agent_class="smoke_agent:SmokeAgent",
        entrypoint="solve",
        arguments={"task": "return sandbox-model-ok"},
        model=NOOAModelEndpoint(
            url=f"http://127.0.0.1:{port}/v1/responses",
            model="policy",
            server_name="policy_model",
        ),
        max_model_calls=3,
        artifacts_dir=str(artifacts),
    )
    request_path.write_text(sandbox_request.model_dump_json(), encoding="utf-8")
    monkeypatch.syspath_prepend(str(Path(__file__).with_name("fixtures")))

    try:
        return_code = await run(request_path)
    finally:
        await web_runner.cleanup()

    result = json.loads((artifacts / "result.json").read_text(encoding="utf-8"))
    assert return_code == 0
    assert result["status"] == "completed"
    assert result["return_value"] == "sandbox-model-ok"
    assert len(result["model_calls"]) == 1
    assert result["invocations"]
    assert model_requests[0]["text"]["format"]["schema"]["required"] == ["value"]
    assert not request_path.exists()


async def test_runner_preserves_model_tool_call_id_for_generated_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def model(request: web.Request) -> web.Response:
        body = await request.json()
        assert [tool["name"] for tool in body["tools"]] == ["execute_python", "return_result"]
        return web.json_response(
            {
                "id": "resp_codeact_smoke",
                "object": "response",
                "created_at": int(time.time()),
                "model": body["model"],
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_codeact_smoke",
                        "call_id": "call_codeact_smoke",
                        "name": "execute_python",
                        "arguments": json.dumps({"code": 'return "sandbox-code-ok"'}),
                        "status": "completed",
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": body["tools"],
                "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
            }
        )

    app = web.Application()
    app.router.add_post("/v1/responses", model)
    web_runner = web.AppRunner(app)
    await web_runner.setup()
    site = web.TCPSite(web_runner, "127.0.0.1", 0)
    await site.start()
    assert site._server is not None
    port = site._server.sockets[0].getsockname()[1]
    artifacts = tmp_path / "artifacts"
    request_path = tmp_path / "request.json"
    sandbox_request = NOOASandboxRequest(
        rollout_id="rollout-codeact-smoke",
        task_id="task-codeact-smoke",
        agent_class="smoke_agent:CodeActSmokeAgent",
        entrypoint="solve",
        arguments={"task": "return sandbox-code-ok"},
        model=NOOAModelEndpoint(
            url=f"http://127.0.0.1:{port}/v1/responses",
            model="policy",
            server_name="policy_model",
        ),
        max_model_calls=3,
        artifacts_dir=str(artifacts),
    )
    request_path.write_text(sandbox_request.model_dump_json(), encoding="utf-8")
    monkeypatch.syspath_prepend(str(Path(__file__).with_name("fixtures")))

    try:
        return_code = await run(request_path)
    finally:
        await web_runner.cleanup()

    result = json.loads((artifacts / "result.json").read_text(encoding="utf-8"))
    assert return_code == 0
    assert result["status"] == "completed"
    assert result["return_value"] == "sandbox-code-ok"
    code_tool_call_ids = [tool["tool_call_id"] for tool in result["tool_calls"] if tool["kind"] == "code"]
    assert "call_codeact_smoke" in code_tool_call_ids
    prefill = next(tool for tool in result["tool_calls"] if tool["tool_call_id"].startswith("prefill_"))
    assert "Task: solve()" in prefill["output"]["stdout"]
