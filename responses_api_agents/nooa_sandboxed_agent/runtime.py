# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral sandbox acquisition, staging, and artifact collection."""

from __future__ import annotations

import logging
import shlex
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec, create_provider
from responses_api_agents.nooa_sandboxed_agent.config import NOOASandboxedAgentConfig
from responses_api_agents.nooa_sandboxed_agent.protocol import NOOASandboxRequest, NOOASandboxResult


REMOTE_ROOT = "/tmp/nemo-gym-nooa"
REMOTE_REQUEST = f"{REMOTE_ROOT}/request.json"
REMOTE_RUNNER = f"{REMOTE_ROOT}/sandbox_runner.py"
REMOTE_ARTIFACTS = f"{REMOTE_ROOT}/artifacts"
REMOTE_AGENT_SOURCE = f"{REMOTE_ROOT}/agent-source"
LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class AcquiredSandbox:
    sandbox: AsyncSandbox
    owned: bool


@dataclass(slots=True)
class SandboxExecution:
    result: NOOASandboxResult
    return_code: int
    error_type: str | None
    stdout: str
    stderr: str
    wall_time_s: float


def _sandbox_spec(config: NOOASandboxedAgentConfig, default_metadata: dict[str, Any]) -> SandboxSpec:
    values = dict(config.sandbox_spec)
    provider_options = dict(values.pop("provider_options", {}))
    metadata = {
        **default_metadata,
        **values.pop("metadata", {}),
        "nemo_gym_agent": config.name or "nooa_sandboxed_agent",
    }
    spec = SandboxSpec(
        image=values.pop("image", None),
        ttl_s=values.pop("ttl_s", config.timeout_s + 600),
        ready_timeout_s=values.pop("ready_timeout_s", 1200),
        workdir=values.pop("workdir", None),
        env=dict(values.pop("env", {})),
        files=dict(values.pop("files", {})),
        metadata=metadata,
        resources=SandboxResources.from_mapping(values.pop("resources", {})),
        entrypoint=values.pop("entrypoint", None),
        provider_options=provider_options,
        ports=values.pop("ports", ()),
    )
    if values:
        raise ValueError(f"Unsupported sandbox_spec fields: {sorted(values)}")
    return spec


async def acquire_sandbox(
    config: NOOASandboxedAgentConfig,
    *,
    seed_response: dict[str, Any],
    resolved_provider: dict[str, Any],
    default_metadata: dict[str, Any],
) -> AcquiredSandbox:
    lease = seed_response.get("sandbox_lease")
    if isinstance(lease, dict):
        descriptor = lease.get("descriptor")
        if not isinstance(descriptor, dict):
            raise ValueError("sandbox_lease.descriptor must be an object")
        provider = create_provider(resolved_provider)
        return AcquiredSandbox(
            sandbox=await AsyncSandbox.connect(descriptor, provider=provider),
            owned=lease.get("ownership", "resources") == "agent",
        )

    legacy_handle = seed_response.get("sandbox_handle")
    if isinstance(legacy_handle, str) and legacy_handle:
        provider = create_provider(resolved_provider)
        return AcquiredSandbox(
            sandbox=await AsyncSandbox.connect({"sandbox_id": legacy_handle}, provider=provider),
            owned=False,
        )

    sandbox = AsyncSandbox(resolved_provider, _sandbox_spec(config, default_metadata))
    await sandbox.start()
    return AcquiredSandbox(sandbox=sandbox, owned=True)


async def release_sandbox(acquired: AcquiredSandbox) -> None:
    if acquired.owned:
        await acquired.sandbox.stop()
    else:
        await acquired.sandbox.detach()


async def _stage_archive(sandbox: AsyncSandbox, local_path: str, remote_name: str, target: str) -> None:
    archive = Path(local_path).expanduser().resolve()
    if not archive.is_file():
        raise ValueError(f"runtime archive not found: {archive}")
    remote_archive = f"{REMOTE_ROOT}/{remote_name}"
    await sandbox.upload(archive, remote_archive)
    command = (
        f"mkdir -p {shlex.quote(target)} && "
        f"tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(target)} && "
        f"rm -f {shlex.quote(remote_archive)}"
    )
    result = await sandbox.exec(command, timeout_s=600)
    if result.return_code != 0 or result.error_type:
        raise RuntimeError(f"failed to extract {archive.name}: {(result.stderr or '')[:1000]}")


async def execute_in_sandbox(
    acquired: AcquiredSandbox,
    request: NOOASandboxRequest,
    config: NOOASandboxedAgentConfig,
) -> SandboxExecution:
    sandbox = acquired.sandbox
    request = request.model_copy(update={"artifacts_dir": REMOTE_ARTIFACTS})
    runner_path = Path(__file__).with_name("sandbox_runner.py")
    with tempfile.TemporaryDirectory(prefix="nemo-gym-nooa-") as temporary_dir:
        local_request = Path(temporary_dir) / "request.json"
        local_request.write_text(request.model_dump_json(), encoding="utf-8")
        prepare = await sandbox.exec(
            f"rm -rf {shlex.quote(REMOTE_ROOT)} && mkdir -p {shlex.quote(REMOTE_ARTIFACTS)}",
            timeout_s=30,
        )
        if prepare.return_code != 0 or prepare.error_type:
            raise RuntimeError(f"failed to prepare sandbox runner directory: {(prepare.stderr or '')[:1000]}")
        await sandbox.upload(runner_path, REMOTE_RUNNER)

        python = config.runtime.python
        if config.runtime.source == "archive":
            assert config.runtime.archive_path is not None
            await _stage_archive(
                sandbox,
                config.runtime.archive_path,
                "runtime.tar.gz",
                config.runtime.extract_dir,
            )
            python = f"{config.runtime.extract_dir}/bin/python"
        if config.agent.source_archive:
            await _stage_archive(sandbox, config.agent.source_archive, "agent-source.tar.gz", REMOTE_AGENT_SOURCE)

        check = await sandbox.exec(f"test -x {shlex.quote(python)}", timeout_s=30)
        if check.return_code != 0:
            raise RuntimeError(f"sandbox NOOA Python is not executable: {python}")

        pythonpath = (
            [REMOTE_AGENT_SOURCE, *config.agent.pythonpath] if config.agent.source_archive else config.agent.pythonpath
        )
        env = {
            "NOOA_SANDBOX_REQUEST": REMOTE_REQUEST,
            "PYTHONPATH": ":".join(pythonpath),
            "HOME": f"{REMOTE_ROOT}/home",
            "TMPDIR": f"{REMOTE_ROOT}/tmp",
        }
        prepare_home = await sandbox.exec(
            f"mkdir -p {shlex.quote(env['HOME'])} {shlex.quote(env['TMPDIR'])}",
            timeout_s=30,
        )
        if prepare_home.return_code != 0 or prepare_home.error_type:
            raise RuntimeError(f"failed to prepare sandbox runtime directories: {(prepare_home.stderr or '')[:1000]}")
        # Upload the signed request only once every other preparation step has
        # succeeded. The runner unlinks it immediately after parsing.
        await sandbox.upload(local_request, REMOTE_REQUEST)
        started = time.perf_counter()
        execution = await sandbox.exec(
            f"{shlex.quote(python)} {shlex.quote(REMOTE_RUNNER)}",
            env=env,
            timeout_s=config.timeout_s,
        )
        wall_time_s = max(0.0, time.perf_counter() - started)

        try:
            local_result = Path(temporary_dir) / "result.json"
            exists = await sandbox.exec(f"test -f {shlex.quote(REMOTE_ARTIFACTS + '/result.json')}", timeout_s=30)
            if exists.return_code == 0:
                await sandbox.download(f"{REMOTE_ARTIFACTS}/result.json", local_result)
                if local_result.stat().st_size > config.max_artifact_bytes:
                    raise ValueError(f"NOOA result artifact exceeds max_artifact_bytes={config.max_artifact_bytes}")
                result = NOOASandboxResult.model_validate_json(local_result.read_text(encoding="utf-8"))
            else:
                status = "cancelled" if execution.error_type == "cancelled" else "failed"
                result = NOOASandboxResult(
                    status=status,
                    error=(execution.stderr or execution.error_type or "sandbox runner produced no result artifact")[
                        :4000
                    ],
                    observation_gaps=["result_artifact_unavailable"],
                )
        finally:
            try:
                cleanup_paths = [REMOTE_ROOT]
                if config.runtime.source == "archive":
                    cleanup_paths.append(config.runtime.extract_dir)
                quoted_paths = " ".join(shlex.quote(path) for path in cleanup_paths)
                await sandbox.exec(f"rm -rf {quoted_paths}", timeout_s=30)
            except Exception:
                LOG.warning("Failed to remove staged NOOA files from sandbox %s", sandbox.sandbox_id, exc_info=True)

    return SandboxExecution(
        result=result,
        return_code=execution.return_code,
        error_type=execution.error_type,
        stdout=execution.stdout or "",
        stderr=execution.stderr or "",
        wall_time_s=wall_time_s,
    )
