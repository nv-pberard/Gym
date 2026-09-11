# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral sandbox acquisition, staging, and artifact collection."""

from __future__ import annotations

import json
import logging
import shlex
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec, create_provider
from responses_api_agents.nooa_sandboxed_agent.config import NOOASandboxedAgentConfig
from responses_api_agents.nooa_sandboxed_agent.protocol import NOOASandboxRequest, NOOASandboxResult


REMOTE_ROOT_PREFIX = "/tmp/nemo-gym-nooa-"
LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class AcquiredSandbox:
    sandbox: AsyncSandbox
    owned: bool
    cleanup_url_path: str | None = None


@dataclass(slots=True)
class SandboxExecution:
    result: NOOASandboxResult
    return_code: int
    error_type: str | None
    stdout: str
    stderr: str
    wall_time_s: float
    artifacts_path: str | None = None


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
    cleanup_url_path = seed_response.get("cleanup_url_path")
    if cleanup_url_path is not None and (
        not isinstance(cleanup_url_path, str) or not cleanup_url_path.startswith("/")
    ):
        raise ValueError("cleanup_url_path must be an absolute URL path")

    lease = seed_response.get("sandbox_lease")
    if isinstance(lease, Mapping):
        descriptor = lease.get("descriptor")
        if not isinstance(descriptor, Mapping):
            raise ValueError("sandbox_lease.descriptor must be an object")
        ownership = lease.get("ownership", "resources")
        if ownership not in {"agent", "resources"}:
            raise ValueError("sandbox_lease.ownership must be 'agent' or 'resources'")
        provider = create_provider(resolved_provider)
        return AcquiredSandbox(
            sandbox=await AsyncSandbox.connect(dict(descriptor), provider=provider),
            owned=ownership == "agent",
            cleanup_url_path=cleanup_url_path,
        )

    descriptor = seed_response.get("sandbox_descriptor")
    if isinstance(descriptor, Mapping):
        provider = create_provider(resolved_provider)
        return AcquiredSandbox(
            sandbox=await AsyncSandbox.connect(dict(descriptor), provider=provider),
            owned=False,
            cleanup_url_path=cleanup_url_path,
        )

    legacy_handle = seed_response.get("sandbox_handle")
    if isinstance(legacy_handle, str) and legacy_handle:
        provider = create_provider(resolved_provider)
        return AcquiredSandbox(
            sandbox=await AsyncSandbox.connect({"sandbox_id": legacy_handle}, provider=provider),
            owned=False,
            cleanup_url_path=cleanup_url_path,
        )

    sandbox = AsyncSandbox(resolved_provider, _sandbox_spec(config, default_metadata))
    await sandbox.start()
    return AcquiredSandbox(sandbox=sandbox, owned=True, cleanup_url_path=cleanup_url_path)


async def release_sandbox(acquired: AcquiredSandbox) -> None:
    if acquired.owned:
        await acquired.sandbox.stop()
    else:
        await acquired.sandbox.detach()


async def _stage_archive(
    sandbox: AsyncSandbox,
    local_path: str,
    remote_name: str,
    target: str,
    *,
    remote_root: str,
) -> None:
    archive = Path(local_path).expanduser().resolve()
    if not archive.is_file():
        raise ValueError(f"runtime archive not found: {archive}")
    remote_archive = f"{remote_root}/{remote_name}"
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
    run_id = uuid4().hex
    remote_root = f"{REMOTE_ROOT_PREFIX}{run_id}"
    remote_request = f"{remote_root}/request.json"
    remote_runner = f"{remote_root}/sandbox_runner.py"
    remote_artifacts = f"{remote_root}/artifacts"
    remote_agent_source = f"{remote_root}/agent-source"
    runtime_extract_dir = f"{config.runtime.extract_dir}-{run_id}"
    request = request.model_copy(update={"artifacts_dir": remote_artifacts})
    runner_path = Path(__file__).with_name("sandbox_runner.py")
    local_artifacts = Path(config.results_dir).expanduser().resolve() / run_id if config.results_dir else None
    if local_artifacts is not None:
        local_artifacts.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="nemo-gym-nooa-") as temporary_dir:
        local_request = Path(temporary_dir) / "request.json"
        local_request.write_text(request.model_dump_json(), encoding="utf-8")
        prepare = await sandbox.exec(
            f"mkdir -p {shlex.quote(remote_artifacts)}",
            timeout_s=30,
        )
        if prepare.return_code != 0 or prepare.error_type:
            raise RuntimeError(f"failed to prepare sandbox runner directory: {(prepare.stderr or '')[:1000]}")
        await sandbox.upload(runner_path, remote_runner)

        python = config.runtime.python
        if config.runtime.source == "archive":
            assert config.runtime.archive_path is not None
            await _stage_archive(
                sandbox,
                config.runtime.archive_path,
                "runtime.tar.gz",
                runtime_extract_dir,
                remote_root=remote_root,
            )
            python = f"{runtime_extract_dir}/bin/python"
        if config.agent.source_archive:
            await _stage_archive(
                sandbox,
                config.agent.source_archive,
                "agent-source.tar.gz",
                remote_agent_source,
                remote_root=remote_root,
            )

        check = await sandbox.exec(f"test -x {shlex.quote(python)}", timeout_s=30)
        if check.return_code != 0:
            raise RuntimeError(f"sandbox NOOA Python is not executable: {python}")

        pythonpath = (
            [remote_agent_source, *config.agent.pythonpath] if config.agent.source_archive else config.agent.pythonpath
        )
        env = {
            "NOOA_SANDBOX_REQUEST": remote_request,
            "PYTHONPATH": ":".join(pythonpath),
            "HOME": f"{remote_root}/home",
            "TMPDIR": f"{remote_root}/tmp",
        }
        prepare_home = await sandbox.exec(
            f"mkdir -p {shlex.quote(env['HOME'])} {shlex.quote(env['TMPDIR'])}",
            timeout_s=30,
        )
        if prepare_home.return_code != 0 or prepare_home.error_type:
            raise RuntimeError(f"failed to prepare sandbox runtime directories: {(prepare_home.stderr or '')[:1000]}")
        # Upload the request only once every other preparation step has succeeded.
        # The runner unlinks it immediately after parsing.
        await sandbox.upload(local_request, remote_request)
        started = time.perf_counter()
        execution = await sandbox.exec(
            f"{shlex.quote(python)} {shlex.quote(remote_runner)}",
            cwd=config.runtime.workdir,
            env=env,
            timeout_s=config.timeout_s,
        )
        wall_time_s = max(0.0, time.perf_counter() - started)

        try:
            local_result = (local_artifacts or Path(temporary_dir)) / "result.json"
            exists = await sandbox.exec(f"test -f {shlex.quote(remote_artifacts + '/result.json')}", timeout_s=30)
            if exists.return_code == 0:
                await sandbox.download(f"{remote_artifacts}/result.json", local_result)
                if local_result.stat().st_size > config.max_artifact_bytes:
                    local_result.unlink()
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
                if local_artifacts is not None:
                    local_result.write_text(result.model_dump_json(), encoding="utf-8")

            if local_artifacts is not None:
                for artifact_name in ("events.jsonl", "traceback.log"):
                    remote_path = f"{remote_artifacts}/{artifact_name}"
                    exists = await sandbox.exec(f"test -f {shlex.quote(remote_path)}", timeout_s=30)
                    if exists.return_code != 0:
                        continue
                    local_path = local_artifacts / artifact_name
                    await sandbox.download(remote_path, local_path)
                    if local_path.stat().st_size > config.max_artifact_bytes:
                        local_path.unlink()
                        LOG.warning("Skipped oversized NOOA artifact %s", remote_path)

                (local_artifacts / "execution.json").write_text(
                    json.dumps(
                        {
                            "return_code": execution.return_code,
                            "error_type": execution.error_type,
                            "stdout": execution.stdout or "",
                            "stderr": execution.stderr or "",
                            "wall_time_s": wall_time_s,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        finally:
            try:
                cleanup_paths = [remote_root]
                if config.runtime.source == "archive":
                    cleanup_paths.append(runtime_extract_dir)
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
        artifacts_path=str(local_artifacts) if local_artifacts is not None else None,
    )
