# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tarfile
from pathlib import Path

from responses_api_agents.nooa_sandboxed_agent.config import NOOARuntimeConfig
from responses_api_agents.nooa_sandboxed_agent.runtime_builder import prepare_runtime_archive


def test_prepare_runtime_archive_is_content_addressed_and_cached(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_run(command, *, check, env):
        calls.append((command, check, env))
        python = Path(env["NOOA_RUNTIME_DIR"]) / "bin" / "python3"
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text("portable", encoding="utf-8")

    monkeypatch.setattr("responses_api_agents.nooa_sandboxed_agent.runtime_builder.subprocess.run", fake_run)
    config = NOOARuntimeConfig(source="auto", cache_dir=str(tmp_path))

    first = prepare_runtime_archive(config)
    second = prepare_runtime_archive(config)

    assert first == second
    assert len(calls) == 1
    assert calls[0][0][0] == "bash"
    assert first.is_file()
    with tarfile.open(first) as archive:
        assert "./bin/python3" in archive.getnames()


def test_runtime_recipe_changes_with_target_architecture(monkeypatch, tmp_path: Path) -> None:
    def fake_run(command, *, check, env):
        del command, check
        python = Path(env["NOOA_RUNTIME_DIR"]) / "bin" / "python3"
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text("portable", encoding="utf-8")

    monkeypatch.setattr("responses_api_agents.nooa_sandboxed_agent.runtime_builder.subprocess.run", fake_run)

    x86 = prepare_runtime_archive(
        NOOARuntimeConfig(source="auto", cache_dir=str(tmp_path), architecture="x86_64-unknown-linux-gnu")
    )
    arm = prepare_runtime_archive(
        NOOARuntimeConfig(source="auto", cache_dir=str(tmp_path), architecture="aarch64-unknown-linux-gnu")
    )

    assert x86 != arm
