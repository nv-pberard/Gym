# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and cache the portable NOOA runtime staged into task sandboxes."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from uuid import uuid4

from responses_api_agents.nooa_sandboxed_agent.config import NOOARuntimeConfig


AGENT_DIR = Path(__file__).resolve().parent
LOCK_STALE_AFTER_S = 3600


def _runtime_recipe(config: NOOARuntimeConfig) -> str:
    digest = hashlib.sha256()
    for value in (
        config.python_version,
        config.python_build_standalone_release,
        config.architecture,
    ):
        digest.update(value.encode())
        digest.update(b"\0")
    for path in (AGENT_DIR / "setup_runtime.sh", AGENT_DIR / "runtime-requirements.txt"):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:20]


def _acquire_lock(lock: Path) -> None:
    while True:
        try:
            lock.mkdir(exist_ok=False)
            return
        except FileExistsError:
            if time.time() - lock.stat().st_mtime > LOCK_STALE_AFTER_S:
                shutil.rmtree(lock, ignore_errors=True)
            else:
                time.sleep(1)


def prepare_runtime_archive(config: NOOARuntimeConfig) -> Path:
    """Return a content-addressed, portable runtime archive for ``source=auto``."""
    if config.source != "auto":
        raise ValueError("prepare_runtime_archive requires runtime.source=auto")
    cache_root = Path(config.cache_dir).expanduser()
    if not cache_root.is_absolute():
        cache_root = AGENT_DIR / cache_root
    cache_root.mkdir(parents=True, exist_ok=True)
    recipe = _runtime_recipe(config)
    runtime_dir = cache_root / f"runtime-{recipe}"
    archive = cache_root / f"runtime-{recipe}.tar.gz"
    sentinel = runtime_dir / ".installed"
    if archive.is_file() and sentinel.is_file() and sentinel.read_text().strip() == recipe:
        return archive

    lock = cache_root / f".runtime-{recipe}.lockdir"
    _acquire_lock(lock)
    try:
        if archive.is_file() and sentinel.is_file() and sentinel.read_text().strip() == recipe:
            return archive
        shutil.rmtree(runtime_dir, ignore_errors=True)
        runtime_dir.mkdir()
        env = os.environ.copy()
        env.update(
            {
                "NOOA_RUNTIME_DIR": str(runtime_dir),
                "NOOA_RUNTIME_REQUIREMENTS": str(AGENT_DIR / "runtime-requirements.txt"),
                "NOOA_PYTHON_VERSION": config.python_version,
                "NOOA_PBS_RELEASE": config.python_build_standalone_release,
                "NOOA_RUNTIME_ARCH": config.architecture,
            }
        )
        subprocess.run(["bash", str(AGENT_DIR / "setup_runtime.sh")], check=True, env=env)
        if not (runtime_dir / "bin" / "python3").is_file():
            raise RuntimeError("portable NOOA runtime setup did not produce bin/python3")
        sentinel.write_text(recipe, encoding="utf-8")
        temporary = cache_root / f".{archive.name}.{uuid4().hex}.tmp"
        try:
            with tarfile.open(temporary, "w:gz", compresslevel=1) as output:
                output.add(runtime_dir, arcname=".")
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
        return archive
    finally:
        shutil.rmtree(lock, ignore_errors=True)
