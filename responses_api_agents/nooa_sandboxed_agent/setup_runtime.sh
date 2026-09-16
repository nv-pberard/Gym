# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#!/bin/bash
set -euo pipefail

export PYTHONNOUSERSITE=1

: "${NOOA_RUNTIME_DIR:?NOOA_RUNTIME_DIR is required}"
: "${NOOA_RUNTIME_REQUIREMENTS:?NOOA_RUNTIME_REQUIREMENTS is required}"
: "${NOOA_PYTHON_VERSION:?NOOA_PYTHON_VERSION is required}"
: "${NOOA_PBS_RELEASE:?NOOA_PBS_RELEASE is required}"
: "${NOOA_RUNTIME_ARCH:?NOOA_RUNTIME_ARCH is required}"

portable_python_can_run() {
    "$NOOA_RUNTIME_DIR/bin/python3" -c "" >/dev/null 2>&1
}

install_packages() {
    if portable_python_can_run; then
        "$NOOA_RUNTIME_DIR/bin/python3" -m pip install --no-cache-dir "$@"
        return
    fi
    command -v uv >/dev/null || {
        echo "uv is required to prepare a cross-platform NOOA runtime" >&2
        return 1
    }
    uv pip install \
        --prefix "$NOOA_RUNTIME_DIR" \
        --python-version "$NOOA_PYTHON_VERSION" \
        --python-platform "$NOOA_RUNTIME_ARCH" \
        "$@"
}

mkdir -p "$NOOA_RUNTIME_DIR"
if [ ! -x "$NOOA_RUNTIME_DIR/bin/python3" ]; then
    runtime_url="https://github.com/astral-sh/python-build-standalone/releases/download/${NOOA_PBS_RELEASE}/cpython-${NOOA_PYTHON_VERSION}+${NOOA_PBS_RELEASE}-${NOOA_RUNTIME_ARCH}-install_only.tar.gz"
    echo "Downloading portable NOOA Python: $runtime_url"
    curl -fsSL "$runtime_url" | tar xz -C "$NOOA_RUNTIME_DIR" --strip-components=1
fi

if portable_python_can_run; then
    install_packages --upgrade pip
fi
install_packages --requirement "$NOOA_RUNTIME_REQUIREMENTS"
