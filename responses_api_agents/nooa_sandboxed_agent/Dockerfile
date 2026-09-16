# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM python:3.13-slim

ENV LITELLM_LOCAL_MODEL_COST_MAP=True

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/nooa

COPY runtime-requirements.txt /tmp/nooa-runtime-requirements.txt
RUN /opt/nooa/bin/pip install --no-cache-dir --requirement /tmp/nooa-runtime-requirements.txt \
    && rm /tmp/nooa-runtime-requirements.txt
