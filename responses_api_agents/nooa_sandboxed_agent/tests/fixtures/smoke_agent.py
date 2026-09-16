# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nooa import Agent, CodeActStrategy, PredictStrategy, strategy


class SmokeAgent(Agent):
    @strategy(PredictStrategy())
    async def solve(self, task: str) -> str:  # type: ignore[empty-body]
        """Solve the task and return a short string."""

        ...


class CodeActSmokeAgent(Agent):
    @strategy(CodeActStrategy())
    async def solve(self, task: str) -> str:  # type: ignore[empty-body]
        """Solve the task by executing Python."""

        ...
