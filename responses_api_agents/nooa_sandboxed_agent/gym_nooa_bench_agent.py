# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small public-entrypoint adapter for NOOA's benchmark-neutral BenchAgent."""

from nooa_bench.bench_agent import BenchAgent


class GymBenchAgent(BenchAgent):
    """Expose BenchAgent's Harbor evaluation contract to the Gym runner."""

    async def solve(self, task: str) -> dict:
        result = await self._run_evaluation({"user_message": task})
        if not result.get("success"):
            raise RuntimeError(result.get("error") or "NOOA BenchAgent did not complete the task")
        return result
