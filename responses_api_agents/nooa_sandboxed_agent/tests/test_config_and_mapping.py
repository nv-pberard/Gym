# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from responses_api_agents.nooa_sandboxed_agent.config import (
    NOOAAgentSpec,
    NOOAArgumentBinding,
    NOOARuntimeConfig,
)
from responses_api_agents.nooa_sandboxed_agent.mapping import latest_user_text, materialize_arguments


def test_materialize_arguments_only_reads_explicit_safe_paths() -> None:
    row = {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": [{"type": "input_text", "text": "solve me"}]},
            ]
        },
        "verifier_metadata": {"answer": "secret"},
    }
    bindings = {
        "task": NOOAArgumentBinding(
            source="responses_create_params.input",
            transform="latest_user_text",
        )
    }

    assert materialize_arguments(row, bindings) == {"task": "solve me"}

    with pytest.raises(ValidationError, match="verifier field"):
        NOOAArgumentBinding(source="verifier_metadata.answer")

    for source in ("patch", "test_patch", "expected_answer", "reference_answer", "gold"):
        with pytest.raises(ValidationError, match="verifier field"):
            NOOAArgumentBinding(source=source)


def test_latest_user_text_uses_last_user_message() -> None:
    assert (
        latest_user_text(
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "last"},
            ]
        )
        == "last"
    )


def test_agent_spec_rejects_unsafe_names_and_reserved_llm() -> None:
    base = {
        "agent_class": "example.agent:ExampleAgent",
        "entrypoint": "run",
        "arguments": {},
    }
    with pytest.raises(ValidationError, match="model_aliases keys"):
        NOOAAgentSpec(**base, model_aliases={"_reviewer": "review_model"})
    with pytest.raises(ValidationError, match="argument names"):
        NOOAAgentSpec(**(base | {"arguments": {"class": {"source": "task"}}}))
    with pytest.raises(ValidationError, match="init_kwargs.llm"):
        NOOAAgentSpec(**base, init_kwargs={"llm": "not-allowed"})
    with pytest.raises(ValidationError, match="reserved or conflicting names"):
        NOOAAgentSpec(**base, model_aliases={"llm": "review_model"})
    with pytest.raises(ValidationError, match="reserved or conflicting names"):
        NOOAAgentSpec(**base, tool_namespace="reviewer_llm", model_aliases={"reviewer_llm": "review_model"})


def test_archive_runtime_requires_archive_path() -> None:
    with pytest.raises(ValidationError, match="archive_path is required"):
        NOOARuntimeConfig(source="archive")
    assert NOOARuntimeConfig(source="auto").archive_path is None
    with pytest.raises(ValidationError, match="only valid"):
        NOOARuntimeConfig(source="auto", archive_path="runtime.tar.gz")


def test_runtime_workdir_must_be_an_absolute_non_root_path() -> None:
    assert NOOARuntimeConfig(workdir="/app").workdir == "/app"
    with pytest.raises(ValidationError, match="absolute, non-root"):
        NOOARuntimeConfig(workdir="app")
    with pytest.raises(ValidationError, match="non-empty or null"):
        NOOARuntimeConfig(expected_nooa_version="")
