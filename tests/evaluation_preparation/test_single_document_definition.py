"""Opt-in wire contract; never strip commentary to turn an old failure into acceptance."""

import json
from pathlib import Path

import pytest

from packages.llm_provider import DeterministicFakeProvider
from packages.llm_provider.provider import strict_json
from packages.multimodal_research.definition_expansion import (
    definition_config,
    read_expansion_policy,
)
from tests.evaluation_preparation.test_definition_expansion import fixture


def test_new_prompt_reaches_original_provider_and_old_policy_stays_old(tmp_path):
    from packages.multimodal_research.active_requirements import load

    assert (
        load("configs/research/multimodal_active_requirements_v6.json")["definition_expansion"][
            "prompt_version"
        ]
        == "definition-draft-6.1.0"
    )
    new = read_expansion_policy(
        json.loads(Path("configs/research/multimodal_definition_gap_loop_v2.json").read_text())
    )
    old = read_expansion_policy(
        json.loads(Path("configs/research/multimodal_definition_gap_loop_v1.json").read_text())
    )
    assert new.prompt_version == "definition-draft-6.1.0"
    assert old.prompt_version == "definition-draft-6.0.0"
    active = json.loads(Path("configs/research/multimodal_active_requirements_v6.json").read_text())
    assert active["definition_expansion"] == new.model_dump(mode="json") or (
        read_expansion_policy(active["definition_expansion"]) == new
    )
    e = fixture(tmp_path)
    seen = []

    def respond(body):
        seen.append(body)
        prompt = body["messages"][0]["content"]
        assert "exactly ONE JSON document" in prompt
        assert "research-definition-proposal/5.0.0" in str(body)
        return dict(
            schema_version="research-definition-proposal/5.0.0",
            definitions=[],
            no_addition_reason="No new supported research definition.",
        )

    config = definition_config(
        e.provider.config, prompt_version=new.prompt_version, source_output_tokens=8192
    )
    assert config.transport.max_retries == 0
    result = DeterministicFakeProvider(config, [respond]).generate_definitions(
        payload={},
        scope=e.scope,
        budget=e.budget,
        before_request=lambda: None,
    )
    assert not result.definitions and len(seen) == e.budget.used == 1


@pytest.mark.parametrize(
    "raw",
    [
        b'{"commentary":"planning"}{"schema_version":"research-definition-proposal/5.0.0"}',
        b'{"definitions":[],"definitions":[{}]}',
    ],
)
def test_multiple_documents_and_duplicate_keys_remain_rejected(raw):
    with pytest.raises(ValueError):
        strict_json(raw)


def test_connector_diagnostics_do_not_expose_arbitrary_error_text():
    from packages.knowledge_connectors import ConnectorError, ErrorCategory
    from packages.multimodal_research.stage_failure import safe_error

    error = ConnectorError(ErrorCategory.RETRY_EXHAUSTED, last_error=ErrorCategory.TIMEOUT)
    assert safe_error(error) == "knowledge_retry_exhausted"
    assert safe_error(ValueError("private-arbitrary-text")) == "local_owner_error"
