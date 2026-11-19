"""Source feedback names the real field violation without fixing a model answer."""

import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from packages.evidence_reuse.definitions import OwnerBoundDefinitionProposal
from packages.grounding.models import canonical, validate
from packages.llm_provider.definition_format import fields, project
from packages.llm_provider.provider import DeterministicFakeProvider, ProviderError
from packages.llm_provider.transport import Reply
from tests.agent_provider.test_definition_feedback import setup
from tests.agent_provider.test_definition_identity import inputs
from tests.evaluation_preparation.test_owner_dependency_definition import confirmed


def missing_patient(good, *, raw=False):
    result = deepcopy(good)
    draft = result["definitions"][0]
    condition = dict(
        index=0,
        text="Separate functional evidence is still required.",
        kind="patient_input",
        blocks=["interpretation"],
        modality="VF",
        concept="visual_field",
        rationale="Do not remove this condition to obtain an assessed result.",
        excerpt_indices=[0],
        predicate="unresolved",
        input_alias=None,
        requirement_alias=None,
        method=None,
        unit=None,
        quality_check=None,
    )
    draft.update(dependencies=[condition["text"]], dependency_contracts=[condition])
    if raw:
        draft.update(
            schema_version="source-definition-draft/6.1.0",
            input_aliases=["input:0"],
            acquisition="vision",
            method="live_image_observation",
            unit="qualitative",
        )
        draft.pop("requirement_alias")
    validate(OwnerBoundDefinitionProposal, result)
    return result


def bad_input(good):
    bad = deepcopy(good)
    bad["definitions"][0]["dependency_contracts"][0].update(modality=None, concept=None)
    return bad


@pytest.mark.parametrize("raw", [False, True])
def test_precise_selected_variant_feedback_not_other_union_branch(tmp_path, raw):
    _, _, source, _ = setup(tmp_path)
    good = missing_patient(source, raw=raw)
    bad = bad_input(good)
    before = deepcopy(bad)
    with pytest.raises(ValidationError) as caught:
        validate(OwnerBoundDefinitionProposal, bad)
    value = fields(caught.value, bad)
    assert len(value) == 1
    assert value[0]["path"] == ["definitions", 0, "dependency_contracts", 0]
    assert value[0]["constraint"] == "definition_dependency_input_selector"
    assert "modality and concept" in value[0]["contract"]
    assert bad == before


def test_length_feedback_has_actual_bound_without_input_or_arbitrary_messages(tmp_path):
    _, _, source, _ = setup(tmp_path)
    good = missing_patient(source, raw=True)
    good["definitions"][0]["excerpts"][0]["suffix"] = "private-canary" * 30
    with pytest.raises(ValidationError) as caught:
        validate(OwnerBoundDefinitionProposal, good)
    value = fields(caught.value, good)
    assert any(v["path"][-1] == "suffix" and v.get("max_length") == 240 for v in value)
    assert "private-canary" not in str(value)


def test_request_projection_explicit_unresolved_input_contract_is_copy(tmp_path):
    _, _, good, _ = setup(tmp_path)
    payload = inputs(good)
    before = deepcopy(payload)
    value = project(payload)
    assert payload == before
    rules = value["definition_format_contract"]["constraint_rules"]
    assert "including unresolved" in rules["definition_dependency_input_selector"]
    assert "null owner selectors" in rules["definition_dependency_input_selector"]


def test_provider_corrects_format_once_original_owner_keeps_missing_dependency(tmp_path):
    e, config, good, _ = setup(tmp_path, prompt="definition-draft-6.7.0")
    owner, owner_draft, _, _, _ = confirmed(tmp_path / "owner")
    good["definitions"] = [owner_draft]
    good = missing_patient(good)
    bad = bad_input(good)
    raw = canonical(bad)

    def corrected(body):
        feedback = json.loads(body["messages"][-1]["content"])
        assert feedback["errors"][0]["constraint"] == "definition_dependency_input_selector"
        payload = json.loads(body["messages"][1]["content"][0]["text"])
        assert payload["definition_format_contract"]
        return good

    result = DeterministicFakeProvider(
        config,
        [
            Reply(content=raw, response_confirmed=True, finish_reason="stop", api_requests=1),
            corrected,
        ],
    ).generate_definitions(
        payload=inputs(good), scope=e.scope, budget=e.budget, before_request=lambda: None
    )
    assert e.budget.used == 2 and raw == canonical(bad)
    receipt = owner.accept("format-only", result.definitions[0])
    assert not receipt["dependency_assessment"]["review_allowed"]
    assert receipt["definition"]["interpretation_status"] == "missing"
    assert result.definitions[0].dependencies == tuple(good["definitions"][0]["dependencies"])


def test_second_bad_response_has_diagnostic_and_stops_without_forcing_answer(tmp_path):
    e, config, good, _ = setup(tmp_path, prompt="definition-draft-6.7.0")
    good = missing_patient(good)
    bad = Reply(content=canonical(bad_input(good)), response_confirmed=True, finish_reason="stop")
    with pytest.raises(ProviderError, match="schema"):
        DeterministicFakeProvider(config, [bad, bad, good]).generate_definitions(
            payload=inputs(good), scope=e.scope, budget=e.budget, before_request=lambda: None
        )
    assert e.budget.used == 2
    records = [
        e
        for e in e.budget.journal.read()
        if e.get("schema_version") == "source-definition-contract-diagnostic/1.0.0"
    ]
    assert len(records) == 2 and all(not r["candidate_accepted"] for r in records)
