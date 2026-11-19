"""A new interpretation references an existing method; it never invents a fact."""

import copy

import pytest

from packages.evidence_reuse.definitions import locate_draft
from packages.grounding.models import digest, validate
from packages.llm_provider.models import TaskDefinition
from tests.evaluation_preparation.test_definition_dependency_scope import fixture
from tests.evaluation_preparation.test_definition_expansion import Sources, draft


def candidate():
    value = draft("GOAL:DEPENDENCY_TEST")
    value.update(
        schema_version="source-definition-draft/5.1.0",
        acquisition="confirmed_observation",
        requirement_alias="requirement:0",
        method="synthetic_confirmed_geometry",
        unit="ratio",
        dependency_contracts=[],
    )
    for excerpt in value["excerpts"]:
        excerpt.pop("start")
        excerpt.pop("end")
        excerpt.update(prefix=None, suffix=None)
    return value


def test_located_confirmed_interpretation_keeps_method_and_exact_quote():
    value = candidate()
    before = copy.deepcopy(value)
    resolved = locate_draft(value, target=value["target"], source_owner=Sources())
    assert value == before
    assert resolved.schema_version == "source-definition-draft/5.0.0"
    assert resolved.acquisition == "confirmed_observation"
    assert resolved.method == "synthetic_confirmed_geometry"
    assert resolved.requirement_alias == "requirement:0"
    assert resolved.excerpts[0].start == 0


@pytest.mark.parametrize(
    "alias", ["PRIVATE_CASE", "requirement:-1", "requirement:00", "requirement:64"]
)
def test_invalid_public_alias_rejected(alias):
    value = candidate()
    value["requirement_alias"] = alias
    with pytest.raises(ValueError):
        locate_draft(value, target=value["target"], source_owner=Sources())


def test_old_draft_cannot_silently_enable_confirmed_method():
    value = candidate()
    value["schema_version"] = "source-definition-draft/4.1.0"
    with pytest.raises(ValueError):
        locate_draft(value, target=value["target"], source_owner=Sources())


def test_new_draft_without_confirmed_owner_never_writes():
    owner, _, _, documents = fixture()
    with pytest.raises(ValueError, match="definition_confirmed_requirement_owner_missing"):
        owner.accept("missing-owner", candidate())
    assert documents.read("definition:missing-owner") is None


def confirmed_fixture(tmp_path, *, technical=False):
    from types import SimpleNamespace

    from tests.evaluation_preparation.test_component_observations import installed_fixture

    owner, manifest, sources, documents = fixture()
    previous = owner.accept("previous", draft(owner.binding["target"]))
    item = installed_fixture(tmp_path)
    if technical:
        from packages.evaluation_preparation.quality import mask_quality
        from tests.tool_execution.test_geometry_qc import mask_png

        mask = mask_png((4, 4), bytes([255] * 16))
        qc = mask_quality(mask, mask)
        item.record.update(qc=qc, quality=qc["quality_factor"])
        item.origin["adapter"] = "confirmed-fundus-geometry/1.0.0"
        saved = copy.deepcopy((item.value, item.record, item.origin))
        item.binding = digest(saved)
        item.read = lambda: copy.deepcopy(saved)
    item.base.manifest = manifest
    item.base.task = SimpleNamespace(target_code=owner.binding["target"])
    item.base.ledger[0].update(
        definition=validate(TaskDefinition, previous["definition"]),
        component="image-a",
        input_modalities=["new_image"],
        acquisition_method="disc_cup_measurement",
    )
    item.input_row = copy.deepcopy(item.base.ledger[0])
    item.input_binding = digest((item.input_row, item.input_resources))
    item.install()
    owner.confirmed_observations = item
    owner.requirement_ids = (item.target,)

    def no_extra_dependencies(contract):
        raise AssertionError("This fixture declares no additional dependency reads")

    owner.fact_reader = no_extra_dependencies
    value = candidate()
    value["method"] = "confirmed_fundus_disc_cup_geometry"
    return owner, value, item, previous, documents


def test_original_binding_receipt_keeps_fact_and_previous_definition(tmp_path):
    owner, value, item, previous, documents = confirmed_fixture(tmp_path)
    before = copy.deepcopy((item.base.ledger, item.base.observations, item.report()))
    receipt = owner.accept("interpretation", value)
    assert receipt["schema_version"] == "accepted-research-definition/4.0.0"
    assert receipt["confirmed_basis"]["requirement"]["id"] == item.target
    assert receipt["confirmed_basis"]["observation"] == item.observation().reference.model_dump(
        mode="json"
    )
    assert receipt["published"] is False
    assert owner.read("interpretation") == receipt
    assert owner.read("previous") == previous
    assert (item.base.ledger, item.base.observations, item.report()) == before
    from packages.evidence_reuse.definition_acceptance import requirement_rows

    with pytest.raises(ValueError, match="definition_confirmed_projection_owner_required"):
        requirement_rows(receipt, item.base.manifest)
    assert documents.read("definition:interpretation")[0] == receipt


@pytest.mark.parametrize(
    "field,value",
    [
        ("method", "wrong"),
        ("unit", "mm"),
        ("examination", "different"),
        ("requirement_alias", "requirement:1"),
    ],
)
def test_confirmed_mismatch_no_new_receipt(tmp_path, field, value):
    owner, candidate_, _, _, documents = confirmed_fixture(tmp_path)
    candidate_[field] = value
    with pytest.raises(ValueError, match="definition_confirmed_requirement"):
        owner.accept("wrong", candidate_)
    assert documents.read("definition:wrong") is None


def test_confirmed_current_detects_changed_fact_historical_is_read_only(tmp_path):
    owner, value, item, _, _ = confirmed_fixture(tmp_path)
    receipt = owner.accept("interpretation", value)
    item.value["tampered"] = True
    with pytest.raises(ValueError, match="component_observation_changed"):
        owner.read("interpretation")
    assert owner.read("interpretation", historical=True) == receipt


def test_same_id_interpretation_install_and_current_recheck(tmp_path):
    owner, value, item, _, _ = confirmed_fixture(tmp_path)
    before = copy.deepcopy((item.base.observations, item.input_row, item.report()))
    receipt = owner.accept("interpretation", value)
    adoption = item.prepare_interpretation(owner, "interpretation")
    rows = copy.deepcopy(item.base.ledger)
    rows[0].update(adoption["changes"])
    item.install_interpretation(owner, adoption, rows)
    item.check()
    assert owner.read("interpretation") == receipt
    assert (item.base.observations, item.input_row, item.report()) == before
    assert len(item.base.ledger) == 1
    item.install_interpretation(owner, adoption, copy.deepcopy(rows))
    owner.sources.value["state"] = "withdrawn"
    with pytest.raises(ValueError, match="definition_draft_exact_excerpt"):
        item.check()
    assert owner.read("interpretation", historical=True) == receipt


def test_interpretation_cannot_change_other_projection_fields(tmp_path):
    owner, value, item, _, _ = confirmed_fixture(tmp_path)
    owner.accept("interpretation", value)
    adoption = item.prepare_interpretation(owner, "interpretation")
    before = copy.deepcopy(item.base.ledger)
    rows = copy.deepcopy(before)
    rows[0].update(adoption["changes"], acquisition_method="different")
    with pytest.raises(ValueError, match="component_observation_projection_changed"):
        item.install_interpretation(owner, adoption, rows)
    assert item.base.ledger == before
    item.check()


def test_interpretation_rejects_boolean_integer_alias(tmp_path):
    owner, value, item, _, _ = confirmed_fixture(tmp_path)
    owner.accept("interpretation", value)
    adoption = item.prepare_interpretation(owner, "interpretation")
    rows = copy.deepcopy(item.base.ledger)
    adoption["changes"]["review_allowed"] = int(adoption["changes"]["review_allowed"])
    rows[0].update(adoption["changes"])
    with pytest.raises(ValueError, match="component_interpretation_projection_binding"):
        item.install_interpretation(owner, adoption, rows)


def test_interpretation_does_not_retain_caller_adoption(tmp_path):
    owner, value, item, _, _ = confirmed_fixture(tmp_path)
    owner.accept("interpretation", value)
    adoption = item.prepare_interpretation(owner, "interpretation")
    rows = copy.deepcopy(item.base.ledger)
    rows[0].update(adoption["changes"])
    item.install_interpretation(owner, adoption, rows)
    adoption["changes"]["review_allowed"] = "changed"
    item.check()
    assert item.interpretation_adoption["changes"]["review_allowed"] is True


@pytest.mark.parametrize("reason", ["cancelled", "deadline"])
def test_saved_interpretation_restores_without_dispatch_permission(tmp_path, reason):
    owner, value, item, _, documents = confirmed_fixture(tmp_path)
    receipt = owner.accept("interpretation", value)

    def stopped():
        raise ValueError(reason)

    owner.guard = stopped
    with pytest.raises(ValueError, match=reason):
        owner.accept("new", value)
    assert documents.read("definition:new") is None
    adoption = item.prepare_interpretation(owner, "interpretation")
    rows = copy.deepcopy(item.base.ledger)
    rows[0].update(adoption["changes"])
    item.install_interpretation(owner, adoption, rows)
    item.check()
    assert owner.read_for_report("interpretation") == receipt
    with pytest.raises(ValueError, match=reason):
        owner.read("interpretation")
    owner.sources.value["state"] = "withdrawn"
    with pytest.raises(ValueError, match="definition_draft_exact_excerpt"):
        item.check()


def test_confirmed_receipt_rejects_boolean_integer_alias(tmp_path):
    owner, value, _, _, _ = confirmed_fixture(tmp_path)
    receipt = copy.deepcopy(owner.accept("interpretation", value))
    receipt["dependency_assessment"]["review_allowed"] = 1
    with pytest.raises(ValueError, match="definition_confirmed_receipt_binding"):
        owner.project_confirmed(receipt)


def test_new_provider_wire_uses_confirmed_proposal_schema(tmp_path):
    from packages.llm_provider import DeterministicFakeProvider
    from packages.multimodal_research.definition_expansion import definition_config
    from tests.evaluation_preparation.test_definition_expansion import fixture as expansion_fixture

    e = expansion_fixture(tmp_path)
    config = definition_config(
        e.provider.config, prompt_version="definition-draft-5.0.0", source_output_tokens=8192
    )
    seen = []

    def response(body):
        seen.append(body)
        assert "requirement_alias" in str(body)
        assert "source-definition-draft/5.1.0" in str(body)
        return dict(
            schema_version="research-definition-proposal/4.0.0",
            definitions=[candidate()],
            no_addition_reason=None,
        )

    result = DeterministicFakeProvider(config, [response]).generate_definitions(
        payload={},
        scope=e.scope,
        budget=e.budget,
        before_request=lambda: None,
    )
    assert result.definitions[0].requirement_alias == "requirement:0"
    assert len(seen) == e.budget.used == 1


def test_revision_installs_same_requirement_without_tool_reschedule(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from packages.multimodal_research.definition_expansion import DefinitionExpansion
    from tests.evaluation_preparation.test_definition_expansion import fixture as expansion_fixture

    owner, value, item, _, _ = confirmed_fixture(tmp_path)
    seed = expansion_fixture(tmp_path)
    base = item.base
    base.task.target_code = value["target"] = "GOAL:ADVANCED_GLAUCOMA"
    base.read_dependency_input = lambda: base.manifest
    base.views = dict(views=[dict(component="image-a", status="obtained")])
    seed.campaign.base = base
    policy = json.loads(Path("configs/research/multimodal_definition_gate_v21.json").read_bytes())
    policy["public_target"] = base.task.target_code
    policy["query_requirements"] = {q: [item.target] for q in policy["public_queries"]}
    e = DefinitionExpansion(
        seed.campaign,
        documents=seed.documents,
        sources=owner.sources,
        provider=seed.provider,
        budget=seed.budget,
        scope=seed.scope,
        policy=policy,
    )
    receipt = e.owner.accept("confirmed-interpretation", value)
    projection = e.project(receipt)[0]
    row = dict(
        e.original_rows[0],
        **{
            k: projection[k]
            for k in (
                "definition",
                "basis_kind",
                "definition_receipt_checksum",
                "dependency_assessment",
                "acquisition_allowed",
                "review_allowed",
            )
        },
        definition_receipt_key="confirmed-interpretation",
        expected_method=projection["expected_method"],
        expected_unit=projection["expected_unit"],
    )
    replacement = dict(
        id=item.target,
        definition="confirmed-interpretation",
        before=e.original_rows[0]["definition"].model_dump(mode="json"),
        after=receipt["definition"],
        reason="target_relation_rebound",
        review_required=True,
    )
    revision = e.commit_revision(
        [row],
        {owner.sources.value["code"]: owner.sources.value},
        {"confirmed-interpretation": receipt},
        [],
        [],
        [],
        replaced=[replacement],
    )
    assert revision["denominator_before"] == revision["denominator_after"] == 1
    assert not revision["added"] and e.campaign.targets == ()
    assert base.ledger[0]["id"] == item.target
    assert base.ledger[0]["definition"].reference == projection["definition"].reference
    item.check()
    with monkeypatch.context() as patch:
        facts = []
        original_check = item.check_fact

        def counted():
            facts.append(True)
            return original_check()

        patch.setattr(item, "check_fact", counted)
        assert e.current() == revision
        assert len(facts) == 2, "one owner reconstruction, before/after fact verification"
    assert e.install(e.current()) == revision
    assert e.budget.used == 0

    # New component/expansion instances reconstruct only the saved original fact
    # and immutable revision; do not retain the first installed projection.
    cold_owner, _, cold_item, _, _ = confirmed_fixture(tmp_path / "cold")
    cold_seed = expansion_fixture(tmp_path / "cold", documents=seed.documents)
    cold_base = cold_item.base
    cold_base.task.target_code = base.task.target_code
    cold_base.read_dependency_input = lambda: cold_base.manifest
    cold_base.views = copy.deepcopy(base.views)
    cold_seed.campaign.base = cold_base
    cold_seed.campaign.protocol = copy.deepcopy(e.campaign.protocol)
    cold = DefinitionExpansion(
        cold_seed.campaign,
        documents=e.documents,
        sources=cold_owner.sources,
        provider=cold_seed.provider,
        budget=cold_seed.budget,
        scope=e.scope,
        policy=policy,
    )
    assert cold.binding == e.binding
    assert cold.install(cold.current()) == revision
    cold_item.check()
    assert cold_base.ledger == base.ledger
    assert cold_base.observations == base.observations
    assert cold.budget.used == 0 and cold.campaign.targets == ()


def test_source_proposal_advances_to_same_id_interpretation(tmp_path):
    exercise_source_proposal(tmp_path)


def exercise_source_proposal(tmp_path, *, typed=False, candidate_choices=False, feedback=False):
    import json
    from pathlib import Path

    from packages.llm_provider import DeterministicFakeProvider
    from packages.multimodal_research.definition_expansion import (
        DefinitionExpansion,
        definition_config,
    )
    from tests.evaluation_preparation.test_definition_expansion import fixture as expansion_fixture

    if typed:
        from tests.evaluation_preparation.test_owner_dependency_definition import confirmed

        owner, value, item, _, _ = confirmed(tmp_path)
    else:
        owner, value, item, _, _ = confirmed_fixture(tmp_path)
    seed = expansion_fixture(tmp_path)
    base = item.base
    base.task.target_code = value["target"] = "GOAL:ADVANCED_GLAUCOMA"
    base.read_dependency_input = lambda: base.manifest
    base.views = dict(views=[dict(component="image-a", status="obtained")])
    seed.campaign.base = base
    policy = json.loads(
        Path(
            "configs/research/multimodal_definition_gate_v22.json"
            if typed
            else "configs/research/multimodal_definition_gate_v21.json"
        ).read_bytes()
    )
    policy["query_requirements"] = {q: [item.target] for q in policy["public_queries"]}
    if candidate_choices:
        assert typed
        policy["schema_version"] = "campaign-definition-expansion/1.28.0"
        policy["prompt_version"] = "definition-draft-6.4.0"
        policy["unresolved_relation_retrieval"] = None
        policy["max_rounds"] = len(policy["public_queries"])
        # The original synthetic fixture has a 30-minute total window. Keep
        # its final-review reserve explicit, without changing public budgets.
        policy["reserve_review_seconds"] = 60
        for passage_plan in policy["passage_plans"].values():
            passage_plan["schema_version"] = "definition-passage-plan/5.0.0"
    if feedback:
        assert candidate_choices
        seed.scope = seed.scope.model_copy(
            update={"case_code": "CASE:SYNTHETIC_DEFINITION_FEEDBACK"}
        )
        policy["schema_version"] = "campaign-definition-expansion/1.29.0"
        policy["prompt_version"] = "definition-draft-6.5.0"
    # Only external search/material responses are fixtures. Proposal parsing,
    # source/definition ownership, revision and component installation are real.
    sources = owner.sources
    sources.index_keys = ("synthetic-index",)
    sources.restore_indexes = lambda indexes: None
    sources.search = lambda **kw: dict(
        schema_version="research-definition-search/1.1.0",
        materials=[sources.read(sources.value["code"])],
        acquired_indexes=[],
        backlog=[],
    )
    sources.reconcile_pool = lambda **kw: dict(
        schema_version="research-definition-pool/1.0.0",
        materials=[sources.read(sources.value["code"])],
        backlog=[],
        retrieval={"synthetic": True},
    )

    def response(body):
        if typed:
            payload = json.loads(body["messages"][1]["content"][0]["text"])
            assert "source-definition-draft/7.1.0" in body["messages"][0]["content"]
            assert payload["input_capabilities"][0]["input_alias"] == "input:0"
            assert payload["confirmed_observation_capabilities"][0]["method"] == value["method"]
            if candidate_choices:
                choices = payload["candidate_contract_choices"]
                selected = choices["confirmed_observation"][0]
                assert selected["copy_fields"]["method"] == value["method"]
                assert selected["copy_fields"]["unit"] == value["unit"]
                assert selected["method_condition_selectors"]["predicate"] == "confirmed_method"
                assert "candidate_contract_choices" in body["messages"][0]["content"]
            else:
                assert "candidate_contract_choices" not in payload
        return dict(
            schema_version="research-definition-proposal/5.0.0"
            if typed
            else "research-definition-proposal/4.0.0",
            definitions=[value],
            no_addition_reason=None,
        )

    provider_config = seed.provider.config
    replies = [response]
    if feedback:
        from packages.grounding.models import canonical
        from packages.llm_provider.config import review_format_upgrade
        from packages.llm_provider.transport import Reply

        provider_config = review_format_upgrade(provider_config)

        def malformed(body):
            raw = response(body)
            raw = copy.deepcopy(raw)
            raw["definitions"][0]["dependency_contracts"][0]["input_alias"] = "input:0"
            return Reply(content=canonical(raw), finish_reason="stop", response_confirmed=True)

        def corrected(body):
            extra = json.loads(body["messages"][-1]["content"])
            assert "research-definition-proposal/5.0.0" in extra["task"]
            assert any(
                x.get("constraint") == "definition_dependency_confirmed_selector"
                for x in extra["errors"]
            )
            return response(body)

        replies = [malformed, corrected]
    provider = DeterministicFakeProvider(
        definition_config(
            provider_config, prompt_version=policy["prompt_version"], source_output_tokens=8192
        ),
        replies,
    )
    e = DefinitionExpansion(
        seed.campaign,
        documents=seed.documents,
        sources=sources,
        provider=provider,
        budget=seed.budget,
        scope=seed.scope,
        policy=policy,
    )
    original = copy.deepcopy(base.observations)
    revision = e.advance()
    assert len(revision["replaced"]) == 1 and not revision["added"], revision
    assert revision["denominator_before"] == revision["denominator_after"] == 1
    assert base.observations == original and e.campaign.targets == ()
    assert base.ledger[0]["id"] == item.target
    assert e.budget.used == (2 if feedback else 1)
    item.check()
    assert e.advance() == revision and e.budget.used == (2 if feedback else 1)
    if feedback:
        from packages.llm_provider.review_correction import verify_correction

        records = e.budget.records()
        assert records[0].error == "schema" and records[1].error is None
        assert records[1].retry_index == 1
        verify_correction(records[1], e.budget)
