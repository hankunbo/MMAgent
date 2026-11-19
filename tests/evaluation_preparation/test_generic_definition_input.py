"""Original definition owner consumes confirmed Generic facts, not new patient data."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation.generic_definition_input import (
    GenericDefinitionInputView,
    public_proposition,
)
from packages.grounding.models import canonical, digest
from packages.multimodal_research.definition_expansion import (
    DefinitionExpansion,
    GenericConfirmedIndexPolicy,
    read_expansion_policy,
)
from tests.evaluation_preparation.test_definition_expansion import fixture as expansion_fixture
from tests.evaluation_preparation.test_generic_definition_projection import fixture as facts_fixture


def setup(monkeypatch, tmp_path):
    from packages.evaluation_preparation import generic_definition_input, generic_pending_review

    f, wire, state, _ = facts_fixture()
    f.current["state_checksum"] = digest(state)
    f.owner.snapshot = copy.deepcopy(f.current)
    f.owner.snapshot_checksum = digest(f.current)
    monkeypatch.setattr(
        generic_pending_review,
        "original_phase",
        lambda _: (None, state, SimpleNamespace(verify=lambda _: None), None),
    )
    public = dict(title="Synthetic research question", target=f.current["target"])
    monkeypatch.setattr(generic_definition_input, "public_proposition", lambda *a: public)
    view = GenericDefinitionInputView(f.owner)
    original = expansion_fixture(tmp_path)
    original.campaign.base = view
    original.campaign.guard = view.check
    policy = json.loads(
        Path(
            "docs/acceptance/grape-method-definition-context-ryxb297f/"
            "confirmed-cct-author-method-policy.json"
        ).read_bytes()
    )
    policy.pop("confirmed_method_metadata_profile")
    policy.pop("confirmed_method_metadata_targets")
    query = policy["public_queries"][0]
    policy.update(
        schema_version="campaign-definition-expansion/1.41.0",
        public_target=f.current["target"],
        confirmed_requirements=list(f.owner.selected),
        public_proposition_checksum=digest(view.public_goal),
        query_requirements={query: list(f.owner.selected)},
    )
    # Reuse the well-formed public retrieval shape, never the fixture's CCT
    # semantics as a claimed relation for the new selected measurement.
    new_query = "Cup area measurement research relationship"
    for name in ("query_requirements", "public_topic_sets", "passage_plans"):
        policy[name] = {new_query: policy[name][query]}
    policy["public_queries"] = [new_query]
    policy["public_topic_sets"] = {new_query: ["cup_disc_ratio"]}
    policy["passage_plans"][new_query].update(
        queries=["cup area relationship", "alternative artifact"],
        query_intents={"cup area relationship": "target", "alternative artifact": "alternative"},
    )
    e = DefinitionExpansion(
        original.campaign,
        documents=original.documents,
        sources=f.sources,
        provider=original.provider,
        budget=original.budget,
        scope=state.material.scope,
        policy=policy,
    )
    return f, wire, state, view, e, policy


def test_original_public_root_is_bound_without_using_old_patient_results():
    from db.runtime.repository.case_pipeline import identity
    from packages.evaluation_preparation.dynamic_materials import DynamicCaseEvidence

    base = SimpleNamespace(
        binding={"case_code": "CASE:SYNTHETIC"},
        knowledge_catalog=lambda: dict(tasks={}),
        root_version="generic-source-case-root/2.0.0",
    )
    base.structure = DynamicCaseEvidence.build_structure(base)
    root = next(n for n in base.structure["nodes"] if n["parent"] is None)
    fields = {k: root[k] for k in ("key", "node_type", "parent")}
    target = identity("GOAL", "CASE:SYNTHETIC" + root["key"])
    base.view = {"nodes": [dict(fields, node_code=target)]}
    value = public_proposition(base, target)
    assert value["title"] == "glaucoma" == root["key"]
    assert value["original_catalog_checksum"] == digest(base.knowledge_catalog())
    assert value["previous_patient_results_reused"] is False
    assert "GOAL:ADVANCED_GLAUCOMA" not in canonical(value).decode()
    with pytest.raises(ValueError, match="public_root_binding"):
        public_proposition(base, "GOAL:OTHER")
    base.view["nodes"].append(copy.deepcopy(base.view["nodes"][0]))
    with pytest.raises(ValueError, match="public_root_ambiguous"):
        public_proposition(base, target)


@pytest.mark.parametrize("fault", ["old_template_root", "catalog", "root_version"])
def test_original_source_driven_identity_is_not_old_template_identity(fault):
    from db.runtime.repository.case_pipeline import identity
    from packages.evaluation_preparation.dynamic_materials import DynamicCaseEvidence

    base = SimpleNamespace(
        binding={"case_code": "CASE:SYNTHETIC"},
        knowledge_catalog=lambda: dict(tasks={}),
        root_version="generic-source-case-root/2.0.0",
    )
    base.structure = DynamicCaseEvidence.build_structure(base)
    root = next(n for n in base.structure["nodes"] if n["parent"] is None)
    target = identity("GOAL", base.binding["case_code"] + root["key"])
    base.view = dict(nodes=[dict(root, node_code=target)])
    if fault == "old_template_root":
        root["key"] = base.view["nodes"][0]["key"] = "ROOT"
        target = base.view["nodes"][0]["node_code"] = identity(
            "GOAL", base.binding["case_code"] + "ROOT"
        )
    elif fault == "catalog":
        base.structure["definition_catalog_checksum"] = digest("changed")
    else:
        base.root_version = "caller-defined-root"
    with pytest.raises(ValueError, match="public_root_binding"):
        public_proposition(base, target)


def test_original_expansion_installs_revision_without_reacquisition(monkeypatch, tmp_path):
    f, _, state, view, e, _ = setup(monkeypatch, tmp_path)
    before = digest(state), digest(view.observations), digest(view.original_rows)
    assert type(e.policy) is GenericConfirmedIndexPolicy
    assert e.owner.confirmed_observations is f.owner
    receipt = e.owner.accept("original-definition-expansion", f.candidate)
    adoption = view.prepare_interpretation(e.owner, "original-definition-expansion")
    rows = copy.deepcopy(view.ledger)
    rows[0].update(adoption["changes"])
    revision = dict(
        rows=rows,
        replaced=[dict(id=f.target, definition=adoption["key"])],
        accepted={adoption["key"]: receipt},
        added=[],
        materials=receipt["sources"],
    )
    revision["review_source_scope"] = e.source_scope_record(revision)
    e.install(revision)
    view.check()
    assert e.campaign.targets == ()
    assert view.ledger[0]["review_allowed"]
    assert view.ledger[0]["expected_method"] == "same_mask_area_cdr"
    assert view.ledger[0]["expected_unit"] == "ratio"
    assert (digest(state), digest(view.observations), digest(view.original_rows)) == before
    e.install(revision)
    view.check()
    assert e.budget.used == 0


@pytest.mark.parametrize("fault", ["input", "ledger", "observation", "public_goal"])
def test_changed_bound_material_rejected(monkeypatch, tmp_path, fault):
    f, _, _, view, _, _ = setup(monkeypatch, tmp_path)
    if fault == "input":
        f.current["input_checksum"] = digest("changed input")
    elif fault == "ledger":
        view.ledger[0]["expected_unit"] = "mm"
    elif fault == "observation":
        view.observations[f.target] = view.observations[f.target].model_copy(update={"value": "9"})
    else:
        view.public_goal = dict(view.public_goal, title="Advanced disease")
    with pytest.raises(ValueError, match="generic_definition_"):
        view.check()


@pytest.mark.parametrize("fault", ["owner", "target", "public_binding", "selection"])
def test_policy_cannot_bypass_original_facts(monkeypatch, tmp_path, fault):
    f, _, _, view, e, policy = setup(monkeypatch, tmp_path)
    if fault == "owner":
        e.campaign.base = SimpleNamespace(**vars(view))
    elif fault == "target":
        policy["public_target"] = "GOAL:OTHER"
    elif fault == "public_binding":
        policy["public_proposition_checksum"] = digest("different root")
    else:
        policy["confirmed_requirements"] = [f.target, f.target]
    with pytest.raises(ValueError, match="generic_definition_|definition_public_target_binding"):
        DefinitionExpansion(
            e.campaign,
            documents=e.documents,
            sources=e.sources,
            provider=e.provider,
            budget=e.budget,
            scope=e.scope,
            policy=policy,
        )


def test_schema_new_version_does_not_expand_old_target_vocabulary(monkeypatch, tmp_path):
    _, _, _, _, _, policy = setup(monkeypatch, tmp_path)
    policy.update(schema_version="campaign-definition-expansion/1.39.0")
    policy.pop("confirmed_requirements")
    policy.pop("public_proposition_checksum")
    with pytest.raises(ValueError, match="definition_public_target_scope"):
        read_expansion_policy(policy)


def test_selected_source_provider_to_owner_to_original_review(monkeypatch, tmp_path):
    from packages.evaluation_preparation.generic_definition_projection import (
        project_interpretations,
    )
    from packages.llm_provider import definition_passages as passages
    from packages.llm_provider.provider import DeterministicFakeProvider
    from packages.multimodal_research.definition_expansion import definition_config
    from tests.agent_provider.batch_support import generate

    f, wire, seed, view, e, _ = setup(monkeypatch, tmp_path)
    payload = view.draft_context(
        dict(
            target=f.current["target"],
            public_queries=list(e.policy.public_queries),
            source_materials=[f.sources.read(f.sources.value["code"])],
        )
    )
    draft = copy.deepcopy(f.candidate)
    draft.update(schema_version="source-definition-draft/7.2.0")
    draft["excerpts"] = [dict(passage_id=next(iter(passages.choices(payload))))]
    response = dict(
        schema_version="research-definition-proposal/6.0.0",
        definitions=[draft],
        no_addition_reason=None,
    )
    before = canonical(response)
    config = definition_config(e.provider.config, prompt_version="definition-draft-6.8.0")
    result = DeterministicFakeProvider(config, [response]).generate_definitions(
        payload=payload,
        scope=e.scope,
        budget=e.budget,
        before_request=view.check,
    )
    assert canonical(response) == before
    receipt = e.owner.accept("wire-defined", result.definitions[0])
    state, manifest = project_interpretations(seed, e.owner, ["wire-defined"])
    wire.owner.state = state
    submission = generate(wire)
    wire.acceptance.submit(submission, key="new-source-review")
    assert wire.acceptance.accept(key="new-source-review").status == "accepted"
    assert set(
        r["id"] for r in wire.acceptance.read(key="new-source-review")[2]["scores"]["reviews"]
    ) == {f.target}
    assert manifest["interpretation_revisions"][0]["receipt_checksum"] == digest(receipt)
    assert e.budget.used == 1 and len(wire.budget.records()) == 1
    assert state.material.observations == seed.material.observations
    assert manifest["added_requirements"] == []


def test_actual_expansion_advance_consumes_source_and_installs_same_round(monkeypatch, tmp_path):
    from packages.llm_provider import definition_passages as passages
    from packages.llm_provider.provider import DeterministicFakeProvider
    from packages.multimodal_research.definition_expansion import definition_config

    f, _, state, view, e, _ = setup(monkeypatch, tmp_path)
    material = f.sources.read(f.sources.value["code"])
    f.sources.index_keys = tuple(e.policy.index_keys)
    f.sources.restore_indexes = lambda _: None
    f.sources.search = lambda **_: dict(
        schema_version="research-definition-search/1.1.0",
        materials=[material],
        acquired_indexes=[],
        backlog=[],
    )
    f.sources.reconcile_pool = lambda **_: dict(
        schema_version="research-definition-pool/1.0.0",
        materials=[material],
        backlog=[],
        retrieval={"test_only": True},
    )
    payload = dict(target=e.binding["target"], source_materials=[material])
    raw = dict(
        f.candidate,
        schema_version="source-definition-draft/7.2.0",
        excerpts=[dict(passage_id=next(iter(passages.choices(payload))))],
    )
    e.provider = DeterministicFakeProvider(
        definition_config(e.provider.config, prompt_version="definition-draft-6.8.0"),
        [
            dict(
                schema_version="research-definition-proposal/6.0.0",
                definitions=[raw],
                no_addition_reason=None,
            )
        ],
    )
    before = digest(state), digest(view.observations)
    revision = e.advance()
    assert revision["added"] == []
    assert [r["id"] for r in revision["replaced"]] == [f.target]
    assert revision["denominator_before"] == revision["denominator_after"] == 1
    assert view.ledger[0]["review_allowed"]
    assert e.campaign.targets == ()  # Confirmed measurement requires no second tool call.
    assert (digest(state), digest(view.observations)) == before
    assert e.current() == revision
    assert e.advance() == revision and e.budget.used == 1
    f.sources.value["state"] = "withdrawn"
    with pytest.raises(ValueError, match="source_changed|exact_excerpt"):
        e.current()
