"""Definition owner -> selected original Provider/acceptance, synthetic only."""

import copy
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation.generic_definition_projection import (
    phase_type,
    project_interpretations,
)
from packages.evidence_details.acceptance_models import DomainState
from packages.grounding.models import digest, validate
from tests.agent_provider.batch_support import generate
from tests.evaluation_preparation.test_dynamic_evidence import dynamic_fixture
from tests.evaluation_preparation.test_generic_definition_facts import fixture as facts_fixture


def fixture():
    f = facts_fixture()
    wire = dynamic_fixture(6)
    state = wire.owner.current()
    raw = state.model_dump(mode="json")
    old_id = raw["policy"]["items"][0]["id"]

    def rename(value):
        if isinstance(value, dict):
            return {k: rename(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rename(v) for v in value]
        return f.target if value == old_id else value

    raw = rename(raw)
    definition = f.current["facts"][f.target]["requirement"]["definition"]
    row = next(r for r in raw["policy"]["requirements"] if r["id"] == f.target)
    row.update(definition=copy.deepcopy(definition), input_modalities=["new_image"])
    item = next(i for i in raw["policy"]["items"] if i["id"] == f.target)
    old_code = item["observation_code"]
    observed = f.state.material.observations[0]
    item.update(
        definition=copy.deepcopy(definition),
        observation_code=observed.reference.code,
        method=observed.method,
        unit=observed.unit,
        source_codes=[r["code"] for r in definition["source_refs"]],
    )
    raw["material"]["observations"] = [
        observed.model_dump(mode="json") if o["reference"]["code"] == old_code else o
        for o in raw["material"]["observations"]
    ]
    raw["material"]["sources"].append(f.sources.value["material"])
    scope = raw["material"]["scope"]
    f.domain.binding.update(
        input_checksum=scope["input_checksum"], target=raw["material"]["goal_code"]
    )
    f.candidate["target"] = f.domain.binding["target"]
    f.current.update(
        input_checksum=scope["input_checksum"],
        target=f.domain.binding["target"],
        run_id="RUN:PREVIOUS_CONFIRMED",
    )
    fact = f.current["facts"][f.target]
    fact["origin"].update(input_checksum=scope["input_checksum"], target=f.domain.binding["target"])
    fact["observation_binding"] = digest((observed, fact["origin"]))
    f.base.visible_value = SimpleNamespace(checksum=scope["input_checksum"])
    f.base.visible.read = lambda _: f.base.visible_value
    f.owner.snapshot = copy.deepcopy(f.current)
    f.owner.snapshot_checksum = digest(f.current)
    scope["policy_checksum"] = raw["material"]["policy_ref"]["checksum"] = digest(raw["policy"])
    scope.pop("material_checksum")
    scope["material_checksum"] = digest(raw["material"])
    state = validate(DomainState, raw)
    wire.owner.state = state
    receipt = f.domain.accept("new-definition", f.candidate)
    return f, wire, state, receipt


def test_current_definition_reaches_original_provider_acceptance_and_score():
    f, wire, seed, receipt = fixture()
    before = digest(seed)
    state, manifest = project_interpretations(seed, f.domain, ["new-definition"])
    assert len(state.policy.items) == 1
    assert len(state.policy.requirements) == len(seed.policy.requirements)
    assert state.material.observations == seed.material.observations
    assert state.policy.groups == seed.policy.groups
    assert state.policy.items[0].definition.model_dump(mode="json") == receipt["definition"]
    untouched = [r for r in seed.policy.requirements if r.id != f.target]
    assert [r for r in state.policy.requirements if r.id != f.target] == untouched
    wire.owner.state = state
    submission = generate(wire)
    assert wire.acceptance.submit(submission, key="revised")["status"] == "submitted_not_accepted"
    assert wire.acceptance.accept(key="revised").status == "accepted"
    saved = wire.acceptance.read(key="revised")[2]
    assert {r["id"] for r in saved["scores"]["reviews"]} == {f.target}
    assert manifest["added_requirements"] == []
    assert manifest["interpretation_revisions"][0]["receipt_checksum"] == digest(receipt)
    assert manifest["whole_case_aggregation_required"]
    assert digest(seed) == before
    assert len(wire.budget.records()) == 1
    assert project_interpretations(seed, f.domain, ["new-definition"]) == (state, manifest)
    assert len(wire.budget.records()) == 1


@pytest.mark.parametrize(
    "mutation", ["old_run", "wrong_input", "wrong_goal", "observation", "definition"]
)
def test_changed_scope_or_fact_cannot_consume_revision(mutation):
    f, _, seed, _ = fixture()
    raw = seed.model_dump(mode="json")
    if mutation == "old_run":
        raw["material"]["scope"]["run_id"] = f.current["run_id"]
    elif mutation == "wrong_input":
        raw["material"]["scope"]["input_checksum"] = digest("other")
    elif mutation == "wrong_goal":
        raw["material"]["goal_code"] = "GOAL:OTHER"
    elif mutation == "observation":
        raw["material"]["observations"][0]["value"] = "different measured value"
    else:
        definition = raw["policy"]["requirements"][0]["definition"]
        definition["examination"] = "changed examination"
        definition["reference"]["checksum"] = digest(
            {k: v for k, v in definition.items() if k != "reference"}
        )
        raw["policy"]["items"][0]["definition"] = copy.deepcopy(definition)
    # Construct deliberately contradictory input without asking the Schema to
    # reject it first, to verify the controlled projection's own current checks.
    from packages.llm_provider.models import Material

    raw["material"]["scope"].pop("material_checksum")
    raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
    material = validate(Material, raw["material"])
    policy = validate(type(seed.policy), raw["policy"])
    changed = seed.model_copy(update=dict(material=material, policy=policy))
    with pytest.raises(ValueError, match="generic_interpretation_"):
        project_interpretations(changed, f.domain, ["new-definition"])


def test_source_revocation_and_duplicate_selection_never_dispatch():
    f, wire, seed, _ = fixture()
    with pytest.raises(ValueError, match="receipt_selection"):
        project_interpretations(seed, f.domain, ["new-definition", "new-definition"])
    f.sources.value["state"] = "withdrawn"
    with pytest.raises(ValueError, match="definition_draft_exact_excerpt"):
        project_interpretations(seed, f.domain, ["new-definition"])
    assert len(wire.budget.records()) == 0


def test_phase_composition_rejects_nonoriginal_owner():
    f, _, _, _ = fixture()
    with pytest.raises(ValueError, match="phase_owner_binding"):
        phase_type(SimpleNamespace(), f.domain, ["new-definition"])
    phase = phase_type(f.owner, f.domain, ["new-definition"])(f.base, target="another")
    with pytest.raises(ValueError, match="no_new_acquisition"):
        phase.project_current()
    with pytest.raises(ValueError, match="phase_owner_binding"):
        phase_type(f.owner, SimpleNamespace(), ["new-definition"])


@pytest.mark.parametrize("keys", [[], [None], [False], [""], ["new-definition"] * 2])
def test_receipt_selection_is_explicit_and_bounded(keys):
    f, wire, seed, _ = fixture()
    with pytest.raises(ValueError, match="receipt_selection"):
        project_interpretations(seed, f.domain, keys)
    assert len(wire.budget.records()) == 0


def test_examination_only_definition_cannot_be_used_for_direction_review():
    f, wire, seed, _ = fixture()
    candidate = dict(f.candidate, relation_use="examination_only")
    receipt = f.domain.accept("examination-only", candidate)
    assert receipt["definition"]["interpretation_status"] == "missing"
    with pytest.raises(ValueError, match="not_reviewable"):
        project_interpretations(seed, f.domain, ["examination-only"])
    assert len(wire.budget.records()) == 0


def test_phase_record_is_rebuilt_before_original_short_write(monkeypatch):
    from packages.evaluation_preparation.generic_aliases import GenericPhase
    from packages.evaluation_preparation.generic_definition_projection import projection_record

    f, _, seed, _ = fixture()
    f.base.project = lambda **_: (seed, {})
    f.base.verify = lambda _: None
    # Only the synthetic base source projection is supplied here; the real phase
    # adapter, definition owner, current receipt/source checks and record execute.
    monkeypatch.setattr(GenericPhase, "project_current", lambda _: seed)
    phase = phase_type(f.owner, f.domain, ["new-definition"])(f.base)
    state, _ = phase.project(run_id=seed.material.scope.run_id, config=None)
    record = projection_record(phase, state, state.material.scope.run_id)
    assert record["projected_state_checksum"] == digest(state)
    record["selected_ids"].clear()
    assert phase.definition_revision_manifest["selected_ids"] == [f.target]
    with pytest.raises(ValueError, match="phase_record_binding"):
        projection_record(phase, state, "RUN:OTHER")
    with pytest.raises(ValueError, match="phase_producer_required"):
        projection_record(
            SimpleNamespace(definition_revision_manifest=record), state, state.material.scope.run_id
        )
    f.sources.value["state"] = "withdrawn"
    with pytest.raises(ValueError, match="definition_draft_exact_excerpt"):
        projection_record(phase, state, state.material.scope.run_id)
