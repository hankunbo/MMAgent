from types import SimpleNamespace

import pytest

from packages.evaluation_preparation.generic_aliases import alias_review
from packages.evaluation_preparation.generic_capability_projection import (
    GenericCapabilityOwner,
    GenericCapabilityPhase,
    project_demands,
)
from packages.grounding.models import digest
from tests.evaluation_preparation.test_dynamic_evidence import dynamic_fixture


def fixture(*, with_wire=False):
    f = dynamic_fixture(3)
    seed = f.owner.current()
    aliased, _ = alias_review(seed)
    old = seed.policy.requirements[0]
    old_alias = aliased.policy.requirements[0]
    covered = dict(
        id="active:EXISTING",
        definition=old.definition.model_dump(mode="json"),
        input_modalities=["SLO"],
        basis_kind="qualitative",
        missing_reason=None,
        acquisition_allowed=True,
        acquisition_method="vision",
    )
    added = dict(covered, id="active:NEW", missing_reason="not_yet_observed")
    missing = dict(
        covered,
        id="active:MISSING",
        input_modalities=["OCT"],
        acquisition_allowed=False,
        missing_reason="patient_input_missing:OCT",
    )
    snapshot = dict(
        original_case=seed.material.scope.case_code,
        original_input_checksum=seed.material.scope.input_checksum,
        task_alias=dict(
            canonical_target=seed.material.goal_code, selection_target="GOAL:GLAUCOMA_DETECTION"
        ),
        old_requirement_ids_preserved=[r.id for r in seed.policy.requirements],
        selected_requirements=[covered, added, missing],
    )
    policy = dict(
        schema_version="generic-requirement-correspondence/1.0.0",
        scope="legacy_generic_source_root_v2_glaucoma_detection",
        canonical_selection_target="GOAL:GLAUCOMA_DETECTION",
        observation_coverage_not_inferred=True,
        new_source_interpretation_not_adopted=True,
        clinical_equivalence_not_claimed=True,
        rules=[
            dict(
                selected=covered["id"],
                existing=old.id,
                relation="reuse_existing_question_scope_not_adopt_new_interpretation",
                selected_definition_checksum=digest(covered["definition"]),
                existing_definition_checksum=digest(old_alias.definition),
                selected_modalities=["SLO"],
                existing_modalities=["SLO"],
                reason="synthetic_registered_scope",
            )
        ],
    )
    medical = {s.reference.code: s for s in seed.material.sources}
    result = (seed, snapshot, medical, aliased.material.sources, policy)
    return (result, f) if with_wire else result


def test_retains_old_requirements_facts_and_groups_and_selects_only_new_visible_work():
    args = fixture()
    seed, snapshot, medical, previous, policy = args
    before = seed.model_dump_json()
    result, manifest = project_demands(*args)
    assert len(result.policy.requirements) == len(seed.policy.requirements) + 2
    assert result.policy.items == seed.policy.items
    assert result.material.observations == seed.material.observations
    assert result.policy.groups[: len(seed.policy.groups)] == seed.policy.groups
    assert manifest["dispatch_targets"] == ["active:NEW"]
    assert manifest["requirements_after"] == manifest["requirements_before"] + 2
    assert manifest["covered_by_existing_scope"][0]["existing"] == seed.policy.requirements[0].id
    assert manifest["new_observations"] == manifest["new_reviews"] == 0
    assert not manifest["historical_scores_relabelled"]
    assert seed.model_dump_json() == before
    assert project_demands(*args) == (result, manifest)
    aliased, _ = alias_review(result, previous=previous)
    assert aliased.material.sources == previous
    assert next(r for r in aliased.policy.requirements if r.id == "active:MISSING").missing_reason


@pytest.mark.parametrize(
    "mutation",
    [
        "input",
        "target",
        "old_definition",
        "new_definition",
        "missing_source",
        "duplicate_rule",
        "modality",
    ],
)
def test_rejects_changed_current_context_or_declared_correspondence(mutation):
    seed, snapshot, medical, previous, policy = fixture()
    if mutation == "input":
        snapshot["original_input_checksum"] = digest("other")
    elif mutation == "target":
        snapshot["task_alias"]["canonical_target"] = "GOAL:OTHER"
    elif mutation == "old_definition":
        policy["rules"][0]["existing_definition_checksum"] = digest("other")
    elif mutation == "new_definition":
        policy["rules"][0]["selected_definition_checksum"] = digest("other")
    elif mutation == "missing_source":
        medical = {}
    elif mutation == "duplicate_rule":
        policy["rules"] *= 2
    else:
        snapshot["selected_requirements"][1]["input_modalities"] = ["color_fundus"]
    with pytest.raises(ValueError, match="generic_capability_"):
        project_demands(seed, snapshot, medical, previous, policy)


def test_original_acquisition_projection_accepts_selected_new_demand():
    from packages.evaluation_preparation.dynamic_stages import acquisition_state
    from packages.evidence_details.review_tasks import project_review_tasks

    seed, snapshot, medical, previous, policy = fixture()
    state, manifest = project_demands(seed, snapshot, medical, previous, policy)
    image = SimpleNamespace(dimensions=(2, 3), asset_code="ASSET:INPUT", checksum=digest("image"))
    acquired = acquisition_state(
        state, "active:NEW", image, allowed_targets=manifest["dispatch_targets"]
    )
    assert [i.id for i in acquired.policy.items] == ["active:NEW"]
    assert acquired.material.observations[0].status == "unknown"
    assert project_review_tasks(acquired).tasks[0].assessment_id == "active:NEW"
    assert acquired.material.goal_code == seed.material.goal_code


def test_phase_rejects_nonoriginal_qualification_owner():
    with pytest.raises(ValueError, match="original_campaign_owner_required"):
        GenericCapabilityOwner(object())
    with pytest.raises(ValueError, match="phase_owner_binding"):
        GenericCapabilityPhase(object(), selection_owner=object())


def test_linked_missing_requirement_stays_missing_without_duplicate_demand():
    seed, snapshot, medical, previous, policy = fixture()
    requirements = list(seed.policy.requirements)
    requirements[0] = requirements[0].model_copy(
        update=dict(missing_reason="patient_modality_not_provided")
    )
    seed = seed.model_copy(
        update=dict(policy=seed.policy.model_copy(update=dict(requirements=tuple(requirements))))
    )
    result, manifest = project_demands(seed, snapshot, medical, previous, policy)
    assert result.policy.requirements[0].missing_reason == "patient_modality_not_provided"
    assert manifest["linked_missing_requirement_ids"] == [requirements[0].id]
    assert manifest["scope_links_are_observation_completion"] is False
    assert "active:EXISTING" not in {r.id for r in result.policy.requirements}
    assert manifest["new_observations"] == 0


def test_new_requirement_observation_reaches_original_provider_acceptance_and_score():
    from apps.vectorengine_case_materials import ref
    from packages.evaluation_preparation.comprehensive_materials import merge_partial
    from packages.evaluation_preparation.quality_materials import upgrade
    from packages.grounding.models import validate
    from packages.llm_provider.models import ObservationInput
    from tests.agent_provider.batch_support import generate
    from tests.evaluation_preparation.test_partial_observation import proposal

    args, wire = fixture(with_wire=True)
    seed, snapshot, medical, previous, policy = args
    seed = upgrade(seed)
    expanded, manifest = project_demands(seed, snapshot, medical, previous, policy)
    value = proposal().observation_value()
    observed = seed.material.observations[0].model_copy(
        update=dict(
            reference=ref("observation", "OBS:NEW_CAPABILITY", digest(value)),
            value=value,
            method="live_image_observation",
            unit="qualitative",
        )
    )
    reviewed = merge_partial(expanded, {"active:NEW": validate(ObservationInput, observed)})
    reviewed, _ = alias_review(reviewed, previous=previous)
    wire.owner.state = reviewed
    submission = generate(wire)
    wire.acceptance.submit(submission, key="capability-new-requirement")
    assert wire.acceptance.accept(key="capability-new-requirement").status == "accepted"
    saved = wire.acceptance.read(key="capability-new-requirement")[2]
    assert "active:NEW" in {r["id"] for r in saved["scores"]["reviews"]}
    assert "active:MISSING" not in {r["id"] for r in saved["scores"]["reviews"]}
    assert len(reviewed.policy.requirements) == manifest["requirements_after"]
    original_alias, _ = alias_review(seed, previous=previous)
    assert (
        reviewed.policy.requirements[: len(seed.policy.requirements)]
        == original_alias.policy.requirements
    )
    # Source aliases are reversible; original input/observations have not changed.
    assert seed.material.observations[0].reference != observed.reference
    assert len(wire.budget.records()) == 1


def test_increment_can_review_only_new_item_while_old_judgments_remain_compatible():
    from apps.vectorengine_case_materials import ref
    from packages.evaluation_preparation.comprehensive_materials import merge_partial
    from packages.evaluation_preparation.generic_pending_review import selected_state
    from packages.evaluation_preparation.quality_materials import upgrade
    from packages.evidence_details.accepted_review_reuse import compatible_local_ids
    from tests.agent_provider.batch_support import generate
    from tests.evaluation_preparation.test_partial_observation import proposal

    args, original = fixture(with_wire=True)
    seed, snapshot, medical, previous, policy = args
    seed = upgrade(seed)
    old, _ = alias_review(seed, previous=previous)
    original.owner.state = old
    accepted = generate(original, key="original-judgments")
    original.acceptance.submit(accepted, key="original-judgments")
    assert original.acceptance.accept(key="original-judgments").status == "accepted"
    original_candidate = accepted.candidate.model_dump_json()
    expanded, _ = project_demands(seed, snapshot, medical, previous, policy)
    value = proposal().observation_value()
    observed = seed.material.observations[0].model_copy(
        update=dict(
            reference=ref("observation", "OBS:NEW_CAPABILITY", digest(value)),
            value=value,
            method="live_image_observation",
            unit="qualitative",
        )
    )
    full = merge_partial(expanded, {"active:NEW": observed})
    full, _ = alias_review(full, previous=previous)
    retained = compatible_local_ids(old, full, accepted.candidate, include_unassessed=True)
    assert set(retained) == {i.assessment_id for i in accepted.candidate.items}
    _, fresh = fixture(with_wire=True)
    fresh.owner.state = selected_state(full, ["active:NEW"])
    increment = generate(fresh, key="new-only")
    fresh.acceptance.submit(increment, key="new-only")
    assert fresh.acceptance.accept(key="new-only").status == "accepted"
    assert {i.assessment_id for i in increment.candidate.items} == {"active:NEW"}
    assert accepted.candidate.model_dump_json() == original_candidate
    assert not set(retained) & {i.assessment_id for i in increment.candidate.items}
