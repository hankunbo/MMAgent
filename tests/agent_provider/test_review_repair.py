"""08C repair: actual wire tasks, authoritative acquisition, immutable history."""

import copy
import json

import pytest

from agents.provider_agents import StructuredRoleAgent
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.review_tasks import project_review_tasks
from packages.evidence_scoring.accepted import build_accepted
from packages.grounding.models import digest, validate
from packages.llm_provider import DeterministicFakeProvider
from packages.llm_provider.config import load_provider_config
from packages.llm_provider.models import ReviewProposal, ReviewTemplate, TaskDefinition
from tests.agent_provider.support import (
    Documents,
    budget_for,
    definition,
    materials,
    review,
    submission,
)
from tests.agent_provider.test_review_boundaries import rebind, round_trip, two_item_wire


def missing_review(scope, status="input_missing", available=False):
    raw = review(scope).model_dump(mode="json")
    raw["items"][0].update(
        status=status,
        observation_available=available,
        judgment=dict(
            value=None,
            lower=None,
            upper=None,
            reason="Synthetic unresolved observation",
            missing_reason="Synthetic missing input or reference",
            method="not_assessed",
        ),
    )
    return validate(ReviewProposal, raw)


def observation_state(reader, status, value):
    observations = reader.state.material.model_dump(mode="json")["observations"]
    observations[0].update(status=status, value=value)
    rebind(reader, observations=observations)


@pytest.mark.parametrize(
    "status,value,item_status,available,coverage",
    [
        ("unknown", None, "input_missing", False, 0.0),
        ("unknown", "Unknown VLM finding", "input_missing", False, 0.0),
        ("missing", None, "input_missing", False, 0.0),
        ("ungradable", "Image cannot be assessed", "observation_ungradable", False, 0.0),
        ("obtained", "Synthetic measured feature", "reference_missing", True, 1.0),
        ("obtained", "Synthetic measured feature", "method_mismatch", True, 0.0),
        ("obtained", "Synthetic intermediate output", "intermediate", True, 0.0),
    ],
)
@pytest.mark.parametrize("zero_denominator", [False, True])
def test_state_matrix_persisted(status, value, item_status, available, coverage, zero_denominator):
    reader, b = materials()
    observation_state(reader, status, value)
    if zero_denominator:
        group = reader.state.policy.groups[0].model_copy(
            update={"importance": dict(slo_image_research=0.0, comprehensive_research=0.0)}
        )
        rebind(reader, policy=reader.state.policy.model_copy(update={"groups": (group,)}))
    candidate = missing_review(reader.state.material.scope, item_status, available)
    before = reader.state.model_dump_json()
    service, receipt, _ = round_trip(reader, b, candidate)
    assert receipt.status == "accepted"
    saved = service.read(key="review-boundary")
    score = saved[2]["scores"]
    assert score["schema_version"] == "case-research-scoring/6.1.0"
    for cov in score["root"]["coverage"].values():
        assert cov == dict(
            denominator=0.0 if zero_denominator else 1.0,
            acquisition=None if zero_denominator else coverage,
            interpretation=None if zero_denominator else 0.0,
        )
    assert score["root"]["summary"]["confidence"]["value"] is None
    assert score["reviews"][0]["judgment"]["value"] is None
    assert service.accept(key="review-boundary") == receipt
    assert service.read(key="review-boundary") == saved
    assert reader.state.model_dump_json() == before


@pytest.mark.parametrize(
    "status,value,item_status,available",
    [
        ("unknown", None, "reference_missing", True),
        ("unknown", "Uncertain VLM observation", "reference_missing", True),
        ("unknown", None, "reference_missing", False),
        ("ungradable", "Unusable", "observation_ungradable", True),
        ("missing", None, "reference_missing", True),
        ("obtained", None, "reference_missing", True),
        ("obtained", "", "reference_missing", True),
        ("obtained", "Present", "reference_missing", False),
        ("obtained", "Present", "observation_ungradable", True),
    ],
)
def test_forged_availability_or_status_rejected(status, value, item_status, available):
    reader, b = materials()
    observation_state(reader, status, value)
    c = missing_review(reader.state.material.scope, item_status, available)
    service, receipt, _ = round_trip(reader, b, c)
    assert receipt.status == "rejected"
    assert service.read(key="review-boundary") is None


def test_installed_missing_definition_preserves_acquired_not_interpreted():
    reader, b = materials()
    item = reader.state.policy.items[0].model_copy(
        update={"definition": validate(TaskDefinition, definition(interpretation=None))}
    )
    rebind(reader, policy=reader.state.policy.model_copy(update={"items": (item,)}))
    c = missing_review(reader.state.material.scope, "reference_missing", True)
    service, receipt, provider = round_trip(reader, b, c)
    assert receipt.status == "accepted"
    assert service.read(key="review-boundary")[2]["scores"]["root"]["coverage"][
        "slo_image_research"
    ] == dict(denominator=1.0, acquisition=1.0, interpretation=0.0)
    wire = json.loads(provider.transport.calls[0]["messages"][1]["content"][0]["text"])
    assert wire["review_tasks"]["tasks"][0]["definition"]["interpretation_status"] == "missing"
    with pytest.raises(ValueError, match="applicability"):
        build_accepted(reader.state, review(reader.state.material.scope))


def test_wire_order_independent_and_old_answers_absent():
    first, reordered = two_item_wire(True), two_item_wire(True, reverse=True)
    assert first["review_tasks"]["tasks"] == reordered["review_tasks"]["tasks"]
    tasks = first["review_tasks"]["tasks"]
    assert {t["assessment_id"]: t["observation"]["code"] for t in tasks} == {
        "item:one": "OBS:TWO",
        "item:two": "OBS:ONE",
    }
    assert {t["assessment_id"]: t["allowed_sources"][0]["code"] for t in tasks} == {
        "item:one": "SOURCE:ONE",
        "item:two": "SOURCE:TWO",
    }
    raw = json.dumps(first)
    assert all(
        key not in raw
        for key in ("fixed_model_row", "judgment", "prior_answers", "interpretations")
    )


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "scope",
        "reference",
        "definition",
        "unknown_observation",
        "method",
        "eye",
        "missing_definition",
    ],
)
def test_owner_and_projection_reject_invalid_tasks(change):
    reader, _ = materials()
    if change in {"duplicate", "scope", "reference"}:
        raw = project_review_tasks(reader.state).model_dump(mode="json")
        if change == "duplicate":
            raw["tasks"].append(copy.deepcopy(raw["tasks"][0]))
        elif change == "scope":
            raw["scope_checksum"] = digest("foreign")
        else:
            raw["tasks"][0]["observation"]["checksum"] = digest("foreign")
        with pytest.raises(ValueError):
            validate(ReviewTemplate, raw).check_material(reader.state.material)
        return
    p = reader.state.policy.items[0]
    if change == "definition":
        p = p.model_copy(
            update={"definition": validate(TaskDefinition, definition("SOURCE:FOREIGN"))}
        )
    elif change == "missing_definition":
        from packages.evidence_details.acceptance_models import TaskPolicy

        raw = reader.state.policy.model_dump(mode="json")
        raw["schema_version"] = "research-task-policy/1.0.0"
        del raw["items"][0]["definition"]
        raw["items"][0].pop("fixed_observation_ref", None)
        rebind(reader, policy=validate(TaskPolicy, raw))
    else:
        p = p.model_copy(
            update={"observation_code": "OBS:FOREIGN"}
            if change == "unknown_observation"
            else {"method": "wrong"}
            if change == "method"
            else {"laterality": "LEFT"}
        )
    if change != "missing_definition":
        rebind(reader, policy=reader.state.policy.model_copy(update={"items": (p,)}))
    with pytest.raises(ValueError):
        project_review_tasks(reader.state)


@pytest.mark.parametrize(
    "change",
    ["unknown_item", "duplicate_item", "wrong_observation", "wrong_source", "policy_drift"],
)
def test_result_binding_and_return_time_revalidation(change):
    reader, b = materials()
    config = load_provider_config(environ={})
    raw = review(reader.state.material.scope).model_dump(mode="json")

    def response(_):
        if change == "policy_drift":
            item = reader.state.policy.items[0].model_copy(update={"applicable": False})
            rebind(reader, policy=reader.state.policy.model_copy(update={"items": (item,)}))
        elif change == "duplicate_item":
            raw["items"].append(copy.deepcopy(raw["items"][0]))
        elif change == "unknown_item":
            raw["items"][0]["assessment_id"] = "item:foreign"
        else:
            raw["items"][0][
                "observation_citations" if change == "wrong_observation" else "reference_citations"
            ] = ["OBS:OTHER" if change == "wrong_observation" else "SOURCE:OTHER"]
        return raw

    provider = DeterministicFakeProvider(config, [response])
    budget = budget_for(b, config)
    with pytest.raises(ValueError, match="schema|scope"):
        StructuredRoleAgent("review", provider=provider, materials=reader, budget=budget).call(
            reader.state.material.scope, deadline=b.deadline, round_id="A"
        )
    assert budget.records()[-1].error in {"scope", "schema"}


def test_fixed_domain_row_is_assembled_without_prompt_answer():
    reader, b = materials()
    fixed = build_accepted(reader.state, review(reader.state.material.scope))["reviews"][0]
    fixed.update(id="item:model", role="model", kind="model", title="SECRET_OLD_MODEL_TITLE")
    fixed["judgment"]["method"] = "fixed_model_margin"
    from packages.evidence_scoring.continuous_models import ContinuousReview

    fixed = validate(ContinuousReview, fixed)
    p = reader.state.policy.items[0].model_copy(
        update={
            "id": "item:model",
            "role": "model",
            "kind": "model",
            "fixed_observation_ref": reader.state.material.observations[0].reference,
            "fixed_model_row": fixed,
        }
    )
    rebind(
        reader,
        policy=reader.state.policy.model_copy(update={"items": (*reader.state.policy.items, p)}),
    )
    service, receipt, provider = round_trip(reader, b, review(reader.state.material.scope))
    assert receipt.status == "accepted"
    wire = json.dumps(provider.transport.calls[0])
    assert "SECRET_OLD_MODEL_TITLE" not in wire and "fixed_model_row" not in wire
    score = service.read(key="review-boundary")[2]["scores"]
    assert next(r for r in score["reviews"] if r["id"] == "item:model") == fixed.model_dump(
        mode="json"
    )


def test_acceptance_reconstructs_wire_against_owner_not_caller_template():
    reader, b = materials()
    config = load_provider_config(environ={})
    original = reader.review_template

    def forged(scope):
        t = original(scope)
        task = t.tasks[0].model_copy(update={"node": "GOAL:UNAPPROVED"})
        return t.model_copy(update={"tasks": (task,)})

    reader.review_template = forged
    budget = budget_for(b, config)
    provider = DeterministicFakeProvider(
        config, [review(reader.state.material.scope).model_dump(mode="json")]
    )
    result = StructuredRoleAgent("review", provider=provider, materials=reader, budget=budget).call(
        reader.state.material.scope, deadline=b.deadline, round_id="A"
    )
    service = ResearchAcceptance(owner=reader, documents=Documents(), requests=budget)
    assert service.submit(submission(reader, result, budget), key="forged").status == "rejected"


def test_legacy_v6_exact_semantics_are_readable_but_not_new_admission():
    from packages.evidence_details.acceptance_models import TaskPolicy
    from packages.evidence_scoring.api import build
    from packages.evidence_scoring.api import validate as validate_score

    reader, b = materials()
    raw = reader.state.policy.model_dump(mode="json")
    raw["schema_version"] = "research-task-policy/1.0.0"
    del raw["items"][0]["definition"]
    raw["items"][0].pop("fixed_observation_ref", None)
    rebind(reader, policy=validate(TaskPolicy, raw))
    observation_state(reader, "unknown", None)
    candidate = missing_review(reader.state.material.scope, "reference_missing", True)
    base = dict(state=reader.state, candidate=candidate)
    historical = build(base, version="6.0.0")
    original = json.dumps(historical, sort_keys=True)
    assert historical["root"]["coverage"]["slo_image_research"]["acquisition"] == 1.0
    assert validate_score(base, historical) == historical
    with pytest.raises(ValueError, match="definition_upgrade"):
        build(base, version="6.1.0")
    with pytest.raises(ValueError, match="definition_upgrade"):
        round_trip(reader, b, candidate)
    assert json.dumps(historical, sort_keys=True) == original


@pytest.mark.parametrize("field", ["observation_citations", "reference_citations"])
def test_existing_but_wrong_task_reference_rejected(field):
    reader, b = materials()
    observations = reader.state.material.model_dump(mode="json")["observations"]
    observations.append(copy.deepcopy(observations[0]))
    from tests.agent_provider.support import ref

    observations[-1]["reference"] = ref("observation", "OBS:OTHER").model_dump(mode="json")
    sources = reader.state.material.model_dump(mode="json")["sources"]
    sources.append(copy.deepcopy(sources[0]))
    sources[-1]["reference"] = ref("source", "SOURCE:OTHER").model_dump(mode="json")
    rebind(reader, observations=observations, sources=sources)
    raw = review(reader.state.material.scope).model_dump(mode="json")
    raw["items"][0][field] = ["OBS:OTHER" if field == "observation_citations" else "SOURCE:OTHER"]
    with pytest.raises(ValueError, match="scope"):
        round_trip(reader, b, validate(ReviewProposal, raw))


def test_bound_materials_reads_current_domain_not_static_template():
    from packages.llm_provider.materials import BoundMaterials

    reader, _ = materials()
    m = reader.state.material
    bound = BoundMaterials(
        material=m,
        read_graph=lambda _: reader.state.material.scope.graph,
        read_visible=lambda _: reader.state.material.scope.visible_checksum,
        read_source=lambda ref, *_: next(
            s for s in reader.state.material.sources if s.reference == ref
        ),
        read_observation=lambda _, ref: next(
            o for o in reader.state.material.observations if o.reference == ref
        ),
        read_image=lambda _: b"",
        review_template=reader.current,
    )
    assert bound.review_template(m.scope) == project_review_tasks(reader.state)
    p = reader.state.policy.items[0].model_copy(update={"applicable": False})
    rebind(reader, policy=reader.state.policy.model_copy(update={"items": (p,)}))
    with pytest.raises(ValueError, match="owner_material_changed"):
        bound.review_template(m.scope)
