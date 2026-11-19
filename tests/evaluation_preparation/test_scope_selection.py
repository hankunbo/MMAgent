"""Real projection owners; injected scope facts here, durable acceptance in PG."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from packages.dataset_adapters.multimodal.adapters import prepare
from packages.dataset_adapters.reading import grayscale_image
from packages.grounding.models import digest
from packages.multimodal_research import scope_selection
from packages.multimodal_research.active_requirements import load
from packages.object_store import checksum
from tests.dataset_adapters.test_multimodal import gamma
from tests.evaluation_preparation import test_input_scope_phase as scope_fixture

PROFILE = "configs/research/multimodal_catalog_acquisition_v4.json"


def gamma_views(tmp_path):
    prepared = prepare("gamma", gamma(tmp_path / "data"), "0001")
    output = tmp_path / "run"
    directory = output / "views"
    directory.mkdir(parents=True)
    views = []
    for c in prepared.manifest.components:
        if not c.payload_ids:
            continue
        raw = grayscale_image((8, 8), bytes(64))
        name = checksum(c.id.encode())[7:31] + ".png"
        (directory / name).write_bytes(raw)
        views.append(
            dict(
                component=c.id,
                status="obtained",
                file=name,
                checksum=checksum(raw),
                width=8,
                height=8,
                modality=c.modality,
                indices=[0],
                available_members=1,
                inspected_fraction=1,
                transform=dict(kind="synthetic_display"),
                limits=["Synthetic fixture, not a patient observation"],
            )
        )
    (directory / "views.json").write_text(
        json.dumps(
            dict(
                input_checksum=prepared.manifest.checksum,
                views=views,
            )
        )
    )
    return prepared, output


def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(scope_fixture, "PROFILE", PROFILE)
    monkeypatch.setattr(scope_fixture, "prepared_view", gamma_views)
    return scope_fixture.fixture(tmp_path)


def facts(base, visibility="partly_visible"):
    return [
        dict(
            task=e["task"],
            phase=dict(run_id="SYNTHETIC", target=e["target"]),
            records_checksum=digest(e),
            receipt_checksum=digest(("receipt", e)),
            observation=dict(reference=dict(code="OBS:SYNTHETIC", checksum=digest(e))),
            regions=[
                dict(
                    region=r,
                    visibility=visibility,
                    coverage=0 if visibility == "not_identified" else 0.7,
                    clarity=0.6,
                    finding="synthetic view",
                    limitation="sampling only",
                )
                for r in e["task"]["requested_regions"]
            ],
        )
        for e in base.input_scope_plan["entries"]
    ]


@pytest.mark.parametrize(
    "visibility,native,action",
    [
        ("visible", True, "current_native_owner_then_qualitative_fallback"),
        ("partly_visible", True, "current_native_owner_then_qualitative_fallback"),
        ("uncertain", False, "vision_scope_clarification"),
        ("not_identified", False, "vision_scope_clarification"),
    ],
)
def test_visibility_changes_method_not_medical_inventory(
    tmp_path, monkeypatch, visibility, native, action
):
    base, _, _ = setup(tmp_path, monkeypatch)
    before = digest(base.ledger)
    plan = scope_selection.project(base, facts(base, visibility))
    cup = next(r for r in plan["rows"] if r["id"] == "active:cup_geometry")
    assert cup["native_candidate"] is native and cup["action"] == action
    assert len(plan["rows"]) == len(base.ledger) == 18
    assert sum(r["state"] == "missing_input" for r in plan["rows"]) == 2
    assert digest(base.ledger) == before and plan["removed_requirements"] == []
    assert not plan["physical_scale_authorized"] and plan["added_disease_votes"] == 0


def test_initial_install_defers_native_even_if_runner_provided(tmp_path, monkeypatch):
    from packages.multimodal_research.active_acquisition import install

    base, _, _ = setup(tmp_path, monkeypatch)
    calls = []
    install(base, load(PROFILE), native_runner=lambda *a, **kw: calls.append("native"))
    assert calls == [] and base.active_owner.native is None
    assert base.initial_active_acquisition["native"] is None


def test_unassessed_scope_keeps_original_question_and_requires_clarification(tmp_path, monkeypatch):
    base, _, _ = setup(tmp_path, monkeypatch)
    plan = scope_selection.project(base, [])
    cup = next(r for r in plan["rows"] if r["id"] == "active:cup_geometry")
    assert cup["state"] == "scope_not_assessed" and not cup["native_candidate"]
    assert cup["action"] == "vision_scope_clarification"


def test_projection_context_not_a_task_or_score_and_binding_required(tmp_path, monkeypatch):
    from types import SimpleNamespace

    base, _, _ = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="scope_selection_before_medical_phase_required"):
        scope_selection.context_observations(base, "active:rim_diffuse")
    plan = scope_selection.project(base, facts(base))
    base.scope_selection_owner = SimpleNamespace(check=lambda: plan)
    context = scope_selection.context_observations(base, "active:rim_diffuse")
    assert len(context) == 1
    value = json.loads(context[0].value)
    assert value["state"] == "partly_observable"
    assert value["requirement"] == "active:rim_diffuse"
    assert context[0].image is None and context[0].method == "input_scope_execution_context"
    assert len(base.ledger) == 18 and not base.observations


def test_region_policy_and_new_schema_are_exact(tmp_path):
    profile = load(PROFILE)
    policy = scope_selection.read_policy(profile)
    path = Path("packages/multimodal_research/schemas/ScopeSelectionPolicy.json")
    assert json.loads(path.read_bytes()) == policy.model_json_schema()
    changed = deepcopy(profile)
    changed["scope_selection_policy"]["checksum"] = digest("changed")
    with pytest.raises(ValueError, match="scope_selection_policy_changed"):
        scope_selection.read_policy(changed)


def test_native_readiness_is_not_consulted_before_scope(tmp_path, monkeypatch):
    from packages.multimodal_research import entry_preflight

    base, _, _ = setup(tmp_path, monkeypatch)

    def unexpected(**kwargs):
        raise AssertionError("native readiness before accepted scope")

    monkeypatch.setattr(entry_preflight, "native_readiness", unexpected)
    assert (
        entry_preflight.check_local_route(base.manifest, base.task, base.active_owner.profile, None)
        == "native_method_deferred_to_accepted_input_scope"
    )


def test_committed_plan_missing_mirror_recovers_without_new_write_or_dispatch(tmp_path):
    from contextlib import nullcontext
    from types import SimpleNamespace

    calls = []
    plan = dict(
        schema_version="input-scope-method-plan/1.0.0",
        scope_records=[dict(phase=dict(run_id="SYNTHETIC"))],
        rows=[],
    )
    saved = {}
    docs = SimpleNamespace(
        read=lambda k: saved.get(k),
        session=nullcontext,
        write=lambda k, v: (saved.update({k: v}), calls.append("write")),
    )
    campaign = SimpleNamespace(
        campaign_id="synthetic-memory-plan",
        output=tmp_path,
        documents=lambda _: docs,
        guard_control=lambda: calls.append("guard"),
    )
    owner = scope_selection.SelectionOwner(campaign)
    owner.current = lambda: plan
    assert owner.save() == plan
    _, _, path = owner.location(plan)
    path.unlink()  # Model a lost local mirror; the original record remains.
    assert owner.save() == plan
    assert calls == ["guard", "write"]


def test_original_review_projection_carries_context_without_creating_score_items(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from packages.multimodal_research.campaign import Phase, configuration

    base, _, seed = setup(tmp_path, monkeypatch)
    plan = scope_selection.project(base, facts(base))
    base.scope_selection_owner = SimpleNamespace(check=lambda: plan)
    rid = seed.material.scope.run_id
    base.contexts[rid] = SimpleNamespace(run_id=rid, request=SimpleNamespace(task=base.task))
    phase = Phase(base, observation_reader=lambda: {})
    state, _ = phase.project(run_id=rid, config=configuration(directional=True))
    contexts = [
        o for o in state.material.observations if o.method == "input_scope_execution_context"
    ]
    assert len(contexts) == 16
    assert len(state.policy.requirements) == 18
    context_codes = {o.reference.code for o in contexts}
    assert not any(p.observation_code in context_codes for p in state.policy.items)
