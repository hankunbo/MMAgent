"""Input availability belongs to the original component, not the field name."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from packages.dataset_adapters.multimodal.adapters import prepare
from packages.dataset_adapters.multimodal.models import Component
from packages.grounding.models import digest
from packages.multimodal_research.active_acquisition import ActiveOwner
from packages.multimodal_research.active_requirements import load, project
from packages.multimodal_research.demand_coverage import project as coverage
from tests.dataset_adapters.test_multimodal import grape
from tests.evaluation_preparation.test_active_acquisition import source_fixture, task

POLICY = "component-status/1.0.0"


def profile():
    return dict(load(), numeric_input_disposition=POLICY)


def test_real_adapter_missing_cells_are_gaps_without_changing_requirements_or_history(tmp_path):
    prepared = prepare("grape", grape(tmp_path), "1:OD", cutoff_visit=1)
    medical = source_fixture()
    old_rows, old_plan = project(prepared.manifest, medical, task(), load())
    rows, plan = project(prepared.manifest, medical, task(), profile())
    # This is a status correction, not a new medical definition or a new vote.
    assert rows == old_rows
    assert project(prepared.manifest, medical, task(), load()) == (old_rows, old_plan)
    before = {r["id"]: r for r in old_plan["rows"]}
    current = {r["id"]: r for r in plan["rows"]}
    missing = [
        c for c in prepared.manifest.components if c.kind == "scalar" and c.status == "missing"
    ]
    assert missing
    for component in missing:
        identifier = "check:" + component.id
        assert before[identifier]["state"] == "scheduled"
        assert current[identifier]["state"] == "input_gap"
        assert current[identifier]["reason"] == component.reason
    assert current["check:visit:0:iop"]["state"] == "scheduled"  # source value 0 is present
    assert plan["schema_version"] == "source-bound-acquisition-plan/1.1.0"
    assert plan["numeric_input_disposition"] == POLICY
    base = SimpleNamespace(
        manifest=prepared.manifest, sources=SimpleNamespace(read=lambda: medical), task=task()
    )
    owner = ActiveOwner(base, profile(), plan, {"queries": []}, None)
    owner.check()
    with pytest.raises(ValueError, match="active_source_or_requirement_changed"):
        ActiveOwner(base, load(), plan, {"queries": []}, None).check()
    report = dict(
        schema_version="dynamic-case-report/1.0.0",
        result={"phases": []},
        requirements=[dict(id=r["id"], source_requirement=current[r["id"]]) for r in rows],
        scores=None,
    )
    saved = deepcopy(report)
    result = coverage(report)
    assert report == saved
    assert all(
        next(r for r in result["rows"] if r["id"] == "check:" + c.id)["state"] == "input_missing"
        for c in missing
    )
    assert result["counts"]["scored"] == 0


@pytest.mark.parametrize(
    "value,status,reason",
    [(0, "obtained", None), (-2, "obtained", None), (1, "partial", "partial_source")],
)
def test_zero_negative_and_partial_values_remain_schedulable(value, status, reason):
    component = Component(
        id="measurement",
        kind="scalar",
        concept="iop",
        modality="structured_measurement",
        status=status,
        value=value,
        reason=reason,
        source_locator="synthetic#cell",
    )
    manifest = SimpleNamespace(
        components=(component,),
        adapter_version="multimodal-adapters/1.0.0",
        checksum=digest(component),
    )
    _, plan = project(manifest, source_fixture(), task(), profile())
    row = next(r for r in plan["rows"] if r["id"] == "check:measurement")
    assert row["state"] == "scheduled" and row["reason"] is None


@pytest.mark.parametrize("invalid", [None, False, "component-status/2.0.0"])
def test_unknown_policy_rejected_before_projection(tmp_path, invalid):
    prepared = prepare("grape", grape(tmp_path), "1:OD", cutoff_visit=1)
    with pytest.raises(ValueError, match="active_numeric_input_disposition"):
        project(
            prepared.manifest,
            source_fixture(),
            task(),
            dict(profile(), numeric_input_disposition=invalid),
        )


@pytest.mark.parametrize(
    "old,new",
    [
        ("multimodal_active_requirements_v9.json", "multimodal_active_requirements_v10.json"),
        (
            "multimodal_grape_active_requirements_v3.json",
            "multimodal_grape_active_requirements_v4.json",
        ),
        (
            "multimodal_harvard_visual_priority_v2.json",
            "multimodal_harvard_visual_priority_v3.json",
        ),
    ],
)
def test_explicit_profile_changes_only_input_status_policy(old, new):
    before = load("configs/research/" + old)
    after = load("configs/research/" + new)
    assert after.pop("numeric_input_disposition") == POLICY
    assert after.pop("id") == before.pop("id") + "-component-status-v1"
    assert after == before


def test_withheld_numeric_component_does_not_enter_public_demand():
    component = Component(
        id="hidden",
        kind="scalar",
        concept="iop",
        modality="structured_measurement",
        status="withheld",
        reason="not_for_inference",
        source_locator="synthetic#hidden",
    )
    manifest = SimpleNamespace(
        components=(component,),
        adapter_version="multimodal-adapters/1.0.0",
        checksum=digest(component),
    )
    rows, plan = project(manifest, source_fixture(), task(), profile())
    assert not any(r["component"] == "hidden" for r in rows)
    assert not any(r.get("component") == "hidden" for r in plan["rows"])
