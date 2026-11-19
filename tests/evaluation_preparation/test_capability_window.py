"""Automatic successor bounds, original observations and exact local material."""

from copy import copy, deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research.active_acquisition import successor_projection
from packages.multimodal_research.active_requirements import load
from packages.multimodal_research.campaign import configuration
from packages.multimodal_research.campaign_renewal import RenewedCampaign
from packages.multimodal_research.capability_window import plan_for, projected_base
from tests.evaluation_preparation.test_capability_transition import native_fixture
from tests.evaluation_preparation.test_scoped_dispatch import campaign


def plan_fixture(tmp_path):
    now = datetime(2026, 11, 7, tzinfo=UTC)
    parent = SimpleNamespace(
        output=tmp_path / "previous",
        campaign_id="previous",
        base=SimpleNamespace(
            manifest=SimpleNamespace(checksum=digest("input")),
            task=SimpleNamespace(checksum=digest("task")),
        ),
        config=configuration(),
        cumulative_request_limit=lambda: 40,
        protocol={"deadline": (now - timedelta(hours=1)).isoformat()},
    )
    tip = dict(result=digest("terminal"), previous_attempts=[dict(requests=33)])
    inventory = dict(
        review_target_ids=["one", "two"],
        confirmed_review={"assessment_ids": ["one"]},
        acquisition_targets=[],
        input_scope={"targets": ["scope:one", "scope:two"]},
        confirmed_observations_checksum=digest("facts"),
        procedure_reuse={"plans": []},
    )
    return (
        parent,
        tip,
        inventory,
        dict(output=tmp_path / "next", authorization="explicit goal", now=now, implementation={}),
    )


def test_plan_retains_budget_history_task_and_reserved_review(tmp_path):
    parent, tip, inventory, kwargs = plan_fixture(tmp_path)
    before = deepcopy((tip, inventory, parent.protocol))
    plan = plan_for(parent, tip, inventory, **kwargs)
    assert plan["max_new_requests"] == 7 and plan["max_model_dispatches"] == 6
    assert plan["previous_attempts"] == tip["previous_attempts"]
    assert plan["prior_deadline"] == parent.protocol["deadline"]
    assert plan["input"] == parent.base.manifest.checksum
    assert plan["task"] == parent.base.task.checksum
    assert plan["reuse_assessment_ids"] == ["one"]
    assert datetime.fromisoformat(plan["deadline"]) == kwargs["now"] + timedelta(seconds=5400)
    assert (tip, inventory, parent.protocol) == before


@pytest.mark.parametrize("fault", ["budget", "negative", "authorization", "seconds", "unaware"])
def test_invalid_or_insufficient_window_rejected(tmp_path, fault):
    parent, tip, inventory, kwargs = plan_fixture(tmp_path)
    if fault == "budget":
        tip["previous_attempts"][0]["requests"] = 39
    elif fault == "negative":
        tip["previous_attempts"][0]["requests"] = -1
    elif fault == "authorization":
        kwargs["authorization"] = " "
    elif fault == "seconds":
        kwargs["seconds"] = 5401
    else:
        kwargs["now"] = kwargs["now"].replace(tzinfo=None)
    with pytest.raises(ValueError, match="capability_window_bounds"):
        plan_for(parent, tip, inventory, **kwargs)


def test_successor_base_does_not_retarget_or_mutate_parent(tmp_path):
    base, native, _ = native_fixture(tmp_path)
    base.contexts = {"old": object()}
    base.output = tmp_path / "old"
    profile = load("configs/research/multimodal_catalog_acquisition_v6.json")
    transition = successor_projection(base, profile)
    parent = SimpleNamespace(base=base)
    new = projected_base(parent, transition, profile, tmp_path / "next")
    assert new.output != base.output
    assert new.manifest == base.manifest and new.task == base.task
    assert new.observations == base.observations
    assert new.observations is not base.observations
    assert new.contexts is not base.contexts
    assert new.active_owner.native is native
    assert new.definition_current is base.definition_current
    assert [r["id"] for r in new.ledger] == [r["id"] for r in base.ledger]


def test_inventory_original_native_reference_survives_json_claim(tmp_path, monkeypatch):
    import json

    from packages.grounding.models import canonical
    from packages.multimodal_research import campaign_renewal, input_scope, reuse
    from packages.multimodal_research import capability_window as window
    from packages.multimodal_research.campaign import MultimodalCampaign

    base, _, _ = native_fixture(tmp_path)
    base.contexts = {}
    parent, tip, inv, kwargs = plan_fixture(tmp_path)
    parent.base = base
    parent.target, parent.contexts = object(), object()
    parent.accepted_observations = lambda: base.observations
    monkeypatch.setattr(campaign_renewal, "stopped_tip", lambda *a, **kw: tip)
    monkeypatch.setattr(campaign_renewal, "attempted_acquisition_targets", lambda *a: set())
    monkeypatch.setattr(window, "review_origin", lambda *a: (object(), inv["confirmed_review"]))
    monkeypatch.setattr(MultimodalCampaign, "available_targets", lambda b: list(b.observations))
    monkeypatch.setattr(input_scope, "project_plan", lambda *a: {"entries": []})
    monkeypatch.setattr(input_scope, "configuration", lambda *a: parent.config)
    monkeypatch.setattr(reuse, "repository", lambda *a: None)
    monkeypatch.setattr(
        reuse,
        "ProcedureReuse",
        lambda *a: SimpleNamespace(
            plan={"plans": []},
            prepare=lambda: None,
        ),
    )
    _, found, _ = window.inventory(parent, kwargs["output"])
    native = found["transition"]["retained_native_observations"][0]
    assert native["quality_source"]["code"] == "ASSET:MEASUREMENT"
    assert found == json.loads(canonical(found))
    # A changed reference remains a different plan; serialization is not a
    # relaxation of source identity or the current-owner proof above.
    altered = deepcopy(found)
    altered["transition"]["retained_native_observations"][0]["quality_source"]["code"] = "changed"
    assert digest(altered) != digest(found)


@pytest.mark.parametrize("eligible", [[], ["bound_tool"]])
def test_current_native_fact_survives_survey_without_new_install(monkeypatch, eligible):
    from packages.multimodal_research import active_acquisition, scope_selection

    c = campaign()
    native = dict(
        schema_version="native-fundus-acceptance/1.0.0",
        status="obtained",
        plan={"target": "bound_tool"},
    )
    checks = []
    c.base.active_owner = SimpleNamespace(
        profile={}, native=native, check=lambda: checks.append("original_owner")
    )
    plan = dict(rows=[dict(id=k, native_candidate=True) for k in eligible])
    monkeypatch.setattr(scope_selection, "enabled", lambda base: True)
    monkeypatch.setattr(
        scope_selection, "SelectionOwner", lambda _: SimpleNamespace(save=lambda: plan)
    )
    monkeypatch.setattr(
        active_acquisition,
        "install_native",
        lambda *a, **kw: pytest.fail("duplicate native execution"),
    )
    assert scope_selection.ensure(c) == plan
    assert c.base.active_owner.native is native and checks == ["original_owner"]
    assert c.base.scope_native_targets == frozenset({"bound_tool"})
    assert scope_selection.ensure(c) == plan
    assert c.base.scope_native_targets == frozenset({"bound_tool"})
    assert checks == ["original_owner", "original_owner"]


def assert_original_successor_constructor(tmp_path, monkeypatch, target, *, version=None):
    from packages.multimodal_research import capability_window as window
    from tests.evaluation_preparation.test_input_scope_phase import fixture as scope_fixture

    base, original, _ = scope_fixture(tmp_path)
    old_output = base.output
    parent, tip, inv, kwargs = plan_fixture(tmp_path)
    parent.base, parent.config, parent.target = base, original.config, target
    parent.synthetic = True
    parent.output = old_output
    plan = plan_for(parent, tip, inv, **kwargs)
    if version is not None:
        plan["schema_version"] = version
    following = copy(base)
    following.output = kwargs["output"]
    following.output.mkdir()
    following.confirmed_review_reader = object()
    monkeypatch.setattr(window, "install_base", lambda *args: following)
    installed = []
    monkeypatch.setattr(window, "install_methods", lambda c: installed.append(c))
    restored = RenewedCampaign(parent, following.output, plan)
    assert restored.base is following and base.output == old_output
    assert restored.scope_targets and restored.targets == ()
    assert restored.config == parent.config
    assert restored.reused_review_count == 1
    assert installed == [restored]
    assert (following.output / "input-scope-plan.json").is_file()
    assert restored.scoped_acquisition_inventory() == ()
