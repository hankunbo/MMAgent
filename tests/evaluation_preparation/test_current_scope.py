"""Dependency-scoped public definitions; coherent local snapshots, no live calls."""

from types import SimpleNamespace

import pytest

from apps.vectorengine_live_assembly import LiveSnapshots
from packages.multimodal_research.campaign import InputEvidence
from tests.evaluation_preparation.test_definition_trial import scoped


def material_owner():
    calls = []
    owner = InputEvidence.__new__(InputEvidence)
    owner.read_dependency_input = lambda: calls.append("input_source")
    owner.active_owner = SimpleNamespace(check=lambda: calls.append("requirements"))
    owner.definition_current = lambda: calls.append("definitions")
    owner.component_observations = SimpleNamespace(
        check=lambda: calls.append("component"),
        check_projection=lambda: calls.append("component_binding"),
    )
    return owner, calls


def test_definition_materials_exclude_unconsumed_component_but_review_keeps_it():
    owner, calls = material_owner()
    owner.check_definition_materials()
    assert calls == ["input_source", "requirements", "definitions", "component_binding"]
    calls.clear()
    owner.check()
    assert calls == [
        "input_source",
        "requirements",
        "definitions",
        "component_binding",
        "component",
    ]


@pytest.mark.parametrize("dependency", ["input", "requirements", "definitions"])
def test_source_dependency_changes_are_not_cached(dependency):
    owner, _ = material_owner()
    owner.check_definition_materials()

    def changed():
        raise ValueError("changed_" + dependency)

    if dependency == "input":
        owner.read_dependency_input = changed
    elif dependency == "requirements":
        owner.active_owner.check = changed
    else:
        owner.definition_current = changed
    with pytest.raises(ValueError, match="changed_" + dependency):
        owner.check_definition_materials()


@pytest.mark.parametrize("reason", ["ok", "cancelled", "deadline", "budget"])
def test_source_guard_preserves_trial_control(tmp_path, reason):
    from packages.llm_provider.provider import ProviderError

    c = scoped(
        tmp_path, seconds=30 if reason == "deadline" else 900, used=12 if reason == "budget" else 0
    )
    c.base, calls = material_owner()
    if reason == "cancelled":
        (tmp_path / "cancel").touch()
    if reason == "ok":
        c.guard_definition_materials()
        assert calls == ["input_source", "requirements", "definitions", "component_binding"]
    else:
        with pytest.raises(ProviderError, match=reason):
            c.guard_definition_materials()
    assert "component" not in calls


def test_snapshot_step_reads_one_current_and_checks_each_next_call():
    from apps.vectorengine_case_materials import ref
    from packages.grounding.models import digest

    scope = SimpleNamespace(
        run_id="run",
        input_checksum="input",
        visible_checksum=digest("visible"),
        graph=ref("graph", "GRAPH:A", digest("graph")),
    )
    material = SimpleNamespace(scope=scope)
    calls = []

    def current():
        calls.append(1)
        return SimpleNamespace(material=material)

    snapshots = LiveSnapshots(SimpleNamespace(current=current))
    binding = SimpleNamespace(
        run_id="run",
        input_checksum="input",
        visible_checksum=scope.visible_checksum,
        case_code="CASE:A",
    )
    assert snapshots.step(binding, "load").refs[0].code == "CASE:A"
    assert len(calls) == 1
    scope.input_checksum = "changed"
    with pytest.raises(ValueError, match="live_snapshot_binding"):
        snapshots.step(binding, "load")
    assert len(calls) == 2


def test_reserved_source_wire_keeps_original_dispatch_cap(tmp_path):
    c = scoped(tmp_path, seconds=900, used=12)
    c.base, calls = material_owner()
    c.guard_definition_materials(dispatch=True)
    assert "component" not in calls
    from packages.llm_provider.provider import ProviderError

    c.usage = lambda: 13
    with pytest.raises(ProviderError, match="budget"):
        c.guard_definition_materials(dispatch=True)


def test_frozen_component_summary_is_historical_not_a_new_current_score():
    from packages.grounding.models import digest
    from packages.multimodal_research.stage_failure import frozen_component_summary, render

    original = dict(origin={"artifact": "original-ref"}, geometry={"value": 0.5})
    campaign = SimpleNamespace(protocol={"component_observations": original})
    summary = frozen_component_summary(campaign)
    assert summary["historical_confirmation"] == original
    assert summary["protocol_checksum"] == digest(campaign.protocol)
    assert summary["current_qualification"] == "not_rechecked_technical_closure_only"
    assert summary["medical_direction"] is None and not summary["new_score"]
    page = render(dict(scores=None, confirmed_component_snapshot=summary))
    assert "original-ref" in page and "0.5" in page and "本轮未评分" in page
    assert frozen_component_summary(SimpleNamespace(protocol={})) is None


@pytest.mark.parametrize("change", ["none", "method", "quality", "origin"])
def test_definition_projection_binding_without_current_tool_read(tmp_path, change):
    from tests.evaluation_preparation.test_component_observations import installed_fixture

    item = installed_fixture(tmp_path)
    item.install()

    def read():
        raise ValueError("original_current_tool_changed")

    item.read = read
    if change == "method":
        obs = item.base.observations[item.target]
        item.base.observations[item.target] = obs.model_copy(update={"method": "forged"})
    elif change == "quality":
        item.base.ledger[0]["quality"] = 0.9
    elif change == "origin":
        item.origin["extra"] = "forged"
    if change == "none":
        item.check_projection()
        with pytest.raises(ValueError, match="original_current_tool_changed"):
            item.check()  # Local validity is never current consumption permission.
    else:
        with pytest.raises(ValueError, match="component_observation_.*changed"):
            item.check_projection()
