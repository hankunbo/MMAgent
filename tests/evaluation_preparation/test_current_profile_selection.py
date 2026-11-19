"""Automatic new Scope contract and exact old-claim profile recovery."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import campaign_renewal
from packages.multimodal_research import capability_window as window
from packages.multimodal_research.active_requirements import load
from tests.evaluation_preparation.test_capability_window import plan_fixture


def profile(path):
    return dict(path=path, checksum=digest(load(path)))


def claimed(path, output):
    plan = dict(
        schema_version=window.CURRENT_VERSION,
        capability_transition=dict(profile=profile(path)),
    )
    return (dict(output=str(output.absolute()), plan=plan),)


@pytest.mark.parametrize("path", [window.CURRENT_PROFILE, window.NEW_CURRENT_PROFILE])
def test_frozen_profile_keeps_version_and_checks_current_bytes(path):
    plan = dict(capability_transition=dict(profile=profile(path)))
    assert window.frozen_current_profile(plan) == path
    changed = deepcopy(plan)
    changed["capability_transition"]["profile"]["checksum"] = digest("different")
    with pytest.raises(ValueError, match="profile_binding"):
        window.frozen_current_profile(changed)


@pytest.mark.parametrize("fault", ["missing", "arbitrary_path", "extra", "old_contract"])
def test_profile_cannot_be_selected_by_unqualified_path(fault):
    value = profile(window.NEW_CURRENT_PROFILE)
    if fault == "missing":
        value.pop("checksum")
    elif fault == "arbitrary_path":
        value["path"] = "configs/research/not-approved.json"
    elif fault == "extra":
        value["authorized"] = True
    else:
        value["path"] = "configs/research/multimodal_catalog_acquisition_v7.json"
    with pytest.raises(ValueError, match="profile_binding"):
        window.frozen_current_profile(dict(capability_transition=dict(profile=value)))


@pytest.mark.parametrize("original", [None, window.CURRENT_PROFILE, window.NEW_CURRENT_PROFILE])
def test_new_profile_or_original_claim_recovery_uses_runtime_not_local_file(
    tmp_path, monkeypatch, original
):
    output = tmp_path / "successor"
    saved = None if original is None else claimed(original, output)
    parent = SimpleNamespace(
        output=tmp_path / "old",
        protocol={},
        documents=lambda run: SimpleNamespace(read=lambda key: saved),
    )
    monkeypatch.setattr(campaign_renewal, "stopped_tip", lambda *a, **k: dict(run_id="RUN:OLD"))
    assert window.preparation_profile(parent, output) == (original or window.NEW_CURRENT_PROFILE)
    assert not output.exists()


@pytest.mark.parametrize("fault", ["output", "version", "two_claims", "checksum"])
def test_conflicting_claim_is_not_reinterpreted_as_fresh(tmp_path, monkeypatch, fault):
    output = tmp_path / "successor"
    saved = list(claimed(window.CURRENT_PROFILE, output))
    if fault == "output":
        saved[0]["output"] = str(tmp_path / "other")
    elif fault == "version":
        saved[0]["plan"]["schema_version"] = window.VERSION
    elif fault == "two_claims":
        saved.append(deepcopy(saved[0]))
    else:
        saved[0]["plan"]["capability_transition"]["profile"]["checksum"] = digest("other")
    parent = SimpleNamespace(
        output=tmp_path / "old",
        protocol={},
        documents=lambda run: SimpleNamespace(read=lambda key: tuple(saved)),
    )
    monkeypatch.setattr(campaign_renewal, "stopped_tip", lambda *a, **k: dict(run_id="RUN:OLD"))
    with pytest.raises(ValueError, match="already_claimed|profile_binding"):
        window.preparation_profile(parent, output)


@pytest.mark.parametrize("path", [window.CURRENT_PROFILE, window.NEW_CURRENT_PROFILE])
def test_cold_check_projects_frozen_profile_and_rejects_other_medical_changes(
    tmp_path, monkeypatch, path
):
    parent, tip, found, args = plan_fixture(tmp_path)
    parent.base.visible_value = SimpleNamespace(checksum=digest("visible"))
    found["profile"] = profile(path)
    plan = window.plan_for(parent, tip, found, **args, current=True)
    calls = []
    reader = object()

    def inventory(*args, **kwargs):
        calls.append(kwargs["profile_path"])
        return tip, found, reader

    monkeypatch.setattr(window, "inventory", inventory)
    assert window.check(parent, plan) is reader
    assert calls == [path]
    changed = deepcopy(plan)
    changed["review_target_ids"].append("extra")
    with pytest.raises(ValueError, match="original_binding_changed"):
        window.check(parent, changed)


@pytest.mark.parametrize("path", [window.CURRENT_PROFILE, window.NEW_CURRENT_PROFILE])
def test_unknown_successor_keeps_original_scope_profile(tmp_path, monkeypatch, path):
    from packages.multimodal_research import scope_unknown_successor as owner
    from tests.evaluation_preparation.test_scope_unknown_successor import planned

    parent, previous, tip, found, proof, _, _ = planned(tmp_path, monkeypatch)
    target = proof["original_attempt"]["target"]
    proof["task"] = dict(exact="original-scope")
    found["profile"] = profile(path)
    found["scope_plan"] = dict(entries=[dict(target=target, task=proof["task"])])
    found["transition"] = dict(coverage=dict(selected=2), changes=[])
    found["input_scope"]["dispatch_targets"] = [target]
    for name in ("native_checksum", "bound_observations_checksum"):
        found[name] = digest(name)
    previous.protocol["capability_transition"] = deepcopy(found)
    previous.protocol["previous_attempts"] = deepcopy(tip["previous_attempts"])
    previous.base = parent.base
    calls = []

    def inventory(*args, **kwargs):
        calls.append(kwargs["profile_path"])
        return tip, found, object()

    monkeypatch.setattr(owner, "inspect", lambda *a: proof)
    monkeypatch.setattr(window, "inventory", inventory)
    result = owner.inventory(parent, previous, tmp_path / "next", "request:old", {})
    assert result[-1] == proof and calls == [path]
    assert found["scope_plan"] == previous.protocol["capability_transition"]["scope_plan"]
