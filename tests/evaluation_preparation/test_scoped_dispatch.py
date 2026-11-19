"""Scope assessment may narrow, never expand, a frozen successor dispatch set."""

from types import SimpleNamespace

import pytest

from packages.multimodal_research.campaign import MultimodalCampaign
from packages.multimodal_research.campaign_renewal import RenewedCampaign


def campaign(*, renewed=True, mode="acquisition"):
    value = object.__new__(RenewedCampaign if renewed else MultimodalCampaign)
    value.base = SimpleNamespace(
        manifest=SimpleNamespace(components=[SimpleNamespace(id="fundus", kind="image")]),
        views={"views": [dict(component="fundus", status="obtained")]},
        ledger=[
            dict(id=key, component="fundus", acquisition_method="vision")
            for key in ("confirmed", "new", "outside_window", "bound_tool")
        ],
        observations={"bound_tool": object()},
        active_owner=SimpleNamespace(),
    )
    value.protocol = dict(mode=mode, acquisition_targets=["confirmed", "new", "bound_tool"])
    value.accepted_observations = lambda: {"confirmed": object()}
    return value


def test_successor_scope_selection_keeps_original_dispatch_slice():
    c = campaign()
    assert c.scoped_acquisition_targets() == ("new",)


def test_root_scope_selection_keeps_unconfirmed_candidates():
    c = campaign(renewed=False)
    assert c.scoped_acquisition_targets() == ("new", "outside_window")


@pytest.mark.parametrize("mode", ["review", "definitions"])
def test_other_successor_phases_do_not_start_patient_acquisition(mode):
    assert campaign(mode=mode).scoped_acquisition_targets() == ()


def test_current_observation_owner_failure_is_not_empty_history():
    c = campaign()

    def failed():
        raise ValueError("renewal_observations_changed")

    c.accepted_observations = failed
    with pytest.raises(ValueError, match="renewal_observations_changed"):
        c.scoped_acquisition_targets()


@pytest.mark.parametrize("completed", [True, False])
def test_ensure_routes_both_cold_and_warm_selection_through_campaign_owner(monkeypatch, completed):
    from packages.multimodal_research import active_acquisition, scope_selection

    c = campaign()
    c.base.active_owner = SimpleNamespace(profile={}, plan={}, native=None, check=lambda: None)
    c.base.check = lambda: None
    c.guard_control = lambda: None
    c.targets = ("outside_window", "confirmed", "new")
    selected = dict(rows=[])
    owner = SimpleNamespace(save=lambda: selected, restore=lambda: selected, check=lambda: selected)
    monkeypatch.setattr(scope_selection, "enabled", lambda base: True)
    monkeypatch.setattr(scope_selection, "SelectionOwner", lambda current: owner)
    calls = []

    def install(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["eligible_targets"] == frozenset()
        return None, None

    monkeypatch.setattr(active_acquisition, "install_native", install)
    if completed:
        c.base.scope_native_completed = frozenset()
    assert scope_selection.ensure(c) == selected
    assert c.targets == ("confirmed", "new")
    assert c.scoped_acquisition_targets() == ("new",)
    assert c.scoped_acquisition_targets() == ("new",)
    assert len(calls) == (0 if completed else 1)
