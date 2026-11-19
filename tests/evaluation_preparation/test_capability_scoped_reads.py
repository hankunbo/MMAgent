"""Candidate-only read boundary probes; no deployment, credentials or real calls."""

from types import SimpleNamespace

import pytest

from packages.evaluation_preparation import generic_capability_projection as projection
from packages.evaluation_preparation import generic_capability_successor as successor
from packages.evidence_details import accepted_read_scope as scoped


class Source:
    def __init__(self):
        self.value, self.reads = "current", 0

    def _history_binding(self):
        return "same-owner"

    def _read_current(self):
        self.reads += 1
        return self.value

    def read(self):
        return scoped.historical_source(self)


@pytest.mark.parametrize("fault", [None, "withdraw", "technical"])
def test_phase_proof_retains_fresh_return_and_exception_cleanup(fault):
    source = Source()
    phase = object.__new__(projection.GenericCapabilityPhase)
    scope = SimpleNamespace(
        case_code="case", run_id="run", input_checksum="input", visible_checksum="visible"
    )
    state = SimpleNamespace(
        policy="policy", material=SimpleNamespace(scope=scope, sources="sources")
    )
    phase.original_seed = phase.seed = "seed"
    phase.initial = state
    phase.capability_projection = {"version": 1}
    phase.base = SimpleNamespace(verify=lambda _: source.read())

    def project(_):
        assert source.read() == "current"
        return "seed", {"version": 1}

    def current():
        assert source.read() == "current"
        if fault == "withdraw":
            source.value = "withdrawn"
        elif fault == "technical":
            raise OSError("injected_read_error")
        return state

    phase.selection_owner = SimpleNamespace(project=project)
    phase.project_current = current
    if fault:
        error, message = (
            (ValueError, "historical_source_changed")
            if fault == "withdraw"
            else (OSError, "injected_read_error")
        )
        with pytest.raises(error, match=message):
            phase.verify(state)
    else:
        phase.verify(state)
        assert source.reads == 2  # first current read plus fresh return check
        phase.verify(state)
        assert source.reads == 4  # no reuse across operations
    assert scoped._facts.get() is None
    assert source.read() == source.value


@pytest.mark.parametrize("fault", [None, "withdraw", "duplicate"])
def test_observation_rebuild_has_one_scope_and_preserves_owner_checks(monkeypatch, fault):
    source = Source()
    campaign = object.__new__(successor.CapabilitySuccessor)

    def previous():
        assert source.read() == "current"
        return {"old": "saved-original-observation"}

    def fresh(_):
        assert source.read() == "current"
        assert source.read() == "current"
        if fault == "withdraw":
            source.value = "withdrawn"
        return {"old" if fault == "duplicate" else "new": "confirmed-observation"}

    campaign.acquisition = SimpleNamespace(accepted_observations=previous)
    campaign.checked = SimpleNamespace()
    monkeypatch.setattr(successor.ComprehensiveCampaign, "accepted_observations", fresh)
    if fault:
        with pytest.raises(
            ValueError, match="historical_source_changed|generic_capability_duplicate_observation"
        ):
            campaign.accepted_observations()
    else:
        assert campaign.accepted_observations() == {
            "old": "saved-original-observation",
            "new": "confirmed-observation",
        }
        assert source.reads == 2
        campaign.accepted_observations()
        assert source.reads == 4
    assert scoped._facts.get() is None
