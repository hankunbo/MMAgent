"""A read-only Phase operation reuses facts, but never cached current authority."""

from types import SimpleNamespace

import pytest

from packages.dataset_adapters import current_read_scope as inputs
from packages.evidence_details import accepted_read_scope as facts
from packages.multimodal_research.campaign import Phase
from tests.agent_provider.test_accepted_read_scope import component_chain


@pytest.mark.parametrize("entry", ["current_projection", "verify"])
def test_phase_reads_component_once_then_fresh_at_return(tmp_path, monkeypatch, entry):
    _, reads, campaign, reader = component_chain(tmp_path, monkeypatch)
    state = campaign.owner.current()
    phase = Phase(SimpleNamespace(root_direction_review={"schema_version": "test"}))

    def project(*_):
        result = reader.current(state)
        assert reader.current(state) == result
        return result

    monkeypatch.setattr(phase, "_current_projection", project)
    monkeypatch.setattr(phase, "_verify", project)
    invoke = (
        phase.current_projection if entry == "current_projection" else lambda: phase.verify(state)
    )
    result = invoke()
    assert len(reads) == 2
    assert invoke() == result
    assert len(reads) == 4  # not a cross-operation cache
    assert facts._facts.get() is None and inputs._scope.get() is None
    assert not campaign.budget.records()


@pytest.mark.parametrize("fault", ["read", "exception"])
def test_phase_mutation_or_exception_does_not_leave_cached_authority(tmp_path, monkeypatch, fault):
    component, _, campaign, reader = component_chain(tmp_path, monkeypatch)
    state = campaign.owner.current()
    phase = Phase(SimpleNamespace(root_direction_review={"schema_version": "test"}))

    def project():
        result = reader.current(state)
        if fault == "read":
            component.read = lambda: None  # original binding must reject before return
        else:
            raise OSError("phase_projection_injected_failure")
        return result

    monkeypatch.setattr(phase, "_current_projection", project)
    error, match = (
        (ValueError, "accepted_review_fact_owner_changed")
        if fault == "read"
        else (OSError, "phase_projection_injected_failure")
    )
    with pytest.raises(error, match=match):
        phase.current_projection()
    assert facts._facts.get() is None and inputs._scope.get() is None
    assert not campaign.budget.records()


def test_phase_final_fact_checks_run_after_visible_input_scope(monkeypatch):
    phase = Phase(SimpleNamespace(root_direction_review={"schema_version": "test"}))
    checkpoints = []

    def project():
        assert inputs._scope.get() is not None
        assert facts._facts.get() is not None
        facts._facts.get().return_checks.append(lambda: checkpoints.append("durable"))
        return "read-only"

    def verify(_):
        assert inputs._scope.get() is None and facts._facts.get() is None
        checkpoints.append("fresh_facts")

    monkeypatch.setattr(phase, "_current_projection", project)
    monkeypatch.setattr(facts._Facts, "verify", verify)
    assert phase.current_projection() == "read-only"
    assert checkpoints == ["fresh_facts", "durable"]


def test_legacy_phase_does_not_install_new_scope(monkeypatch):
    phase = Phase(SimpleNamespace())

    def project():
        assert facts._facts.get() is None and inputs._scope.get() is None
        return "old-path"

    monkeypatch.setattr(phase, "_current_projection", project)
    assert phase.current_projection() == "old-path"
