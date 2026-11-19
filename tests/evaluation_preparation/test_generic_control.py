"""Cancellation blocks new dispatch; original owners still finish in-flight work."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from packages.evaluation_preparation import generic_control as control
from packages.llm_provider.provider import ProviderError


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "directory", lambda key: tmp_path)
    spec = dict(key="unit", sample="arbitrary.npz")
    (tmp_path / "input.json").write_text(json.dumps(spec))
    return tmp_path, spec


def test_start_same_key_no_deadline_renewal(workspace):
    _, spec = workspace
    original = control.start(spec)
    assert control.start(spec) == original
    control.guard("unit")
    with pytest.raises(ValueError, match="input_conflict"):
        control.start(dict(spec, sample="other.npz"))


def test_cancel_idempotent_before_start_and_after_facts(workspace):
    root, spec = workspace
    control.cancel("unit")
    previous = (root / "cancel-request.json").read_bytes()
    control.cancel("unit")
    assert previous == (root / "cancel-request.json").read_bytes()
    control.start(spec)
    with pytest.raises(ProviderError, match="cancelled"):
        control.guard("unit")
    assert control.status("unit")["started"]["key"] == "unit"


def test_expiry_and_old_start_never_renewed(workspace):
    root, spec = workspace
    value = control.start(spec)
    value["deadline"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    (root / "started.json").write_text(json.dumps(value))
    assert control.start(spec) == value
    with pytest.raises(ProviderError, match="deadline"):
        control.guard("unit")
    del value["deadline"]
    (root / "started.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="legacy_start"):
        control.guard("unit")


def test_original_executor_event_observes_durable_cancellation(workspace):
    _, spec = workspace
    control.start(spec)
    event = control.Cancellation("unit")
    assert not event.is_set()
    control.cancel("unit")
    assert event.is_set()
