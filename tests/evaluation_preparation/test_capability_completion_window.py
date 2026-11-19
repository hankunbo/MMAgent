"""Acquisition cutoff preserves the reserved Review window without renewing time."""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation.generic_capability_successor import (
    COMPLETION_VERSION,
    CapabilitySuccessor,
)
from packages.llm_provider.provider import ProviderError
from tests.evaluation_preparation.test_generic_capability_successor import campaign


def setup(tmp_path, monkeypatch, *, status="accepted"):
    from packages.evaluation_preparation import generic_capability_recovery as inventory

    value = campaign(tmp_path)
    value.output.mkdir()
    value.protocol["observations"] = dict(completion_protocol=COMPLETION_VERSION)
    value.selection_owner = SimpleNamespace(current=lambda: None)
    guards = []

    def guard():
        assert value._capability_role == "review"
        if datetime.now(UTC) >= value.work_deadline():
            raise ProviderError("deadline")
        guards.append("original_review_guard")

    value.guard = guard
    monkeypatch.setattr(
        inventory,
        "phase_inventory",
        lambda owner, target: dict(
            target=target, status=status, accepted=(target == "new-one" and status == "accepted")
        ),
    )
    return value, guards


@pytest.mark.parametrize("failure_mode", ["returned", "raised"])
def test_acquisition_expiry_does_not_consume_review_reserve(tmp_path, monkeypatch, failure_mode):
    value, guards = setup(tmp_path, monkeypatch)
    original_deadline = value.protocol["deadline"]
    sent = []

    def phase(target):
        sent.append(target)
        if failure_mode == "raised":
            raise ProviderError("deadline")
        return dict(accepted=False, error="deadline")

    value.execute_phase = phase
    value.before_review()
    assert sent == ["new-one"] and guards == ["original_review_guard"] * 2
    assert value.protocol["deadline"] == original_deadline
    assert value._capability_role == "review"
    saved = json.loads((value.output / "acquisition-cutoff.json").read_text())
    assert saved["deferred_targets"] == ["new-two"]
    assert saved["new_acquisition_dispatch_authorized"] is False
    assert value.phase_completed(dict(accepted=True)) is False


@pytest.mark.parametrize("error", ["cancelled", "budget", "scope", "transport"])
def test_other_errors_do_not_unlock_review(tmp_path, monkeypatch, error):
    value, _ = setup(tmp_path, monkeypatch)

    def failed(target):
        raise ProviderError(error)

    value.execute_phase = failed
    with pytest.raises(ProviderError, match=error):
        value.before_review()
    assert not (value.output / "acquisition-cutoff.json").exists()


def test_review_deadline_still_rejects(tmp_path, monkeypatch):
    value, _ = setup(tmp_path, monkeypatch)
    value.protocol["deadline"] = (datetime.now(UTC) + timedelta(seconds=100)).isoformat()
    with pytest.raises(ProviderError, match="deadline"):
        value.close_acquisition_window("new-one")
    assert not (value.output / "acquisition-cutoff.json").exists()


def test_deadline_before_acquisition_cutoff_is_not_swallowed(tmp_path, monkeypatch):
    value, _ = setup(tmp_path, monkeypatch)
    value.protocol["deadline"] = (datetime.now(UTC) + timedelta(hours=2)).isoformat()
    with pytest.raises(ProviderError, match="deadline"):
        value.close_acquisition_window("new-one")


def test_unknown_blocks_fallthrough_without_retry(tmp_path, monkeypatch):
    value, _ = setup(tmp_path, monkeypatch, status="unknown_requires_explicit_resolution")
    with pytest.raises(ValueError, match="unknown_requires_recovery"):
        value.close_acquisition_window("new-one")
    assert not (value.output / "acquisition-cutoff.json").exists()


def test_old_protocol_keeps_explicit_original_semantics(tmp_path, monkeypatch):
    value, _ = setup(tmp_path, monkeypatch)
    value.protocol["observations"] = {}

    def failed(target):
        raise ProviderError("deadline")

    value.execute_phase = failed
    with pytest.raises(ProviderError, match="deadline"):
        value.before_review()
    assert not (value.output / "acquisition-cutoff.json").exists()


def test_read_only_history_cannot_even_register_a_new_phase(tmp_path):
    value = campaign(tmp_path)
    value._capability_history_only = True
    with pytest.raises(ValueError, match="history_cannot_execute"):
        CapabilitySuccessor.execute_phase(value, "new-one")
    with pytest.raises(ValueError, match="history_cannot_execute"):
        CapabilitySuccessor.guard(value)


def test_report_cutoff_is_verified_against_original_phase_inventory(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from packages.evaluation_preparation import generic_capability_execution as execution
    from packages.evaluation_preparation import generic_capability_recovery as recovery
    from packages.grounding.models import digest

    parent = SimpleNamespace(material=SimpleNamespace(scope=SimpleNamespace(run_id="RUN:REVIEW")))
    projected = dict(
        phase_input_checksum="parent",
        selection_checksum="selected",
        correspondence_checksum="policy",
    )
    rows = [
        dict(target="one", status="accepted", accepted=True),
        dict(target="two", status="not_registered"),
    ]
    protocol = {"original": "bound"}
    from packages.evaluation_preparation.generic_capability_successor import COMPLETION_VERSION

    cutoff = dict(
        schema_version=COMPLETION_VERSION,
        protocol_checksum=digest(protocol),
        phases=rows,
        whole_increment_complete=False,
    )
    (tmp_path / "acquisition-cutoff.json").write_text(json.dumps(cutoff))
    owner = SimpleNamespace(
        output=tmp_path,
        protocol=protocol,
        visual_targets=("one", "two"),
        selection={"selection_checksum": "selected", "correspondence_checksum": "policy"},
        documents=lambda run: SimpleNamespace(read=lambda key: (projected,)),
    )
    original_digest = execution.digest
    monkeypatch.setattr(
        execution, "digest", lambda x: "parent" if x is parent else original_digest(x)
    )
    monkeypatch.setattr(recovery, "phase_inventory", lambda _, t: rows[0 if t == "one" else 1])
    assert execution.report_metadata(owner, parent)["acquisition_completion"] == cutoff
    rows[1]["status"] = "changed"
    with pytest.raises(ValueError, match="report_cutoff_binding"):
        execution.report_metadata(owner, parent)


def test_local_export_keeps_deferred_acquisitions_incomplete(tmp_path, monkeypatch):
    import copy
    import json
    from contextlib import contextmanager
    from types import SimpleNamespace

    from packages.evaluation_preparation import generic_definition_execution as execution
    from packages.evaluation_preparation import generic_run
    from packages.evaluation_preparation.generic_capability_successor import (
        COMPLETION_VERSION,
        CapabilitySuccessor,
    )
    from packages.grounding.models import digest

    class Documents:
        run_id = "RUN:REVIEW"
        saved = {}

        def read(self, key):
            return self.saved.get(key)

        def write(self, key, value):
            self.saved[key] = copy.deepcopy(value)

        @contextmanager
        def session(self):
            yield

    docs = Documents()
    protocol = {"previous": {"reservations": 12}}
    value = object.__new__(CapabilitySuccessor)
    value.output = tmp_path / "evidence"
    value.output.mkdir()
    value.protocol = protocol
    value.sample = "synthetic"
    value.base = SimpleNamespace(binding={"case_code": "CASE:TEST"})
    value.max_new_requests = 5
    value.export_suffix = "_test"
    value.usage = lambda: 1
    value.accepted_observations = lambda: {}
    (value.output / "review").mkdir()
    (value.output / "acquisition-cutoff.json").write_text(
        json.dumps(dict(schema_version=COMPLETION_VERSION, protocol_checksum=digest(protocol)))
    )
    phase = dict(run_id=docs.run_id, target=None, accepted=False, error="local_report_failure")
    detail = {"original_score": "unchanged"}
    proof = {"acceptance_key": "accepted", "report_checksum": digest(detail)}
    value.export_result = lambda result: None
    monkeypatch.setattr(execution, "directory", lambda _: tmp_path)
    monkeypatch.setattr(execution, "load_selection", lambda _: value)
    monkeypatch.setattr(
        execution, "committed_state", lambda _: (docs, None, phase, (proof, detail))
    )
    monkeypatch.setattr(generic_run, "persist_export", lambda key, c, result: dict(result=result))
    result = execution.export_saved("new")["result"]
    assert result["phases"][0]["accepted"] is True
    assert result["complete"] is False
    assert result["local_closure"]["api_requests"] == 0
    assert docs.saved["generic-capability-local-case-result"] == (result,)
