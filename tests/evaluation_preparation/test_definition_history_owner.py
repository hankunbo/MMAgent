"""History has original-owner proof; it is not permission to dispatch again."""

import errno

import pytest

from packages.multimodal_research import discovery_continuation as dc
from tests.evaluation_preparation.test_routed_source_backlog import routed_parent


def test_historical_read_uses_explicit_owner_and_keeps_execution_lock(tmp_path, monkeypatch):
    c, ref, spec, plan, failure, records, writes = routed_parent(tmp_path)
    dc.ConfirmedDefinitionContinuation(c, ref)
    before = len(writes)
    checked = []

    def no_lock(*args):
        raise OSError(errno.ENOLCK, "synthetic unavailable filesystem locks")

    def owner(base, path, entry, protocol, saved):
        assert base is c.base and str(path) == ref["parent"]
        assert (entry, protocol, saved) == (spec, plan, failure)
        checked.append(saved["run_id"])

    monkeypatch.setattr(dc.fcntl, "flock", no_lock)
    monkeypatch.setattr(dc, "historical_definition_owner", owner, raising=False)
    value = dc.ConfirmedDefinitionContinuation(c, ref, read_only=True, history_base=c.base)
    assert value.check_attempts() and checked == [failure["run_id"]]
    assert len(writes) == before
    with pytest.raises(OSError) as error:
        dc.ConfirmedDefinitionContinuation(c, ref)
    assert error.value.errno == errno.ENOLCK
    with pytest.raises(ValueError, match="history_read_only"):
        dc.ConfirmedDefinitionContinuation(c, ref, history_base=c.base)


def test_historical_owner_rejection_cannot_fallback_to_local_files(tmp_path, monkeypatch):
    c, ref, _, _, _, _, _ = routed_parent(tmp_path)
    dc.ConfirmedDefinitionContinuation(c, ref)

    def refuse(*args):
        raise ValueError("definition_history_owner_binding")

    monkeypatch.setattr(dc, "historical_definition_owner", refuse, raising=False)
    with pytest.raises(ValueError, match="history_owner_binding"):
        dc.ConfirmedDefinitionContinuation(c, ref, read_only=True, history_base=c.base)
