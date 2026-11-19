"""Private proof wrapper retains original Runtime batching and fresh revalidation."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from packages.evidence_details.accepted_read_scope import _Documents
from packages.research_runtime.agent_records import RuntimeAgentJournal, RuntimeDocuments


def fixture(count=70):
    records = {f"journal:{i}": (dict(phase="synthetic", index=i),) for i in range(count)}
    calls = dict(single=[], batch=[])
    owner = object.__new__(RuntimeDocuments)
    owner.repo = SimpleNamespace(read_observation_bundles=lambda keys: None)

    def single(key):
        calls["single"].append(key)
        return deepcopy(records.get(key))

    def batch(keys):
        calls["batch"].append(tuple(keys))
        return tuple(deepcopy(records.get(k)) for k in keys)

    owner.read, owner.read_many = single, batch
    return owner, records, calls


def test_wrapped_runtime_journal_keeps_ordered_batches_and_tracks_return_proof():
    owner, records, calls = fixture()
    wrapped = _Documents(owner)
    expected = tuple(records[f"journal:{i}"][0] for i in range(70))
    assert RuntimeAgentJournal(wrapped).read() == expected
    assert len(calls["batch"]) == 2 and not calls["single"]
    assert all(1 <= len(keys) <= 64 for keys in calls["batch"])
    assert wrapped.reads["journal:0"] and "journal:70" in wrapped.reads
    wrapped.verify()
    assert len(calls["batch"]) == 4 and not calls["single"]
    assert RuntimeAgentJournal(wrapped).read() == expected
    assert len(calls["batch"]) == 6  # fresh next operation, no long-lived data cache


@pytest.mark.parametrize("mutation", ["record", "missing", "append"])
def test_final_current_read_detects_changes(mutation):
    owner, records, calls = fixture(2)
    wrapped = _Documents(owner)
    RuntimeAgentJournal(wrapped).read()
    if mutation == "record":
        records["journal:0"][0]["index"] = 99
    elif mutation == "missing":
        del records["journal:0"]
    else:
        records["journal:2"] = (dict(phase="synthetic", index=2),)
    with pytest.raises(ValueError, match="accepted_review_reuse_document_changed"):
        wrapped.verify()


def test_unrelated_memory_owner_keeps_original_single_read_path():
    calls = []
    owner = SimpleNamespace(read=lambda key: calls.append(key))
    wrapped = _Documents(owner)
    assert RuntimeAgentJournal(wrapped).read() == ()
    assert calls == ["journal:0"]
    wrapped.verify()
    assert calls == ["journal:0", "journal:0"]


def test_batch_shape_failure_cannot_silently_drop_reference():
    owner, _, _ = fixture()
    owner.read_many = lambda keys: ()
    with pytest.raises(ValueError, match="accepted_review_reuse_document_batch"):
        RuntimeAgentJournal(_Documents(owner)).read()


def test_current_owner_exception_is_not_empty_journal():
    owner, _, _ = fixture()

    def failure(keys):
        raise OSError("synthetic_current_read_failed")

    owner.read_many = failure
    with pytest.raises(OSError, match="synthetic_current_read_failed"):
        RuntimeAgentJournal(_Documents(owner)).read()
