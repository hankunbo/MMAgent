"""Original Runtime bytes, detached historical values and fresh exit proof."""

from copy import copy
from types import SimpleNamespace

import pytest

from packages.evidence_details import accepted_read_scope as scoped
from tests.research_runtime.test_exact_consumption_receipt import setup


def fixture():
    f = setup()
    f.root.model_dump = lambda **kw: dict(
        code=f.root.asset_code,
        checksum=f.root.checksum,
        case=f.root.scope.case_code,
        uri=f.root.uri,
    )
    f.result = dict(run_id=f.docs.run_id, accepted=True, acceptance_key=f.key)
    f.add("dynamic-phase-input", dict(original_input="bound"))
    f.batches = []
    f.singles = []
    original = f.docs.repo.read_observation_bundle

    def batch(keys):
        f.batches.append(keys)
        return tuple(original(k) for k in keys)

    def single(key):
        f.singles.append(key)
        return original(key)

    f.docs.repo.read_observation_bundles = batch
    f.docs.repo.read_observation_bundle = single
    return f


def test_same_run_across_document_instances_reads_once_and_fresh_at_return():
    f = fixture()
    with scoped.historical_fact_read_scope():
        first = scoped.historical_observation_documents(f.docs, f.result)
        assert first[0] == (f.result,)
        for _ in range(20):
            assert scoped.historical_observation_documents(copy(f.docs), f.result) == first
        assert len(f.batches) == 1 and not f.singles
        first[1][0]["original_input"] = "caller mutation"
        assert (
            scoped.historical_observation_documents(f.docs, f.result)[1][0]["original_input"]
            == "bound"
        )
    assert len(f.batches) == 2 and not f.singles
    assert scoped._facts.get() is None


def test_outside_and_next_scope_never_reuse_read_or_qualification():
    f = fixture()
    for _ in range(2):
        assert scoped.historical_observation_documents(f.docs, f.result)[0] == (f.result,)
    assert len(f.singles) == 6 and not f.batches
    for _ in range(2):
        with scoped.historical_fact_read_scope():
            scoped.historical_observation_documents(f.docs, f.result)
    assert len(f.batches) == 4


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("bytes", "agent_object_checksum"),
        ("missing", "historical_observation_documents_changed"),
        ("replacement", "historical_observation_documents_changed"),
        ("root", "agent_runtime_scope"),
        ("binding", "historical_observation_owner_changed"),
    ],
)
def test_current_change_before_return_rejects(mutation, error):
    f = fixture()
    with pytest.raises(ValueError, match=error):
        with scoped.historical_fact_read_scope():
            scoped.historical_observation_documents(f.docs, f.result)
            if mutation == "bytes":
                f.raw[f.receipt.uri] += b" "
            elif mutation == "missing":
                del f.bundles[f.docs.key("accepted:" + f.key)]
            elif mutation == "replacement":
                f.add("dynamic-phase-input", dict(original_input="changed"))
            elif mutation == "root":
                del f.artifacts[f.root.asset_code]
            else:
                f.docs.store = SimpleNamespace(read_bounded=f.docs.store.read_bounded)
    assert scoped._facts.get() is None


def test_wrong_run_is_never_served_another_run_cached_values():
    f = fixture()
    with scoped.historical_fact_read_scope():
        scoped.historical_observation_documents(f.docs, f.result)
        other = copy(f.docs)
        other.run_id = "RUN:OTHER"
        # No bundles exist for this run; no old observed values are substituted.
        result = dict(f.result, run_id=other.run_id)
        assert scoped.historical_observation_documents(other, result) == (None, None, None)
        assert len(f.batches) == 2


def test_unaccepted_phase_does_not_read_accepted_or_input_documents():
    f = fixture()
    result = dict(f.result, accepted=False, acceptance_key=None)
    f.add("dynamic-phase-result", result)
    with scoped.historical_fact_read_scope():
        assert scoped.historical_observation_documents(f.docs, result) == ((result,), None, None)
        assert all(len(keys) == 1 for keys in f.batches)
    assert len(f.batches) == 2


def test_current_owner_technical_failure_is_not_empty_observation():
    f = fixture()
    original = f.docs.repo.read_observation_bundles
    fail = []

    def read(keys):
        if fail:
            raise OSError("synthetic_read_failed")
        return original(keys)

    f.docs.repo.read_observation_bundles = read
    with pytest.raises(OSError, match="synthetic_read_failed"):
        with scoped.historical_fact_read_scope():
            scoped.historical_observation_documents(f.docs, f.result)
            fail.append(True)
    assert scoped._facts.get() is None


def test_capacity_limit_falls_back_to_fresh_reads_without_rejecting_valid_history():
    f = fixture()
    with scoped.historical_fact_read_scope():
        for _ in range(66):
            owner = copy(f.docs)
            owner.store = SimpleNamespace(read_bounded=f.docs.store.read_bounded)
            assert scoped.historical_observation_documents(owner, f.result)[0] == (f.result,)
        assert len(scoped._facts.get().observations) == 64
        assert len(f.batches) == 66
    assert len(f.batches) == 130
