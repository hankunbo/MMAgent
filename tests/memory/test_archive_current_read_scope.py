"""Candidate only: bounded synchronous proofs, never cross-dispatch caching."""

from types import SimpleNamespace

import pytest

from packages.memory import archive as candidate
from packages.object_store import checksum


def fixture():
    value = {"raw": b"original", "case": "CASE:ONE", "research": True}
    calls = []

    def artifact(case, code):
        calls.append(("artifact", case, code))
        return SimpleNamespace(
            scope=SimpleNamespace(case_code="CASE:ONE"),
            research_only=value["research"],
            checksum=checksum(value["raw"]),
        )

    def read_bytes(asset):
        calls.append(("bytes",))
        return value["raw"]

    def read_case(run):
        calls.append(("case", run))
        return value["case"]

    owner = candidate.ArchiveSources(
        runtime=SimpleNamespace(read_case_artifact=artifact),
        read_bytes=read_bytes,
        read_run_case=read_case,
        read_view=lambda *a: None,
    )
    return owner, value, calls


def test_repeated_reads_have_fresh_first_and_return_proofs():
    owner, _, calls = fixture()
    with owner.read_scope():
        for _ in range(10):
            assert owner.artifact("CASE:ONE", "A")[1] == b"original"
            assert owner.case_for_run("RUN:ONE") == "CASE:ONE"
    assert calls.count(("bytes",)) == 2
    assert calls.count(("case", "RUN:ONE")) == 2
    owner.artifact("CASE:ONE", "A")
    assert calls.count(("bytes",)) == 3


@pytest.mark.parametrize("fault", ["bytes", "research", "case"])
def test_current_change_during_scope_rejects_return(fault):
    owner, value, _ = fixture()
    with pytest.raises(ValueError, match="memory_source|memory_scope"):
        with owner.read_scope():
            owner.artifact("CASE:ONE", "A")
            owner.case_for_run("RUN:ONE")
            value[{"bytes": "raw", "research": "research", "case": "case"}[fault]] = {
                "bytes": b"changed",
                "research": False,
                "case": "CASE:OTHER",
            }[fault]


def test_exception_restores_scope_and_preserves_original_error():
    owner, value, calls = fixture()
    with pytest.raises(OSError, match="original technical error"):
        with owner.read_scope():
            owner.artifact("CASE:ONE", "A")
            raise OSError("original technical error")
    value["raw"] = b"after"
    assert owner.artifact("CASE:ONE", "A")[1] == b"after"
    assert calls.count(("bytes",)) == 2


def test_nested_scope_same_owner_and_different_owners_do_not_leak():
    one, _, a = fixture()
    two, _, b = fixture()
    with one.read_scope():
        one.artifact("CASE:ONE", "A")
        with one.read_scope():
            one.artifact("CASE:ONE", "A")
        with two.read_scope():
            two.artifact("CASE:ONE", "A")
            two.artifact("CASE:ONE", "A")
        one.artifact("CASE:ONE", "A")
    assert a.count(("bytes",)) == b.count(("bytes",)) == 2


def test_capacity_overflow_falls_back_to_original_reads(monkeypatch):
    owner, _, calls = fixture()
    monkeypatch.setattr(candidate, "MAX_READ_BYTES", 1)
    monkeypatch.setattr(candidate, "MAX_READ_KEYS", 1)
    with owner.read_scope():
        for _ in range(3):
            owner.artifact("CASE:ONE", "A")
            owner.case_for_run("RUN:ONE")
            owner.case_for_run("RUN:TWO")
    assert calls.count(("bytes",)) == 3
    assert calls.count(("case", "RUN:ONE")) == 2
    assert calls.count(("case", "RUN:TWO")) == 3


def test_returned_metadata_cannot_mutate_saved_proof():
    owner, _, _ = fixture()
    with owner.read_scope():
        asset, _ = owner.artifact("CASE:ONE", "A")
        asset.scope.case_code = "CASE:OTHER"
        current, _ = owner.artifact("CASE:ONE", "A")
        assert current.scope.case_code == "CASE:ONE"


def test_exit_owner_failure_remains_technical():
    owner, _, _ = fixture()
    with pytest.raises(OSError, match="original database unavailable"):
        with owner.read_scope():
            owner.artifact("CASE:ONE", "A")

            def unavailable(*args):
                raise OSError("original database unavailable")

            owner.runtime.read_case_artifact = unavailable
