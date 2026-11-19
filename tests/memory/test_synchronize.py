"""10B-R02: synchronize saved sources, never re-admit a historical envelope."""

from types import SimpleNamespace

import psycopg
import pytest

from apps.memory_sources import ResearchSources
from db.memory.repository import MemoryUnavailable
from packages.grounding.models import canonical
from packages.memory.models import Dependency
from packages.memory_writer.projection import project
from tests.counterfactual.support import fixture
from tests.database.knowledge_support_04a import sql
from tests.memory import test_accepted_sources as accepted
from tests.memory import test_writer
from tests.memory.support import second_case
from tests.memory.test_memory import record
from tests.memory.test_relationships import counts

installed = test_writer.installed


def state(m):
    """Include tombstones and actual status revisions, not only row counts."""
    return dict(
        counts=counts(m),
        withdrawals=sql(m.db.admin, "SELECT * FROM memory.withdrawal ORDER BY identity"),
        statuses=sql(m.db.admin, "SELECT * FROM memory.status ORDER BY key"),
        bytes=sql(m.db.admin, "SELECT key,checksum,payload::text FROM memory.record ORDER BY key"),
    )


def historical(result, saved, expected="candidate"):
    assert canonical(result.record) == canonical(saved.record)
    assert result.checksum == saved.checksum
    assert result.state == expected and not result.current
    assert not result.eligible_for_planning and not result.eligible_for_retrieval


def test_legacy_owner_valid_synchronize(installed, monkeypatch):
    x = installed
    monkeypatch.setitem(x.m.registry.owners, x.rid, x.m.owners)
    saved = x.m.writer.create(record(x.m, "r02-legacy-valid"))
    assert saved.record.schema_version == "research-memory/1.0.0"
    before = state(x.m)
    historical(x.m.writer.synchronize(saved.record.key), saved)
    historical(x.m.writer.synchronize(saved.record.key), saved)
    assert state(x.m) == before
    assert x.m.reader.read(saved.record.key).current


def test_writer_valid_synchronize(installed):
    x = installed
    saved = x.writer.write(run_id=x.rid, request_key="r02-valid")
    assert saved.record.schema_version == "research-memory/1.1.0"
    before = state(x.m)
    historical(x.m.writer.synchronize(saved.record.key), saved)
    historical(x.m.writer.synchronize(saved.record.key), saved)
    assert state(x.m) == before
    assert x.m.reader.read(saved.record.key).current


@pytest.mark.parametrize("fault", ["object_io", "database_io", "missing_port"])
def test_unknown_io_or_missing_port_never_withdraws(installed, monkeypatch, fault):
    x = installed
    saved = x.writer.write(run_id=x.rid, request_key="r02-" + fault)
    before = state(x.m)
    if fault == "missing_port":
        monkeypatch.delitem(x.m.registry.owners, x.rid)
        cls, code = ValueError, "memory_source_port_unavailable"
    else:

        def fail(*args, **kwargs):
            if fault == "object_io":
                raise OSError("synthetic object failure")
            raise psycopg.OperationalError("synthetic original Runtime I/O")

        if fault == "object_io":
            monkeypatch.setattr(x.m.pg.documents, "read", fail)
        else:
            monkeypatch.setattr(x.runtime, "read_memory_execution", fail)
        cls, code = MemoryUnavailable, "memory_source_unavailable"
    with pytest.raises(cls, match="^" + code + "$"):
        x.m.writer.synchronize(saved.record.key)
    assert state(x.m) == before
    historical(x.m.reader.read(saved.record.key, historical=True), saved)


def test_owner_withdrawal_synchronize_and_dependencies(installed, monkeypatch):
    # Separate original case/run prevents persistent tombstones affecting other tests.
    m = installed.m
    pg = second_case(m.pg)
    pg.target = m.pg.target
    source_case = fixture(pg=pg)
    owners = ResearchSources(
        fact_reader=source_case.reader,
        applicability=("USE:SYNTHETIC",),
        simulation=source_case.service,
        documents=pg.documents,
    )
    local = SimpleNamespace(**{**vars(m), "pg": pg, "x": source_case, "owners": owners})
    setup = test_writer.installed.__wrapped__(local)
    x = next(setup)
    try:
        saved = x.writer.write(run_id=x.rid, request_key="r02-withdraw")
        # Original 1.0 adapter, no verify_record method added or simulated.
        monkeypatch.setitem(m.registry.owners, x.rid, owners)
        legacy = m.writer.create(record(local, "r02-legacy-withdraw", kind="graph_snapshot"))
        monkeypatch.setitem(m.registry.owners, x.rid, x.sources)
        child = m.writer.create(
            record(
                local,
                "r02-dependent",
                parents=(
                    Dependency(
                        key=saved.record.key, checksum=saved.checksum, relation="derived_from"
                    ),
                ),
            )
        )
        descendant = m.writer.create(
            child.record.model_copy(
                update=dict(
                    key="r02-transitive",
                    parents=(
                        Dependency(
                            key=child.record.key, checksum=child.checksum, relation="derived_from"
                        ),
                    ),
                )
            )
        )
        monkeypatch.setitem(m.registry.owners, x.rid, x.sources)
        segment_key = saved.record.detail.segments[0]
        segment = m.reader.read(segment_key)
        before = state(m)
        monkeypatch.setattr(source_case.f.reader, "withdrawn", True)
        with pytest.raises(ValueError, match="^source_withdrawn$"):
            m.reader.read(saved.record.key)
        # Segment has only runtime_execution. Its withdrawal invalidates the
        # Episode, then two descendants carrying only research_fact sources.
        assert {s.owner for s in segment.record.sources} == {"runtime_execution"}
        historical(m.writer.synchronize(segment_key), segment, "deprecated")
        historical(m.reader.read(saved.record.key, historical=True), saved, "deprecated")
        historical(m.reader.read(child.record.key, historical=True), child, "deprecated")
        historical(m.reader.read(descendant.record.key, historical=True), descendant, "deprecated")
        historical(m.reader.read(legacy.record.key, historical=True), legacy)
        assert len(state(m)["withdrawals"]) - len(before["withdrawals"]) == 1
        monkeypatch.setitem(m.registry.owners, x.rid, owners)
        historical(m.writer.synchronize(legacy.record.key), legacy, "deprecated")
        historical(m.reader.read(child.record.key, historical=True), child, "deprecated")
        historical(m.reader.read(descendant.record.key, historical=True), descendant, "deprecated")
        monkeypatch.setitem(m.registry.owners, x.rid, x.sources)
        historical(m.writer.synchronize(saved.record.key), saved, "deprecated")
        after = state(m)
        assert after["bytes"] == before["bytes"]
        n = len(after["withdrawals"]) - len(before["withdrawals"])
        assert n > 0
        for table in ("memory.record", "memory.source", "memory.dependency", "memory.status"):
            assert after["counts"][table] == before["counts"][table]
        for table in ("memory.request", "audit.event"):
            assert after["counts"][table] - before["counts"][table] == n
        monkeypatch.setitem(m.registry.owners, x.rid, owners)
        historical(m.writer.synchronize(legacy.record.key), legacy, "deprecated")
        monkeypatch.setitem(m.registry.owners, x.rid, x.sources)
        historical(m.writer.synchronize(saved.record.key), saved, "deprecated")
        assert state(m) == after
        for item in (saved, legacy, child, descendant):
            with pytest.raises(ValueError, match="^memory_parent_invalid$"):
                m.reader.read(item.record.key)
    finally:
        setup.close()


@pytest.mark.parametrize("missing_port", [False, True])
def test_r01_all_omissions(installed, monkeypatch, missing_port):
    accepted.test_repository_cannot_omit_accepted_facts(installed, monkeypatch, missing_port)


def test_r01_partial_omission(installed):
    accepted.test_partial_set_rejected_at_repository_and_revalidation(installed, "scores")


def test_bad_aggregate_sync_does_not_authorize_or_guess_withdrawal(installed, monkeypatch):
    x = installed
    key = "r02-old-bad"
    snapshot = x.sources.snapshot()
    values = project(snapshot, x.sources.snapshot_reference(snapshot), request_key=key)
    # Only illegal historical setup bypasses the envelope guard; real SQL/Audit
    # and per-source checks remain. Never use this as a successful write example.
    with monkeypatch.context() as patch:
        patch.setattr(x.sources, "verify_record", lambda record: None)
        saved = x.m.writer.create_episode(key, values)
    before = state(x.m)
    historical(x.m.writer.synchronize(saved.record.key), saved)
    for current_read in (x.m.reader.read, x.writer.replay):
        with pytest.raises(ValueError, match="^memory_accepted_sources_incomplete$"):
            current_read(saved.record.key)
    monkeypatch.setattr(x.sources, "research", None)
    historical(x.m.writer.synchronize(saved.record.key), saved)
    with pytest.raises(ValueError, match="^memory_fact_reader_required$"):
        x.m.reader.read(saved.record.key)
    assert state(x.m) == before


def test_complete_r01_regression(installed):
    accepted.test_complete_bytes_recovery_conflict_and_withdrawal(installed)


def test_legal_missing_fields_and_unknown_unchanged(installed, monkeypatch):
    accepted.test_unaccepted_root_does_not_require_fact_reader(installed, monkeypatch)
    test_writer.test_pre_graph_failure_no_fact_reader(installed)
    test_writer.test_original_control_termination_matrix(installed, "unknown")
