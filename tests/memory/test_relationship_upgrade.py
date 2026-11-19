"""Real 0001 -> 0002 migration: clean upgrade or atomic legacy-invalid refusal."""

from pathlib import Path

import pytest

from db import manage
from packages.grounding.models import digest
from packages.memory.models import Dependency
from tests.database.knowledge_support_04a import connect, sql
from tests.memory.conftest import memory as memory_setup
from tests.memory.test_memory import record
from tests.memory.test_relationships import counts, insert_raw
from tests.memory.test_relationships import relations as relation_setup


@pytest.fixture(params=["clean", "invalid"])
def legacy(request, database_lease, tmp_path_factory):
    original = manage._migration_files
    # Select the real old migration set without modifying any file or checksum.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            manage,
            "_migration_files",
            lambda domain: tuple(
                p for p in original(domain) if domain != "memory" or p.name.startswith("0001_")
            ),
        )
        generator = memory_setup.__wrapped__(database_lease, tmp_path_factory)
        m = next(generator)
    try:
        yield m, request.param
    finally:
        with pytest.raises(StopIteration):
            next(generator)


def test_incremental_migration_and_legacy_current_boundary(legacy):
    m, mode = legacy
    assert set(manage.recorded_migrations(m.db.migration, "memory")) == {"0001"}
    parent = m.writer.create(record(m, "upgrade-parent", kind="graph_snapshot"))
    valid = record(
        m,
        "upgrade-episode",
        parents=(
            Dependency(key=parent.record.key, checksum=parent.checksum, relation="episode_member"),
        ),
    )
    saved = m.writer.create(valid)
    if mode == "invalid":
        r = relation_setup.__wrapped__(m)
        bad = []
        for relation, kind in [
            ("episode_member", "episode"),
            ("revision_of", "graph_snapshot"),
            ("derived_from", "episode"),
        ]:
            child = record(
                m,
                "legacy-" + relation,
                kind=kind,
                parents=(
                    Dependency(
                        key=r.foreign.record.key, checksum=r.foreign.checksum, relation=relation
                    ),
                ),
            )
            # Valid non-superuser writer rows in the actual old schema reproduce legacy data.
            with connect(manage.test_target("memory_writer")) as c:
                assert not c.execute(
                    "SELECT rolsuper FROM pg_roles WHERE rolname=session_user"
                ).fetchone()[0]
                c.execute("SET ROLE mmagent_memory_writer")
                insert_raw(c, child)
            bad.append(child)
        descendant = record(
            m, "legacy-descendant", parents=(Dependency(key=bad[0].key, checksum=digest(bad[0])),)
        )
        with connect(manage.test_target("memory_writer")) as c:
            c.execute("SET ROLE mmagent_memory_writer")
            insert_raw(c, descendant)
        bad.append(descendant)
        for child in bad:
            historical = m.reader.read(child.key, historical=True)
            assert historical.record == child and historical.checksum == digest(child)
            assert not historical.current
            with pytest.raises(ValueError, match="^memory_relation_scope$"):
                m.reader.read(child.key)
            before = counts(m)
            with pytest.raises(ValueError, match="^memory_relation_scope$"):
                m.writer.create(
                    record(
                        m,
                        "legacy-new-" + child.key,
                        parents=(Dependency(key=child.key, checksum=digest(child)),),
                    )
                )
            assert counts(m) == before
        rows = sql(m.db.admin, "SELECT key,checksum,payload::text FROM memory.record ORDER BY key")
        before = counts(m)
        with pytest.raises(manage.DatabaseCommandError, match="memory_relation_legacy_invalid"):
            manage.migrate(m.db.migration)
        assert set(manage.recorded_migrations(m.db.migration, "memory")) == {"0001"}
        assert sql(
            m.db.admin, "SELECT to_regprocedure('memory.relationship_error(jsonb,jsonb,text)')"
        ) == [(None,)]
        assert counts(m) == before
        assert (
            sql(m.db.admin, "SELECT key,checksum,payload::text FROM memory.record ORDER BY key")
            == rows
        )
        for child in bad:
            assert m.reader.read(child.key, historical=True).record == child
            with pytest.raises(ValueError, match="^memory_relation_scope$"):
                m.reader.read(child.key)
    else:
        rows = sql(m.db.admin, "SELECT key,checksum,payload::text FROM memory.record ORDER BY key")
        manage.migrate(m.db.migration)
        versions = manage.recorded_migrations(m.db.migration, "memory")
        assert set(versions) == {"0001", "0002"}
        assert versions["0002"] == manage._checksum(
            Path("db/memory/migrations/0002_relationship_scope.sql")
        )
        manage.migrate(m.db.migration)
        assert manage.recorded_migrations(m.db.migration, "memory") == versions
        assert (
            sql(m.db.admin, "SELECT key,checksum,payload::text FROM memory.record ORDER BY key")
            == rows
        )
        assert m.reader.read(valid.key) == saved
        assert m.writer.create(valid) == saved
        assert not m.reader.read(valid.key, historical=True).current
