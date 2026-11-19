"""Relationship regressions through original owners and non-superuser Memory ports."""

from types import SimpleNamespace

import pytest

from apps.memory_sources import ResearchSources
from packages.grounding.models import digest
from packages.memory.models import Dependency, Record
from tests.counterfactual.support import fixture
from tests.database.knowledge_support_04a import sql
from tests.memory.support import second_case
from tests.memory.test_memory import record


def counts(m):
    return {
        t: sql(m.db.admin, "SELECT count(*) FROM " + t)[0][0]
        for t in (
            "memory.record",
            "memory.source",
            "memory.dependency",
            "memory.status",
            "memory.request",
            "audit.event",
        )
    }


@pytest.fixture(scope="module")
def relations(memory):
    m = memory
    pg = second_case(m.pg)
    x = fixture(pg=pg)
    owner = ResearchSources(
        fact_reader=x.reader,
        applicability=("USE:SYNTHETIC",),
        simulation=x.service,
        documents=pg.documents,
    )
    m.registry.owners[pg.binding.run_id] = owner
    source = owner.reference("research_fact", x.reader.path_reader.key, "graph")
    foreign = m.writer.create(
        Record(
            key="rel-foreign",
            kind="graph_snapshot",
            case_code=source.case_code,
            run_id=source.run_id,
            applicability=source.applicability,
            sources=(source,),
            nature="source_fact",
        )
    )
    local = m.writer.create(record(m, "rel-local", kind="graph_snapshot"))
    assert foreign.record.case_code != local.record.case_code
    return SimpleNamespace(m=m, local=local, foreign=foreign)


@pytest.mark.parametrize("relation", ["episode_member", "revision_of"])
def test_foreign_case_relation_must_be_rejected(relations, relation):
    m, parent = relations.m, relations.foreign
    value = record(
        m,
        "rel-cross-" + relation,
        kind="graph_snapshot" if relation == "revision_of" else "episode",
        parents=(Dependency(key=parent.record.key, checksum=parent.checksum, relation=relation),),
    )
    before = counts(m)
    error = None
    try:
        m.writer.create(value)
    except ValueError as exc:
        error = str(exc)
    after = counts(m)
    assert error == "memory_relation_scope", dict(error=error, before=before, after=after)
    assert after == before


def test_same_scope_member_and_explicit_abstract(relations):
    m, local, foreign = relations.m, relations.local, relations.foreign
    saved = m.writer.create(
        record(
            m,
            "rel-episode",
            parents=(
                Dependency(
                    key=local.record.key, checksum=local.checksum, relation="episode_member"
                ),
            ),
        )
    )
    assert saved.current
    value = Record(
        key="rel-abstract",
        kind="abstract_pattern",
        case_code=local.record.case_code,
        run_id=local.record.run_id,
        applicability=local.record.applicability,
        sources=local.record.sources + foreign.record.sources,
        nature="abstract_candidate",
        parents=(
            Dependency(key=local.record.key, checksum=local.checksum),
            Dependency(key=foreign.record.key, checksum=foreign.checksum),
        ),
    )
    assert m.writer.create(value).current


@pytest.fixture(scope="module")
def rerun(relations):
    m = relations.m
    pg = second_case(m.pg, same_case=True)
    x = fixture(pg=pg)
    owner = ResearchSources(
        fact_reader=x.reader,
        applicability=("USE:SYNTHETIC",),
        simulation=x.service,
        documents=pg.documents,
    )
    m.registry.owners[pg.binding.run_id] = owner
    source = owner.reference("research_fact", x.reader.path_reader.key, "graph")
    value = Record(
        key="rel-rerun",
        kind="graph_snapshot",
        case_code=source.case_code,
        run_id=source.run_id,
        applicability=source.applicability,
        sources=(source,),
        nature="source_fact",
    )
    assert value.case_code == relations.local.record.case_code
    assert value.run_id != relations.local.record.run_id
    return m.writer.create(value)


@pytest.mark.parametrize("relation", ["episode_member", "revision_of", "derived_from"])
def test_same_case_other_run_rejected(relations, rerun, relation):
    m = relations.m
    child = record(
        m,
        "rel-rerun-" + relation,
        kind="episode" if relation == "episode_member" else "graph_snapshot",
        parents=(Dependency(key=rerun.record.key, checksum=rerun.checksum, relation=relation),),
    )
    before = counts(m)
    with pytest.raises(ValueError, match="^memory_relation_scope$"):
        m.writer.create(child)
    assert counts(m) == before


@pytest.mark.parametrize(
    "relation,child_kind,parent_kind",
    [
        ("episode_member", "graph_snapshot", "graph_snapshot"),
        ("episode_member", "episode", "episode"),
        ("episode_member", "episode", "abstract_pattern"),
        ("revision_of", "episode", "graph_snapshot"),
        ("revision_of", "abstract_pattern", "abstract_pattern"),
        ("derived_from", "episode", "abstract_pattern"),
    ],
)
def test_illegal_kind_and_constructed_revalidation(relations, relation, child_kind, parent_kind):
    m = relations.m
    parent = {
        "graph_snapshot": relations.local,
        "episode": m.reader.read("rel-episode"),
        "abstract_pattern": m.reader.read("rel-abstract"),
    }[parent_kind]
    child = record(
        m,
        "rel-kind-" + relation + "-" + child_kind + "-" + parent_kind,
        parents=(Dependency(key=parent.record.key, checksum=parent.checksum, relation=relation),),
    )
    raw = dict(child)
    raw.update(
        kind=child_kind,
        nature="abstract_candidate" if child_kind == "abstract_pattern" else "source_fact",
    )
    child = Record.model_construct(**raw)
    before = counts(m)
    with pytest.raises(ValueError, match="^memory_relation_kind$"):
        m.writer.create(child)
    assert counts(m) == before


def test_ordinary_derived_does_not_gain_cross_case_scope(relations):
    m, parent = relations.m, relations.foreign
    child = record(
        m,
        "rel-derived-foreign",
        parents=(Dependency(key=parent.record.key, checksum=parent.checksum),),
    )
    before = counts(m)
    with pytest.raises(ValueError, match="^memory_relation_scope$"):
        m.writer.create(child)
    assert counts(m) == before


def test_abstract_source_union_and_applicability(relations):
    m = relations.m
    child = m.reader.read("rel-abstract").record
    before = counts(m)
    with pytest.raises(ValueError, match="^memory_derivation_sources$"):
        m.writer.create(
            child.model_copy(
                update={"key": "rel-union-missing", "sources": relations.local.record.sources}
            )
        )
    # Parent may narrow its domain even if its original source allows more.
    from packages.memory.models import validate_relationships

    narrow = relations.local.record.model_copy(update={"applicability": ("USE:NARROW",)})
    probe = record(
        m, "rel-applicability", parents=(Dependency(key=narrow.key, checksum=digest(narrow)),)
    )
    with pytest.raises(ValueError, match="^memory_relation_applicability$"):
        validate_relationships(probe, {narrow.key: narrow})
    assert counts(m) == before


def insert_raw(c, child):
    """SQL writer probe; valid own sources/FKs, deliberately invalid relationship."""
    from psycopg.types.json import Jsonb

    c.execute(
        "INSERT INTO memory.record(key,kind,case_code,run_id,checksum,payload) "
        "VALUES(%s,%s,%s,%s,%s,%s)",
        (
            child.key,
            child.kind,
            child.case_code,
            child.run_id,
            digest(child),
            Jsonb(child.model_dump(mode="json")),
        ),
    )
    c.execute("INSERT INTO memory.status(key) VALUES(%s)", (child.key,))
    for n, s in enumerate(child.sources):
        c.execute(
            "INSERT INTO memory.source VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                child.key,
                n,
                s.owner,
                s.key,
                s.kind,
                s.version,
                s.case_code,
                s.run_id,
                s.checksum,
                Jsonb(s.model_dump(mode="json")),
            ),
        )
    for p in child.parents:
        c.execute(
            "INSERT INTO memory.dependency VALUES(%s,%s,%s,%s)",
            (child.key, p.key, p.checksum, p.relation),
        )


@pytest.mark.parametrize(
    "relation,foreign,kind,error",
    [
        ("episode_member", True, "episode", "memory_relation_scope"),
        ("revision_of", True, "graph_snapshot", "memory_relation_scope"),
        ("derived_from", True, "episode", "memory_relation_scope"),
        ("episode_member", False, "graph_snapshot", "memory_relation_kind"),
        ("revision_of", False, "episode", "memory_relation_kind"),
    ],
)
def test_sql_non_super_writer_rejects_atomically(relations, relation, foreign, kind, error):
    import psycopg

    from db import manage
    from tests.database.knowledge_support_04a import connect

    m = relations.m
    parent = relations.foreign if foreign else relations.local
    child = record(
        m,
        "rel-sql-" + relation + "-" + kind,
        kind=kind,
        parents=(Dependency(key=parent.record.key, checksum=parent.checksum, relation=relation),),
    )
    before = counts(m)
    with pytest.raises(psycopg.errors.RaiseException) as rejected:
        with connect(manage.test_target("memory_writer")) as c:
            assert not c.execute(
                "SELECT rolsuper FROM pg_roles WHERE rolname=session_user"
            ).fetchone()[0]
            c.execute("SET ROLE mmagent_memory_writer")
            insert_raw(c, child)
    assert rejected.value.diag.message_primary == error
    assert counts(m) == before


def test_legal_revision_idempotency_and_parent_withdrawal(relations):
    m = relations.m
    parent = m.writer.create(record(m, "rel-withdraw-parent", kind="graph_snapshot"))
    revision = record(
        m,
        "rel-revision",
        kind="graph_snapshot",
        parents=(
            Dependency(key=parent.record.key, checksum=parent.checksum, relation="revision_of"),
        ),
    )
    member = record(
        m,
        "rel-withdraw-episode",
        parents=(
            Dependency(
                key=revision.key,
                checksum=digest(revision),
                relation="episode_member",
            ),
        ),
    )
    one = m.writer.create(revision)
    before = counts(m)
    assert m.writer.create(revision) == one
    assert counts(m) == before
    m.writer.create(member)
    assert m.reader.read(member.key).current
    assert not m.reader.read(member.key, historical=True).current
    m.writer.withdraw("rel-withdraw", parent.record.sources[0])
    for key in (parent.record.key, revision.key, member.key):
        assert m.reader.read(key, historical=True).state == "deprecated"
        with pytest.raises(ValueError, match="^memory_parent_invalid$"):
            m.reader.read(key)
    before = counts(m)
    with pytest.raises(ValueError, match="^memory_parent_invalid$"):
        m.writer.create(revision)
    assert counts(m) == before
