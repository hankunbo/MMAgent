"""Explicit synthetic query inputs; no patient/tool execution or inferred unknowns."""

from datetime import UTC, datetime, timedelta

from packages.grounding.models import digest
from packages.memory_review.models import Binding, Conditions, Policy, Query


def conditions():
    return Conditions(
        goal="GOAL:SYNTHETIC",
        template="TEMPLATE:SYNTHETIC",
        graph_conditions=("unresolved",),
        modalities=("SLO",),
        inputs=("image",),
        patient_conditions={"laterality": "left"},
        tool_versions={},
        rule_versions={"synthetic_rule": "1.0.0"},
        failure_types=("missing_input",),
        domain="fixture",
    )


def binding(record=None):
    return Binding(
        target=record.key if record else "candidate",
        target_checksum=digest(record or "candidate"),
        content_version=record.schema_version if record else "research-memory/1.2.0",
        conditions=conditions(),
        group="recovery",
        preferred_branches=("B", "ILLEGAL"),
        terms=("missing_input",),
    )


def policy(key="test-policy", reviewer="reviewer", now=None):
    now = now or datetime.now(UTC)
    identities = {
        "mmagent_test_memory_writer": "synthetic_author",
        "mmagent_test_memory_publisher": "synthetic_publisher",
    }
    identities.setdefault(reviewer, "synthetic_reviewer")
    return Policy(
        key=key,
        approved=True,
        purpose="synthetic_offline",
        test_context="10d-owned-fixture",
        min_cases=2,
        min_reviewers=1,
        reviewers=(reviewer,),
        identities=identities,
        domains=("fixture",),
        valid_from=now - timedelta(seconds=1),
        expires_at=now + timedelta(hours=2),
        validation="synthetic_engineering",
    )


class QueryOwner:
    def __init__(self, run_id, case_code):
        self.value = Query(
            **conditions().model_dump(),
            run_id=run_id,
            case_code=case_code,
            test_context="10d-owned-fixture",
            text=("missing_input",),
        )
        self.calls = 0
        self.tool_expired = False
        self.fail = False

    def current(self, run_id):
        assert run_id == self.value.run_id
        self.calls += 1
        if self.fail:
            raise OSError("synthetic object I/O failure")
        return self.value

    def check_versions(self, query):
        if self.tool_expired:
            raise ValueError("memory_tool_expired")


def pg_planner(x, tag, *, empty=False):
    from types import SimpleNamespace

    from agents.planner.service import Planner
    from packages.memory_review.planner import MemoryRankedActions
    from packages.planner_control.models import Command
    from packages.research_runtime.agent_records import RuntimeDocuments
    from packages.retrieval.memory import MemoryRetrieval
    from tests.memory.test_writer import clone
    from tests.planner.support import FakeOwner
    from tests.planner.test_actions import Source

    host = SimpleNamespace(m=x, runtime=x.contexts, rid=x.pg.binding.run_id)
    branch = clone(host, tag, planner=True, profile={"seconds": 3600})
    b = branch.planner.read(branch.rid).binding
    docs = RuntimeDocuments(
        repo=x.pg.repo,
        contexts=x.contexts,
        sessions=x.pg.documents.sessions,
        store=x.pg.store,
        run_id=branch.rid,
        root=x.pg.root,
    )
    q = QueryOwner(branch.rid, b.case_code)
    if empty:
        q.value = q.value.model_copy(update={"groups": ()})
    retrieval = MemoryRetrieval(repository=x.reader, query_reader=q)
    source = Source(b)
    owner = FakeOwner(b)
    owner.choices = source.choices
    rank = MemoryRankedActions(source, retrieval=retrieval, documents=docs)
    planner = Planner(runtime=x.pg.documents.sessions, owner=owner, ranking=rank)
    for action in ("load", "retrieve", "instantiate", "identify"):
        state = planner.read(branch.rid)
        planner.step(
            branch.rid,
            Command(key=tag + "-" + action, action=action, expected_revision=state.revision),
        )
    state = planner.read(branch.rid)
    cmd = Command(key=tag + "-select", action="select", expected_revision=state.revision)
    return SimpleNamespace(
        planner=planner,
        command=cmd,
        rank=rank,
        retrieval=retrieval,
        source=source,
        owner=owner,
        documents=docs,
        query=q,
        rid=branch.rid,
    )
