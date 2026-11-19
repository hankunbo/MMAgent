"""Real owned PG cancellation, lost commits, Audit and same-event recovery."""

from threading import Event

import pytest

from packages.grounding.models import digest
from tests.database.knowledge_support_04a import sql
from tests.expert_agent.pg_support import runtime_fixture
from tests.expert_agent.test_pg_boundaries import database as database
from tests.path_graph_validation.pg_boundaries_support import ready_pg


@pytest.mark.parametrize("point", ["binding", "report"])
def test_pg_binding_and_control_commit_loss_keeps_cancel_and_facts(
    database, tmp_path, monkeypatch, point
):
    pg = runtime_fixture(tmp_path)
    f, h, service, summary, args = ready_pg(pg)
    event = args["event"].code
    signal = Event()
    before = h.planner.read(f.binding.run_id)
    facts = {
        a.asset_code: pg.documents.read_artifact(a)
        for a in pg.repo.get_case_bundle(f.binding.case_code).artifacts
    }
    write, save = pg.documents.write, pg.documents.sessions.save
    lost = []

    def binding_lost(key, values):
        value = write(key, values)
        failure_key = (
            "review:" + event + ":path-graph-binding"
            if point == "binding"
            else "path-graph-review:" + event
        )
        if key == failure_key:
            signal.set()
            raise OSError("pg_binding_confirmation_lost")
        return value

    def control_lost(old, new):
        value = save(old, new)
        if new.cancelled and not lost:
            lost.append(1)
            raise OSError("pg_control_confirmation_lost")
        return value

    with monkeypatch.context() as m:
        m.setattr(pg.documents, "write", binding_lost)
        m.setattr(pg.documents.sessions, "save", control_lost)
        with pytest.raises(OSError, match="pg_binding_confirmation_lost") as error:
            service.review(**args, cancel=signal)
    assert str(error.value.__cause__) == "pg_control_confirmation_lost"
    state = h.planner.read(f.binding.run_id)
    assert state.cancelled and state.pending == before.pending
    claim = pg.documents.read("review:" + event + ":path-graph-binding")
    requests = 1 if point == "binding" else 2
    assert claim and f.budget.used == requests
    report_before = pg.documents.read("path-graph-review:" + event)
    signal.clear()
    result = service.review(**args)
    assert result["result"]["status"] == ("safe_stop" if point == "binding" else "accept")
    assert result["result"]["domain_status"] == "not_accepted"
    assert f.budget.used == requests and f.budget.summary().api_requests_observed == 0
    if report_before:
        assert pg.documents.read("path-graph-review:" + event) == report_before
    state = h.planner.replay(f.binding.run_id)
    assert state.cancelled and state.pending is None and state.phase == "stopped"
    assert (state.graph, state.reserved, state.started) == (
        before.graph,
        before.reserved,
        before.started,
    )
    assert pg.documents.read("review:" + event + ":path-graph-binding") == claim
    assert pg.documents.read("accepted:" + event + ":domain") is None
    assert service.read(key="summary", historical=True) == summary
    bundle = pg.repo.get_case_bundle(f.binding.case_code)
    assert all(
        pg.documents.read_artifact(a) == facts[a.asset_code]
        for a in bundle.artifacts
        if a.asset_code in facts
    )
    assert pg.store.read_bounded(pg.root.uri, 10000) == pg.raw
    audits = sql(
        database.admin,
        "SELECT payload FROM audit.event WHERE run_id=%s "
        "AND correlation_code='SOURCE:PLANNER_CONTROL' ORDER BY event_sequence",
        (f.binding.run_id,),
    )
    assert audits[-1][0]["state_checksum"] == digest(state)
    assert audits[-1][0]["pending"] is None and audits[-1][0]["phase"] == "stopped"
    assert len(audits) == state.revision + 1
    count = sql(database.admin, "SELECT count(*) FROM audit.event")[0][0]
    assert service.review(**args) == result
    assert sql(database.admin, "SELECT count(*) FROM audit.event")[0][0] == count
    assert sql(database.admin, "SELECT count(*) FROM runtime.run")[0][0] == 1
    assert sql(database.admin, "SELECT count(*) FROM runtime.tool_attempt")[0][0] == 0
