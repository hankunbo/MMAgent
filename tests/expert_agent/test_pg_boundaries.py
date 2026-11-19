"""Two independent lifecycle probes through original Runtime/Audit in owned PG."""

from threading import Event

import pytest

from db import manage
from packages.grounding.models import digest
from tests.agent_provider.support import review_from_wire
from tests.database.knowledge_support_04a import free_port, prepare_database
from tests.expert_agent.pg_support import runtime_fixture
from tests.expert_agent.support import setup
from tests.expert_agent.test_review_boundaries import integrated


@pytest.fixture
def database(database_lease):
    # runtime_fixture uses fixed synthetic identities: a separate owned cluster per
    # parameter avoids an unrelated case_record identity collision between probes.
    with database_lease.cluster(free_port()):
        db = prepare_database()
        manage.bootstrap_grounding_principals(
            db.admin, *(manage.test_target(c).user for c in manage.GROUNDING_CHANNELS)
        )
        manage.bootstrap_runtime_principal(db.admin, manage.test_target("runtime").user)
        yield db


@pytest.mark.parametrize("change", ["cancel_after_accept", "withdraw_after_accept"])
def test_pg_confirmed_acceptance_stops_without_losing_facts(
    database, tmp_path, monkeypatch, change
):
    pg = runtime_fixture(tmp_path)
    f = setup(
        [review_from_wire],
        binding=pg.binding,
        reader=pg.reader,
        documents=pg.documents,
        budget=pg.budget,
        config=pg.config,
    )
    h = integrated(f, runtime=pg.documents.sessions)
    signal = Event()
    accept = h.service.acceptance.accept
    history, calls = [], []

    def accepted_then_change(*args, **kwargs):
        value = accept(*args, **kwargs)
        calls.append(1)
        history.append(h.service.acceptance.read(key=h.args["event"].code + ":domain"))
        if change == "cancel_after_accept":
            signal.set()
        else:
            f.reader.withdrawn = True
        return value

    monkeypatch.setattr(h.service.acceptance, "accept", accepted_then_change)
    errors, result = [], None
    for _ in range(2):
        try:
            result = h.service.run(**h.args, cancel=signal)
        except ValueError as error:
            errors.append(str(error))
        signal.clear()
    state = h.planner.read(f.binding.run_id)
    receipt = h.service.acceptance.read(key=h.args["event"].code + ":domain")[-1]
    from tests.database.knowledge_support_04a import sql

    attempts = sql(database.admin, "SELECT count(*) FROM runtime.tool_attempt")[0][0]
    print(
        {
            "probe": change,
            "storage": "isolated_pg",
            "phase": state.phase,
            "cancelled": state.cancelled,
            "pending": state.pending is not None,
            "domain_status": receipt["status"],
            "result": None if result is None else result["status"],
            "errors": errors,
            "requests": f.budget.used,
            "tool_attempts": attempts,
        }
    )
    assert receipt["status"] == "accepted" and f.budget.used == 1 and attempts == 0
    assert not errors, (
        "confirmed facts remain but source-withdrawal recovery has no terminal checkpoint"
    )
    assert state.phase == "stopped" and state.pending is None
    if change == "cancel_after_accept":
        assert state.cancelled
    key = h.args["event"].code
    assert len(calls) == 1 and state.reserved == 1 and state.started == 0
    assert h.service.acceptance.read(key=key + ":domain") == history[0]
    assert pg.store.read_bounded(pg.root.uri, 10000) == pg.raw
    report = h.service.store.read(key + ":report")[0]
    html = h.service.store.read(key + ":html")[0]
    assert report["receipt"] == history[0][-1]
    assert report["records"] == history[0][0] and report["graph"] == history[0][1]
    assert report["scores"] == history[0][2]["scores"]
    assert result["domain_status"] == "accepted" and result["continuation"] == "stop"
    assert result["report_checksum"] == html["report_checksum"] == digest(report)
    assert receipt["graph"]["checksum"] in html["html"]
    control_report = h.planner.report(f.binding.run_id)
    assert control_report.reason == (
        "cancelled" if change == "cancel_after_accept" else "source_withdrawn"
    )
    assert any(r.checksum == digest(result) for r in control_report.result_refs)
    audits = sql(
        database.admin,
        "SELECT payload FROM audit.event WHERE run_id=%s "
        "AND correlation_code='SOURCE:PLANNER_CONTROL' ORDER BY event_sequence",
        (f.binding.run_id,),
    )
    assert audits[-1][0]["state_checksum"] == digest(state)
    assert audits[-1][0]["pending"] is None and audits[-1][0]["phase"] == "stopped"
    assert audits[-1][0]["event"] == state.events[-1].model_dump(mode="json")
    assert len(audits) == state.revision + 1
    before = sql(database.admin, "SELECT count(*) FROM audit.event")[0][0]
    assert h.service.run(**h.args) == result
    assert sql(database.admin, "SELECT count(*) FROM audit.event")[0][0] == before
    assert sql(database.admin, "SELECT count(*) FROM runtime.run")[0][0] == 1
    assert h.planner.read(f.binding.run_id) == state
