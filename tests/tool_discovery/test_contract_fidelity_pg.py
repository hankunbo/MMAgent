"""Persisted effect of the lossy input contract, via original 06/11A/11B APIs."""

import pytest

from db import manage
from db.runtime.repository.tool_discovery import DiscoveryRepository
from packages.grounding.models import Candidate, canonical, digest, validate
from packages.tool_discovery import Discovery
from packages.tool_discovery.discovery import contract
from packages.tool_discovery.versions import contract_status
from tests.database.knowledge_support_04a import sql
from tests.tool_discovery.conftest import offline_only
from tests.tool_discovery.legacy_support import legacy_discovery
from tests.tool_discovery.support import record, source
from tests.tool_discovery.test_contract_fidelity import changed, mapped_record
from tests.tool_discovery.test_postgres import (
    consume,
    export,
    host,
    scenario,
    snapshots,
)
from tests.tool_discovery.test_postgres import (
    test_audit_failure_rolls_back_result_and_ack_then_stops_pending as audit_rollback,
)
from tests.tool_discovery.test_postgres import (
    test_transition_during_discovery_never_saves_or_acknowledges as transition_rejection,
)


def test_two_image_candidate_is_not_selected_for_one_image_contract(host, scenario):
    s = scenario
    data = host.h.request.candidate.model_dump(mode="json")
    data["inputs"][0].update(required=True, cardinality_min=2, cardinality_max=2)
    alternate = validate(Candidate, data)
    p = record(host.h, semantics=contract(alternate.model_dump(mode="json")))
    s.d = Discovery(
        gap_queue=s.g,
        repository=DiscoveryRepository(manage.test_target("runtime")),
        sources=(source([p]),),
    )
    before = snapshots(host)
    saved = consume(s)
    after = snapshots(host)
    assert s.d.read(s.a.run_id, s.a.key) == saved
    assert before["run"] == after["run"] and after["attempts"] == 0
    print(
        dict(
            status=saved["status"],
            result_status=saved["result"]["status"],
            selected=saved["result"]["selected"],
            ack=saved["ack"]["state"],
            task_match=saved["result"]["candidates"][0]["dimensions"]["task_match"],
            requested_count=1,
            candidate_count=2,
            budget_charged=saved["budget_charged"],
        )
    )
    assert not saved["result"]["selected"], (
        "incompatible required cardinality persisted as selected/discovered"
    )
    assert saved["result"]["candidates"][0]["input_match"]["status"] == "incompatible"
    assert saved["result"]["status"] == "incomplete"
    assert saved["budget_charged"] == 1 and after["counts"][0] == before["counts"][0]
    assert consume(s) == saved and snapshots(host) == after
    export(
        "two-input-rejected",
        dict(synthetic=True, record=saved, tool_attempts=0, before=before, after=after),
    )


__all__ = ["host", "scenario", "offline_only"]


@pytest.mark.parametrize("mode", ["exact", "optional", "mapped"])
def test_current_contract_saved_without_extra_task_budget_or_tool(host, scenario, mode):
    s = scenario
    p = (
        record(host.h)
        if mode == "exact"
        else record(host.h, semantics=changed(host.h, required=False, cardinality_min=0))
        if mode == "optional"
        else mapped_record(host.h, extra=True)
    )
    s.d = Discovery(gap_queue=s.g, repository=s.d.repository, sources=(source([p]),))
    before = snapshots(host)
    saved = consume(s)
    after = snapshots(host)
    assert saved["status"] == "saved" and saved["result"]["selected"]
    assert saved["plan"]["schema_version"] == "tool-discovery-plan/1.1.0"
    assert saved["result"]["schema_version"] == "tool-discovery-result/1.1.0"
    assert saved["ack"]["state"] == "acknowledged" and saved["budget_charged"] == 1
    assert saved["result"]["candidates"][0]["records"][0] == p.model_dump(mode="json")
    assert saved["result"]["candidates"][0]["input_match"]["status"] == (
        "exact" if mode == "exact" else "compatible"
    )
    assert after["run"] == before["run"] and after["attempts"] == 0
    assert after["counts"][0] == before["counts"][0]
    assert after["counts"][1] == before["counts"][1]
    assert after["counts"][2] == before["counts"][2] + 2
    assert after["counts"][3] == before["counts"][3] + 4
    assert after["discovery"] == before["discovery"] + 1
    assert consume(s) == saved == s.d.read(s.a.run_id, s.a.key)
    assert snapshots(host) == after
    assert contract_status(saved) == "current_input_contract"
    export(
        "contract-" + mode,
        dict(synthetic=True, record=saved, tool_attempts=0, before=before, after=after),
    )


def test_current_transaction_failure_has_no_partial_result_or_ack(host, scenario, monkeypatch):
    before = snapshots(host)
    audit_rollback(host, scenario, monkeypatch)
    after = snapshots(host)
    assert after["attempts"] == 0 and after["run"] == before["run"]
    assert after["counts"][0] == before["counts"][0]
    assert after["discovery"] == before["discovery"] + 1
    assert scenario.d.read(scenario.a.run_id, scenario.a.key)["budget_charged"] == 1


def test_current_source_withdrawal_has_no_result_or_ack(host, scenario, monkeypatch):
    before = snapshots(host)
    transition_rejection(host, scenario, monkeypatch, "withdraw")
    after = snapshots(host)
    assert after["attempts"] == 0 and after["run"] == before["run"]
    assert after["counts"][0] == before["counts"][0]


@pytest.mark.parametrize("status", ["saved", "reserved"])
def test_legacy_history_same_key_is_unchanged_and_cannot_upgrade(
    host, scenario, monkeypatch, status
):
    s = scenario
    with legacy_discovery(s, monkeypatch) as old:
        if status == "reserved":

            def lost(*args, **kwargs):
                raise ConnectionError("synthetic_legacy_reserved_interruption")

            monkeypatch.setattr(old, "_evaluate", lost)
            with pytest.raises(ConnectionError, match="legacy_reserved"):
                old.consume(s.a.run_id, s.a.key, claim_key="discover", worker="offline")
            original = old.read(s.a.run_id, s.a.key)
        else:
            original = old.consume(s.a.run_id, s.a.key, claim_key="discover", worker="offline")
    before = snapshots(host)
    raw = canonical(original)
    plan_hash = digest(original["plan"])
    assert original["plan"]["schema_version"] == "tool-discovery-plan/1.0.0"
    assert original["status"] == status
    host.h.f.service.catalogs.disable(host.h.request.candidate.implementation_ref)

    def forbidden(*args, **kwargs):
        raise AssertionError("historical recovery cannot claim, rebuild owner or search")

    monkeypatch.setattr(s.d.sources[0], "read", forbidden)
    monkeypatch.setattr(s.g, "claim", forbidden)
    monkeypatch.setattr(s.g, "recheck", forbidden)
    saved = consume(s)
    assert canonical(saved) == raw and digest(saved["plan"]) == plan_hash
    assert contract_status(saved) == "legacy_missing_input_contract"
    history = s.d.acknowledge_saved(s.a.run_id, s.a.key)
    if status == "saved":
        from packages.tool_gap_queue.models import Receipt

        assert validate(Receipt, history["ack_history"]) == validate(Receipt, original["ack"])
        assert history["result"]["schema_version"] == "tool-discovery-result/1.0.0"
    else:
        assert history["ack_history"] is None and history["result"] is None
    assert snapshots(host) == before
    with pytest.raises(ValueError, match="pending_identity_conflict"):
        s.d.consume(s.a.run_id, s.a.key, claim_key="upgrade-key", worker="offline")
    rows = sql(host.db.admin, "SELECT document FROM runtime.tool_discovery ORDER BY task_id")
    commands = sql(
        host.db.admin,
        "SELECT to_jsonb(c) FROM runtime.tool_gap_command c ORDER BY task_id,command_key",
    )
    manage.migrate(host.db.migration)
    assert rows == sql(
        host.db.admin, "SELECT document FROM runtime.tool_discovery ORDER BY task_id"
    )
    assert commands == sql(
        host.db.admin,
        "SELECT to_jsonb(c) FROM runtime.tool_gap_command c ORDER BY task_id,command_key",
    )
    assert snapshots(host) == before
    export(
        "legacy-" + status,
        dict(
            synthetic=True,
            history_only=True,
            record=saved,
            contract_status=contract_status(saved),
            tool_attempts=0,
        ),
    )


def test_missing_current_sql_is_technical_without_legacy_fallback(host, scenario, monkeypatch):
    import psycopg

    s = scenario
    before = snapshots(host)

    def forbidden(*args, **kwargs):
        raise AssertionError("missing current SQL cannot discover")

    monkeypatch.setattr(s.d.sources[0], "read", forbidden)
    sql(
        host.db.admin,
        "ALTER FUNCTION runtime.apply_tool_discovery_v11(text,text,jsonb,jsonb,jsonb) "
        "RENAME TO unavailable_11b_v11",
    )
    try:
        with pytest.raises(psycopg.errors.UndefinedFunction):
            consume(s)
    finally:
        sql(
            host.db.admin,
            "ALTER FUNCTION runtime.unavailable_11b_v11(text,text,jsonb,jsonb,jsonb) "
            "RENAME TO apply_tool_discovery_v11",
        )
    assert s.d.read(s.a.run_id, s.a.key) is None
    after = snapshots(host)
    assert after["discovery"] == before["discovery"] and after["attempts"] == 0
    assert after["run"] == before["run"] and after["counts"][0] == before["counts"][0]
    assert after["counts"][2] == before["counts"][2] + 1  # original claim only
    assert after["counts"][3] == before["counts"][3] + 1
    s.g.cancel(
        s.a.run_id, s.a.key, command_key="close-missing-sql", reason="current_sql_unavailable"
    )
