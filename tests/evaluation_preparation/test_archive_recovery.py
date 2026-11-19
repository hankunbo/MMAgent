"""A local assembly failure may reuse confirmed bytes, never an unknown wire."""

import copy

import pytest

from packages.multimodal_research.archive_recovery import confirmed_local_failure


def evidence():
    kinds = ("local_search", "search_group", "wire", "fetch_group", "wire")
    ops = [dict(key=str(i), kind=k) for i, k in enumerate(kinds)]
    results = {
        o["key"]: None if o["kind"] == "local_search" else ({"confirmed": True},) for o in ops
    }
    outcomes = {
        o["key"]: ({"http_status": 200, "response_checksum": "sha256:body"},)
        for o in ops
        if o["kind"] == "wire"
    }
    events = [dict(schema_version="agent-budget/1.0.0")]
    return ops, results, outcomes, events


def test_complete_children_admit_only_local_orchestration():
    assert confirmed_local_failure(*evidence())


@pytest.mark.parametrize(
    "change", [None, "reserved", "cancel", "expired", "unrelated", "failure", "completed"]
)
def test_material_filter_repair_is_one_local_attempt_not_a_new_window(
    tmp_path, monkeypatch, change
):
    import json
    from contextlib import nullcontext
    from datetime import UTC, datetime, timedelta
    from types import SimpleNamespace as NS

    from packages.grounding.models import digest
    from packages.multimodal_research import archive_recovery as a
    from packages.multimodal_research import stage_failure

    op = dict(key="definition:gap:0:query:0", kind="local_search")
    values = {"definition:operation:0": (op,)}
    if change == "completed":
        values[op["key"] + ":result"] = (dict(materials=[]),)

    class Docs:
        run_id = "RUN:LOCAL_MATERIAL"

        def read(self, key):
            return values.get(key)

        def write(self, key, value):
            assert key not in values
            values[key] = value

        def session(self):
            return nullcontext()

    docs = Docs()
    file = "packages/evidence_reuse/definition_sources.py"
    before, after = {file: "old", "unrelated": "same"}, {file: "new", "unrelated": "same"}
    if change == "unrelated":
        after["unrelated"] = "new"
    plan = dict(
        schema_version="multimodal-campaign-renewal/1.58.0",
        new_public_search_allowed=False,
        implementation=before,
        deadline=(
            datetime.now(UTC) + timedelta(minutes=-1 if change == "expired" else 5)
        ).isoformat(),
    )
    failure = dict(
        stage="definitions",
        error_code="local_owner_error",
        failure_kind="no_provider_reservation",
        requests=[],
        new_requests=0,
        scores=None,
        definition_revision_saved=False,
    )
    if change == "failure":
        failure["error_code"] = "provider_unknown"
    c = NS(
        output=tmp_path,
        protocol=plan,
        execution_session=nullcontext,
        sessions=NS(read=lambda _: None),
    )
    (tmp_path / "protocol.json").write_text(json.dumps(plan))
    if change == "cancel":
        (tmp_path / "cancel").touch()
    diagnostic = tmp_path / "diagnostic.json"
    diagnostic.write_text(
        json.dumps(
            dict(
                error=dict(type="Rejected", safe_code="table_unavailable"),
                new_requests=0,
                database_writes=0,
            )
        )
    )
    tests = []
    for name in ("cpu", "static"):
        p = tmp_path / (name + ".json")
        p.write_text('{"direct_exit":0}')
        tests.append(p)
    monkeypatch.setattr(a, "documents", lambda _: docs)
    monkeypatch.setattr(a, "freeze", lambda: after)
    events = [dict(schema_version="agent-budget/1.0.0")]
    if change == "reserved":
        events.append(dict(phase="reserved"))
    monkeypatch.setattr(a, "RuntimeAgentJournal", lambda _: NS(read=lambda: events))
    monkeypatch.setattr(stage_failure, "read_saved_failure", lambda _: failure)
    monkeypatch.setattr(stage_failure, "check_definition_binding", lambda *args: None)
    if change:
        with pytest.raises(ValueError, match="material_filter_repair"):
            a.authorize_material_filter(c, diagnostic=diagnostic, tests=tests)
        assert not (tmp_path / "archive-repair.json").exists()
        return
    saved = (tmp_path / "protocol.json").read_bytes()
    result = a.authorize_material_filter(c, diagnostic=diagnostic, tests=tests)
    amendment = a.read(c)
    assert result["deadline"] == plan["deadline"]
    assert amendment["original_failure_checksum"] == digest(failure)
    assert amendment["operations"] == [op]
    assert (tmp_path / "protocol.json").read_bytes() == saved
    with pytest.raises(ValueError, match="not_active"):
        a.authorize_material_filter(c, diagnostic=diagnostic, tests=tests)
    e = NS(campaign=c, documents=docs, budget=NS(used=0), control=lambda: None)
    a.admit(e, op["key"], [op])
    with pytest.raises(ValueError, match="not_confirmed_local_slot"):
        a.admit(e, op["key"], [op])


def test_material_retry_keeps_failure_and_still_obeys_cancel_deadline(tmp_path):
    import json
    from datetime import UTC, datetime, timedelta
    from types import SimpleNamespace as NS

    from packages.grounding.models import digest
    from packages.llm_provider.provider import ProviderError
    from packages.multimodal_research.definition_expansion import DefinitionExpansion

    old = dict(error_code="local_owner_error")
    path = tmp_path / "stage-failure.json"
    path.write_text(json.dumps(old))
    c = NS(
        output=tmp_path,
        protocol=dict(deadline=(datetime.now(UTC) + timedelta(minutes=1)).isoformat()),
    )
    e = NS(campaign=c)
    with pytest.raises(ProviderError, match="stage_previous_failure_read_only"):
        DefinitionExpansion.fast_control(e)
    c.local_material_repair = dict(original_failure_checksum=digest(old))
    DefinitionExpansion.fast_control(e)
    (tmp_path / "cancel").touch()
    with pytest.raises(ProviderError, match="cancelled"):
        DefinitionExpansion.fast_control(e)
    (tmp_path / "cancel").unlink()
    c.protocol["deadline"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    with pytest.raises(ProviderError, match="deadline"):
        DefinitionExpansion.fast_control(e)
    (tmp_path / "local-repair-failure.json").touch()
    with pytest.raises(ProviderError, match="stage_previous_failure_read_only"):
        DefinitionExpansion.fast_control(e)
    assert json.loads(path.read_bytes()) == old


def test_material_retry_new_failure_has_own_owner_record_and_report(tmp_path, monkeypatch):
    import json
    from contextlib import nullcontext
    from types import SimpleNamespace as NS

    from packages.grounding.models import digest
    from packages.multimodal_research import stage_failure as s

    old = dict(new_requests=0, error_code="local_owner_error")
    (tmp_path / "stage-failure.json").write_text(json.dumps(old))
    records = {}

    class Docs:
        def read(self, key):
            return records.get(key)

        def write(self, key, value):
            assert key not in records
            records[key] = value

        def session(self):
            return nullcontext()

    amendment = dict(original_failure_checksum=digest(old))
    c = NS(output=tmp_path, local_material_repair=amendment, documents=lambda rid: Docs())
    monkeypatch.setattr(
        s,
        "snapshot",
        lambda *a: dict(
            run_id="RUN:TEST", new_requests=1, scores=None, error_code="provider_timeout"
        ),
    )
    result = s.close(c, RuntimeError("timeout"))
    assert result["requests"] == 1 and not result["complete"]
    before = copy.deepcopy(records)
    assert s.close(c, RuntimeError("second call")) == result
    assert records == before
    assert json.loads((tmp_path / "stage-failure.json").read_bytes()) == old
    assert json.loads((tmp_path / "local-repair-failure.json").read_bytes())["repair"] == digest(
        amendment
    )
    assert "provider_timeout" in (tmp_path / "local-repair-report/report.html").read_text()


@pytest.mark.parametrize(
    "change", [None, "reserved", "unknown_search", "cancel", "complete", "expired", "unrelated"]
)
def test_proposal_preflight_repair_preserves_claim_budget_and_deadline(
    tmp_path, monkeypatch, change
):
    import json
    from contextlib import nullcontext
    from datetime import UTC, datetime, timedelta
    from types import SimpleNamespace

    from packages.multimodal_research import archive_recovery as a

    ops = [
        dict(key="definition:query:0", kind="local_search"),
        dict(key="definition:proposal", kind="llm"),
    ]
    values = {f"definition:operation:{i}": (o,) for i, o in enumerate(ops)}
    values["definition:query:0:result"] = ({"materials": [{"exact": "original body"}]},)
    events = [dict(schema_version="agent-budget/1.0.0")]
    if change == "unknown_search":
        values.pop("definition:query:0:result")
    if change == "reserved":
        events.append(dict(phase="reserved", request_id="may_have_been_sent"))

    class Documents:
        run_id = "RUN:PROPOSAL_TEST"

        def read(self, key):
            return values.get(key)

        def write(self, key, value):
            assert key not in values
            values[key] = value

        def session(self):
            return nullcontext()

    d = Documents()
    previous = {"packages/llm_provider/provider.py": "old", "unrelated": "unchanged"}
    current = dict(previous, **{"packages/llm_provider/provider.py": "new"})
    if change == "unrelated":
        current["unrelated"] = "changed"
    protocol = dict(
        schema_version="multimodal-campaign-renewal/1.5.0",
        mode="definitions",
        definition_increment_only=True,
        expansion_policy=dict(prompt_version="definition-draft-2.1.0"),
        implementation=previous,
        deadline=(
            datetime.now(UTC) + timedelta(minutes=-1 if change == "expired" else 10)
        ).isoformat(),
    )
    c = SimpleNamespace(
        output=tmp_path, protocol=protocol, sessions=SimpleNamespace(read=lambda _: None)
    )
    (tmp_path / "lock").touch()
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    original = (tmp_path / "protocol.json").read_bytes()
    if change in {"cancel", "complete"}:
        (tmp_path / ("cancel" if change == "cancel" else "result.json")).touch()
    failure = tmp_path / "failure.log"
    failure.write_text("packages.llm_provider.provider.ProviderError: configuration\n")
    proofs = []
    for name in ("cpu", "static"):
        p = tmp_path / (name + ".json")
        p.write_text(json.dumps(dict(exit_code=0)))
        proofs.append(p)
    monkeypatch.setattr(a, "documents", lambda _: d)
    monkeypatch.setattr(a, "freeze", lambda: current)
    monkeypatch.setattr(a, "RuntimeAgentJournal", lambda _: SimpleNamespace(read=lambda: events))
    if change:
        expected = (
            "proposal_repair_implementation_scope"
            if change == "unrelated"
            else "archive_repair_unknown_or_agent_already_started"
            if change in {"reserved", "unknown_search"}
            else "archive_repair_not_active_local_failure"
        )
        with pytest.raises(ValueError, match=expected):
            a.authorize(c, failure_log=failure, tests=proofs)
        assert not (tmp_path / "archive-repair.json").exists()
    else:
        value = a.authorize(c, failure_log=failure, tests=proofs)
        assert value["schema_version"] == "confirmed-material-local-repair/1.4.0"
        assert value["original_model_requests"] == value["new_wire_requests"] == 0
        assert value["deadline"] == protocol["deadline"] and a.read(c) == value
        e = SimpleNamespace(
            campaign=c, documents=d, budget=SimpleNamespace(used=1), control=lambda: None
        )
        with pytest.raises(ValueError, match="not_confirmed_local_slot"):
            a.admit(e, "definition:proposal", ops)
        e.budget.used = 0
        a.admit(e, "definition:proposal", ops)
        with pytest.raises(ValueError, match="not_confirmed_local_slot"):
            a.admit(e, "definition:proposal", ops)
    assert (tmp_path / "protocol.json").read_bytes() == original


@pytest.mark.parametrize("unknown", [False, True])
def test_table_repair_keeps_deadline_bytes_and_requires_confirmed_wire(
    tmp_path, monkeypatch, unknown
):
    import json
    from contextlib import nullcontext
    from datetime import UTC, datetime, timedelta
    from types import SimpleNamespace

    from packages.multimodal_research import archive_recovery as a

    ops, results, outcomes, events = evidence()
    if unknown:
        outcomes["4"] = None
    values = {f"definition:operation:{i}": (o,) for i, o in enumerate(ops)}
    values.update({k + ":result": v for k, v in results.items()})
    values.update({k + ":outcome": v for k, v in outcomes.items()})

    class Documents:
        run_id = "RUN:TEST"

        def read(self, key):
            return values.get(key)

        def session(self):
            return nullcontext()

        def write(self, key, value):
            assert key not in values
            values[key] = value

    d = Documents()
    protocol = dict(
        mode="definitions",
        definition_increment_only=True,
        implementation={"old": "x"},
        deadline=(datetime.now(UTC) + timedelta(minutes=10)).isoformat(),
    )
    c = SimpleNamespace(
        output=tmp_path, protocol=protocol, sessions=SimpleNamespace(read=lambda _: None)
    )
    (tmp_path / "lock").touch()
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    original = (tmp_path / "protocol.json").read_bytes()
    failure = tmp_path / "failure.log"
    failure.write_text("ValueError: table_chunk_limit\n")
    tests = []
    for name in ("cpu", "static"):
        p = tmp_path / (name + ".json")
        p.write_text(json.dumps(dict(exit_code=0)))
        tests.append(p)
    monkeypatch.setattr(a, "documents", lambda _: d)
    monkeypatch.setattr(a, "freeze", lambda: {"new": "y"})
    monkeypatch.setattr(a, "RuntimeAgentJournal", lambda _: SimpleNamespace(read=lambda: events))
    if unknown:
        with pytest.raises(ValueError, match="unknown_or_agent"):
            a.authorize(c, failure_log=failure, tests=tests)
        assert not (tmp_path / "archive-repair.json").exists()
    else:
        value = a.authorize(c, failure_log=failure, tests=tests)
        assert value["schema_version"] == "confirmed-material-local-repair/1.3.0"
        assert a.read(c) == value
        assert value["new_wire_requests"] == 0 and value["confirmed_wire_requests"] == 2
        assert value["deadline"] == protocol["deadline"]
        with pytest.raises(ValueError, match="index_contract_repair_scope"):
            a.authorize(c, failure_log=failure, tests=tests)
    assert (tmp_path / "protocol.json").read_bytes() == original


@pytest.mark.parametrize(
    "change", ["unknown_wire", "http_error", "unknown_fetch", "llm", "wrong_parent", "no_fetch"]
)
def test_unknown_or_started_model_is_not_a_local_archive_failure(change):
    ops, results, outcomes, events = copy.deepcopy(evidence())
    if change == "unknown_wire":
        outcomes.pop("4")
    elif change == "http_error":
        outcomes["4"][0]["http_status"] = 500
    elif change == "unknown_fetch":
        results["3"] = None
    elif change == "llm":
        events.append(dict(phase="reserved"))
    elif change == "wrong_parent":
        ops[0]["kind"] = "wire"
    else:
        ops[3]["kind"] = "search_group"
    assert not confirmed_local_failure(ops, results, outcomes, events)


def exhausted_evidence():
    kinds = [
        "local_search",
        "search_group",
        "wire",
        "fetch_group",
        "wire",
        "fetch_group",
        "wire",
        "local_search",
        "search_group",
        "wire",
        "fetch_group",
        "fetch_group",
    ]
    ops = [dict(key=str(i), kind=k) for i, k in enumerate(kinds)]
    results = {o["key"]: None if i in {7, 11} else ({"saved": True},) for i, o in enumerate(ops)}
    outcomes = {
        o["key"]: ({"http_status": 200, "response_checksum": "sha256:body"},)
        for o in ops
        if o["kind"] == "wire"
    }
    return ops, results, outcomes, [dict(schema_version="agent-budget/1.0.0")]


def test_only_exhausted_local_capacity_not_unknown_wire():
    from packages.multimodal_research.archive_recovery import exhausted_before_wire

    assert exhausted_before_wire(*exhausted_evidence())
    for key in ("2", "4", "6", "9"):
        ops, results, outcomes, events = exhausted_evidence()
        outcomes[key][0]["http_status"] = None
        assert not exhausted_before_wire(ops, results, outcomes, events)
    ops, results, outcomes, events = exhausted_evidence()
    events.append(dict(phase="reserved"))
    assert not exhausted_before_wire(ops, results, outcomes, events)


def test_http_valueerror_misclassification_reproduced_and_fixed_governor_stops_before_send():
    from datetime import UTC, datetime, timedelta
    from types import SimpleNamespace

    from packages.evidence_reuse.definition_sources import (
        DefinitionAdmissionStopped,
        LiteratureDefinitionConnector,
    )
    from packages.knowledge_connectors import ConnectorError, RecordedTransport, SafeHTTPClient
    from packages.llm_provider.provider import ProviderError
    from tests.knowledge_search.support import config

    def reject(*a):
        raise ProviderError("definition_operation_budget")

    e = SimpleNamespace(
        fast_control=lambda: None,
        reserve_wire=reject,
        campaign=SimpleNamespace(
            protocol=dict(deadline=(datetime.now(UTC) + timedelta(minutes=5)).isoformat())
        ),
    )
    g = LiteratureDefinitionConnector(None, e, {}).governor()
    cfg = config()
    transport = RecordedTransport(())
    client = SafeHTTPClient(cfg, transport, governor=g, resolver=lambda h: ("93.184.216.34",))
    args = dict(operation="fetch", expected_mime=("application/xml",))
    with pytest.raises(DefinitionAdmissionStopped) as caught:
        client.get("https://www.ebi.ac.uk/europepmc/webservices/rest/PMC123/fullTextXML", **args)
    assert caught.value.original.code == "definition_operation_budget" and transport.calls == 0
    # This is the old adapter behavior, not a real TLS/URL failure.
    g.before_send = reject
    with pytest.raises(ConnectorError, match="security"):
        client.get("https://www.ebi.ac.uk/europepmc/webservices/rest/PMC123/fullTextXML", **args)
    assert transport.calls == 0


def test_effective_local_limit_is_separate_from_external_budget(tmp_path):
    from tests.evaluation_preparation.test_definition_expansion import fixture

    e = fixture(tmp_path)
    old = e.policy.max_operations
    assert e.operation_limit == old
    e.campaign.definition_operation_limit = 24
    assert e.operation_limit == 24 and e.policy.max_operations == old


def test_index_local_repair_cannot_hide_a_wire_or_source_reservation():
    from packages.multimodal_research.archive_recovery import confirmed_index_failure

    ops = [dict(kind="local_search", key="query")]
    events = [dict(schema_version="agent-budget/1.0.0")]
    assert confirmed_index_failure(ops, {"query": None}, events)
    assert not confirmed_index_failure(ops + [dict(kind="wire", key="w")], {}, events)
    assert not confirmed_index_failure(ops, {"query": ({"done": True},)}, events)
    assert not confirmed_index_failure(ops, {}, events + [dict(phase="reserved")])
