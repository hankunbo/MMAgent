"""No-network repair gate: no promotion of UNKNOWN to a retry permission."""

import copy
import json

import pytest

from apps.vectorengine_single_case import write_once
from packages.object_store import checksum
from tests.evaluation_preparation.test_definition_expansion import fixture


def test_original_tuple_budget_is_not_a_request_or_blanket_replay_permission():
    from packages.multimodal_research.search_recovery import pre_wire_journal

    budget = dict(
        schema_version="agent-budget/1.0.0",
        binding=dict(run_id="RUN:TEST", deadline="fixed", max_requests=40),
    )
    ops = [dict(kind="local_search"), dict(kind="search_group")]
    args = dict(run_id="RUN:TEST", deadline="fixed")
    assert pre_wire_journal((budget,), ops, **args)
    assert not pre_wire_journal((budget, dict(phase="reserved")), ops, **args)
    assert not pre_wire_journal((budget,), ops + [dict(kind="wire")], **args)
    assert not pre_wire_journal((budget,), ops, run_id="RUN:OTHER", deadline="fixed")


def prepared(tmp_path, monkeypatch):
    from packages.multimodal_research import implementation

    e = fixture(tmp_path)
    p = e.campaign.output
    before, after = {"module": "old"}, {"module": "new"}
    for name, value in (
        ("entry.json", {"implementation": before}),
        ("protocol.json", e.campaign.protocol),
    ):
        write_once(p / name, value)
    failure = p / "failure.log"
    failure.write_text("original local assembly failure")
    tests = p / "tests.json"
    write_once(tests, dict(direct_exit=0))
    claim = dict(key="search-one", kind="search_group", binding=e.binding)
    parent = dict(key="query-one", kind="local_search", binding=e.binding)
    e.documents.write("definition:operation:0", (parent,))
    e.documents.write("definition:operation:1", (claim,))
    value = dict(
        schema_version="pre-wire-search-repair/1.0.0",
        campaign_id=p.name,
        entry_checksum=checksum((p / "entry.json").read_bytes()),
        protocol_checksum=checksum((p / "protocol.json").read_bytes()),
        failure_log=str(failure),
        failure_checksum=checksum(failure.read_bytes()),
        tests=[dict(path=str(tests), checksum=checksum(tests.read_bytes()))],
        implementation_before=before,
        implementation_after=after,
        run_id=e.scope.run_id,
        operations=[parent, claim],
    )
    write_once(p / "local-search-repair.json", value)
    monkeypatch.setattr(implementation, "freeze", lambda: after)
    return e, before, after


def test_repaired_local_assembly_preserves_old_claim_and_is_idempotent(tmp_path, monkeypatch):
    e, before, _ = prepared(tmp_path, monkeypatch)
    from packages.multimodal_research.implementation import checked_snapshot

    assert checked_snapshot(e.campaign.output, before) == before
    original = copy.deepcopy(e.operations())
    calls = []
    result = e.operation(
        "query-one",
        "local_search",
        lambda: e.operation("search-one", "search_group", lambda: calls.append(1) or ["hit"]),
    )
    assert result == ["hit"] and len(calls) == 1
    assert e.operations()[:2] == original and len(e.operations()) == 4
    assert e.operations()[2]["kind"] == e.operations()[3]["kind"] == "local_repair"
    assert e.operation("search-one", "search_group", lambda: pytest.fail("repeated")) == result


@pytest.mark.parametrize("change", ["wire", "run", "key", "code", "failure", "protocol"])
def test_repair_refuses_other_binding_or_any_wire(tmp_path, monkeypatch, change):
    e, _, after = prepared(tmp_path, monkeypatch)
    p = e.campaign.output
    if change == "wire":
        e.documents.write(
            "definition:operation:2", (dict(key="sent", kind="wire", binding=e.binding),)
        )
    elif change == "run":
        e.scope = e.scope.model_copy(update=dict(run_id="RUN:OTHER"))
    elif change == "key":
        e.binding = dict(e.binding, target="GOAL:OTHER")
    elif change == "code":
        after["unexpected"] = "change"
    else:
        (p / ("failure.log" if change == "failure" else "protocol.json")).write_text("changed")
    with pytest.raises(ValueError, match="local_search_repair"):
        e.operation("search-one", "search_group", lambda: pytest.fail("must not dispatch"))


def test_failed_repair_stays_unknown_not_an_automatic_second_attempt(tmp_path, monkeypatch):
    e, _, _ = prepared(tmp_path, monkeypatch)

    def lost():
        raise TimeoutError("synthetic remote unknown")

    with pytest.raises(TimeoutError):
        e.operation("search-one", "search_group", lost)
    with pytest.raises(ValueError, match="local_search_repair_current_binding"):
        e.operation("search-one", "search_group", lambda: pytest.fail("replayed unknown"))


def test_amendment_never_replaces_or_rehashes_original_freeze(tmp_path, monkeypatch):
    e, before, after = prepared(tmp_path, monkeypatch)
    from packages.multimodal_research.implementation import checked_snapshot

    p = e.campaign.output
    raw = (p / "entry.json").read_bytes()
    assert checked_snapshot(p, before) == before
    assert json.loads(raw)["implementation"] != after
    assert (p / "entry.json").read_bytes() == raw
    with pytest.raises(ValueError, match="repair_implementation_changed"):
        checked_snapshot(p, {"different": "prior"})
