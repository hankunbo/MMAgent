"""New integration adapter on the original Planner; small failure-injection ports only."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from agents.planner.service import Planner
from packages.grounding.models import canonical
from packages.memory_review.planner import MemoryRankedActions
from packages.planner_control.models import Phase, SelectionPending
from packages.retrieval.memory.service import Retrieval
from tests.memory_review.support import QueryOwner
from tests.planner.support import command
from tests.planner.test_actions import setup_ranked


class Documents:
    def __init__(self, run_id):
        self.run_id = run_id
        self.values = {}
        self.writes = 0

    def session(self):
        return nullcontext()

    def read(self, key):
        return self.values.get(key)

    def write(self, key, values):
        values = tuple(json.loads(canonical(v)) for v in values)
        if key in self.values:
            assert self.values[key] == values
        else:
            self.values[key] = values
            self.writes += 1


class RetrievalPort:
    optional = True

    def __init__(self, q):
        self.q = q
        self.calls = 0
        self.checks = 0
        self.invalid = False
        self.change = None
        self.groups = {"success": [], "counterfactual": [], "recovery": []}
        self.status = "ok"

    def retrieve(self, rid):
        self.calls += 1
        if self.change:
            self.change()
        return Retrieval(self.q, self.groups, (), self.status)

    def recheck(self, q, hits):
        self.checks += 1
        if self.invalid:
            raise ValueError("memory_qualification_revoked")


def setup(*, with_memory=True):
    p, r, o, s = setup_ranked()
    docs = Documents(r.state.binding.run_id)
    q = QueryOwner(r.state.binding.run_id, r.state.binding.case_code).value
    retrieval = RetrievalPort(q)
    if with_memory:
        retrieval.groups["recovery"] = [
            dict(cases=["CASE:PAST1", "CASE:PAST2"], branches=["B", "ILLEGAL"])
        ]
    p.ranking = MemoryRankedActions(s, retrieval=retrieval, documents=docs)
    return SimpleNamespace(p=p, r=r, o=o, s=s, docs=docs, retrieval=retrieval)


def test_installed_order_changes_only_legal_actions_and_readback():
    x = setup()
    cmd = command(x.p, "select")
    event = x.p.step(x.r.state.binding.run_id, cmd)
    assert event.outcome.decision.selected.branch == "B"
    assert all(a.branch != "ILLEGAL" for a in event.outcome.decision.action_set.actions)
    receipt = x.p.ranking.read_influence(x.r.state, event.outcome.decision)
    assert receipt["baseline_order"] != receipt["final_order"]
    assert x.r.state.memory == ()
    restored = Planner(runtime=x.r, owner=x.o, ranking=x.p.ranking)
    assert restored.step(x.r.state.binding.run_id, cmd) == event
    assert x.docs.writes == 1 and x.retrieval.calls == 1 and x.r.state.reserved == 0


@pytest.mark.parametrize("technical", [False, True])
def test_empty_and_technical_status_fall_back_without_wait(technical):
    x = setup(with_memory=False)
    if technical:
        x.retrieval.status = "technical_failure"
    event = x.p.step(x.r.state.binding.run_id, command(x.p, "select"))
    assert event.outcome.decision.selected.branch == "A" and x.r.state.phase == Phase.PREPARE
    receipt = x.p.ranking.read_influence(x.r.state, event.outcome.decision)
    assert receipt["retrieval"]["status"] == ("technical_failure" if technical else "ok")
    assert receipt["baseline_order"] == receipt["final_order"]


def test_original_pending_recovery_no_duplicate_retrieve_write_or_reservation(monkeypatch):
    x = setup()
    cmd = command(x.p, "select")
    save = x.r.save

    def fail_after_pending(old, new):
        result = save(old, new)
        if isinstance(new.pending, SelectionPending):
            raise ConnectionError("synthetic_lost_pending_reply")
        return result

    with monkeypatch.context() as m:
        m.setattr(x.r, "save", fail_after_pending)
        with pytest.raises(ConnectionError):
            x.p.step(x.r.state.binding.run_id, cmd)
    assert isinstance(x.r.state.pending, SelectionPending)
    event = x.p.step(x.r.state.binding.run_id, cmd)
    assert event.outcome.decision.selected.branch == "B" and x.docs.writes == x.retrieval.calls == 1
    assert x.r.state.reserved == 0


def test_query_return_invalidation_confirmed_before_selection():
    x = setup()
    original = x.docs.write

    def withdraw(key, values):
        result = original(key, values)
        x.retrieval.invalid = True
        return result

    x.docs.write = withdraw
    event = x.p.step(x.r.state.binding.run_id, command(x.p, "select"))
    assert event.after == Phase.STOPPED and event.outcome.status == "rejected"
    assert x.r.state.selected is None and x.r.state.reserved == 0


def test_current_action_source_changes_during_query_refuses_stale_set():
    x = setup()
    x.retrieval.change = lambda: x.s.blocked.add("B")
    event = x.p.step(x.r.state.binding.run_id, command(x.p, "select"))
    assert event.outcome.reason == "stale_action_set" and x.docs.writes == 0
    assert x.r.state.selected is None


def test_prepare_rechecks_publication_and_never_launches():
    x = setup()
    x.p.step(x.r.state.binding.run_id, command(x.p, "select"))
    x.retrieval.invalid = True
    event = x.p.step(x.r.state.binding.run_id, command(x.p, "prepare"))
    assert event.after == Phase.STOPPED and event.outcome.status == "rejected"
    assert x.r.state.reserved == 0 and "execute" not in x.o.calls


def test_receipt_survives_failure_before_selection_pending_save(monkeypatch):
    x = setup()
    cmd = command(x.p, "select")
    save = x.r.save

    def fail_before_pending(old, new):
        if isinstance(new.pending, SelectionPending):
            raise ConnectionError("synthetic_pending_rollback")
        return save(old, new)

    with monkeypatch.context() as m:
        m.setattr(x.r, "save", fail_before_pending)
        with pytest.raises(ConnectionError):
            x.p.step(x.r.state.binding.run_id, cmd)
    assert not isinstance(x.r.state.pending, SelectionPending)
    assert x.docs.writes == x.retrieval.calls == 1
    assert x.p.step(x.r.state.binding.run_id, cmd).outcome.decision.selected.branch == "B"
    assert x.docs.writes == x.retrieval.calls == 1 and x.r.state.reserved == 0
