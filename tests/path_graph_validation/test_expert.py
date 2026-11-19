"""09B facts -> original pending 08 Expert; no candidate acceptance or tool dispatch."""

from threading import Event

import pytest

from agents.expert_agent.models import ReviewFacts
from packages.grounding.models import digest
from packages.llm_provider.transport import Reply
from packages.path_graph_validation import PathGraphValidation, ResearchPathReader
from tests.agent_provider.support import review_from_wire
from tests.expert_agent.support import expert_reply, setup
from tests.expert_agent.test_review_boundaries import integrated


def ready(replies=(), facts=None):
    f = setup([review_from_wire, *replies])
    parent = f.reader.state
    h = integrated(f)
    first = h.service.run(**h.args)
    reader = ResearchPathReader(
        acceptance=h.service.acceptance, parent=parent, key=first["acceptance_key"]
    )
    validation = PathGraphValidation(reader=reader, documents=f.documents)
    report = validation.inspect(key="summary")
    for name in ("retrieve", "instantiate", "identify", "select", "prepare", "execute", "verify"):
        h.planner.step(f.binding.run_id, h.command(name))
    event = h.planner.begin_research_review(f.binding.run_id, h.command("review"))
    h.service.context.facts = lambda _: ReviewFacts(**(facts or {}))
    frame = h.service.context.read(event)
    proposed = next(a for a in frame.actions.actions if a.kind == "observe")
    args = dict(key="summary", review=h.service, event=event, proposed_action_id=proposed.id)
    return f, h, validation, report, args


def test_ordinary_path_uses_original_baseline_without_new_request():
    f, h, service, report, args = ready()
    state = h.planner.read(f.binding.run_id)
    r = service.review(**args)
    assert r["result"]["status"] == "baseline"
    assert r["summary_checksum"] == report["checksum"]
    assert r["expert_decision"]["triggers"] == []
    assert f.budget.used == 1
    after = h.planner.read(f.binding.run_id)
    assert (
        after.graph == state.graph
        and after.reserved == state.reserved
        and after.started == state.started
    )
    assert service.review(**args) == r


@pytest.mark.parametrize("mode", ["illegal", "refusal", "budget", "conflict", "withdraw", "cancel"])
def test_required_expert_and_return_revalidation(mode):
    holder, signal = {}, Event()

    def response(body):
        value = expert_reply()(body)
        if mode == "illegal":
            value.update(action="alternative", action_id="invented")
        elif mode == "withdraw":
            holder["f"].reader.withdrawn = True
        elif mode == "cancel":
            signal.set()
        return value

    replies = [Reply(error="refusal")] if mode == "refusal" else [response]
    f, h, service, report, args = ready(replies, {"required": True, "conflict": mode == "conflict"})
    holder["f"] = f
    if mode == "budget":
        f.budget.max_requests = f.budget.used
    graph = h.planner.read(f.binding.run_id).graph
    result = service.review(**args, cancel=signal)
    if mode == "conflict":
        assert result["result"]["status"] == "accept"
        assert result["expert_decision"]["required"]
    else:
        assert result["result"]["status"] == "safe_stop"
        assert result["result"]["continuation"] == "stop"
    assert result["summary_checksum"] == report["checksum"]
    assert h.planner.read(f.binding.run_id).graph == graph
    assert result["result"]["domain_status"] == "not_accepted"
    used = f.budget.used
    signal.clear()
    assert service.review(**args) == result
    assert f.budget.used == used
    assert service.read(key="summary", historical=True) == report


def test_expert_confirm_loss_recovers_original_advice_once(monkeypatch):
    f, h, service, _, args = ready([expert_reply()], {"required": True})
    write, lost = f.documents.write, []

    def intercepted(key, values):
        write(key, values)
        if key.endswith(":decision") and not lost:
            lost.append(1)
            raise OSError("synthetic_decision_confirmation_lost")

    with monkeypatch.context() as m:
        m.setattr(f.documents, "write", intercepted)
        with pytest.raises(OSError, match="confirmation_lost"):
            service.review(**args)
    result = service.review(**args)
    assert result["result"]["status"] == "accept"
    assert f.budget.used == 2
    before = digest(list(f.documents.saved.values()))
    assert service.review(**args) == result
    assert digest(list(f.documents.saved.values())) == before


def test_advice_report_html_matches_json_and_recovers_commit_loss(monkeypatch):
    from packages.path_graph_validation import render_review_html

    f, h, service, summary, args = ready([expert_reply()], {"required": True})
    original, lost = f.documents.write, []

    def write(key, values):
        original(key, values)
        if key.startswith("path-graph-review:") and not lost:
            lost.append(1)
            raise OSError("synthetic_report_confirmation_lost")

    with monkeypatch.context() as m:
        m.setattr(f.documents, "write", write)
        with pytest.raises(OSError, match="confirmation_lost"):
            service.review(**args)
    report = service.review(**args)
    saved = f.documents.read("path-graph-review:" + args["event"].code)
    assert saved[0] == report and saved[1]["html"] == render_review_html(report)
    assert report["summary"] == summary
    assert report["expert_decision"]["reason"] == "expert_advice"
    assert f.budget.used == 2
    report["expert_decision"]["reason"] = "forged"
    report["checksum"] = digest({k: v for k, v in report.items() if k != "checksum"})
    with pytest.raises(ValueError, match="expert_report_binding"):
        render_review_html(report)
