import copy
from threading import Event

import pytest

from agents.expert_agent.reporting import read_research_report, render_research_html
from packages.grounding.models import digest
from packages.llm_provider.transport import Reply
from tests.agent_provider.support import review_from_wire
from tests.expert_agent.support import expert_reply, run, setup


def test_ordinary_step_skips_expert_and_requests():
    f = setup()
    result = run(f)
    assert result["status"] == "baseline"
    assert f.budget.used == 0
    assert not f.context.installs
    assert run(f) == result
    assert f.budget.used == 0


@pytest.mark.parametrize(
    "action,status", [("accept", "accepted"), ("reject", "reject"), ("safe_stop", "safe_stop")]
)
def test_review_advice_domain_acceptance_and_html(action, status):
    f = setup([review_from_wire, expert_reply(action)], facts={"required": True})
    parent = f.reader.state
    result = run(f, role="review")
    assert result["status"] == status
    assert f.budget.used == 2
    assert f.budget.summary().api_requests_observed == 0
    assert len(f.context.installs) == (status == "accepted")
    if status == "accepted":
        report = read_research_report(f.acceptance, key=result["acceptance_key"], parent=parent)
        html = render_research_html(report)
        assert "case-research-scoring/6.1.0" in html
        assert result["graph"]["checksum"] in html
        assert "SOURCE:ONE" in html
        assert f.reader.state.material.scope.graph.code == result["graph"]["code"]
        assert f.documents.consumed(parent.material.scope.graph)
        assert not report["graph"]["clinical_patient_graph"]
    assert run(f, role="review") == result
    assert f.budget.used == 2


@pytest.mark.parametrize("bad", ["alternative", "graph_delta", "scope", "accept_other"])
def test_illegal_expert_never_accepts(bad):
    def reply(body):
        value = expert_reply()(body)
        if bad == "alternative":
            value.update(action="alternative", action_id="action:invented")
        elif bad == "graph_delta":
            value["GraphDelta"] = {"nodes": []}
        elif bad == "scope":
            value["scope_checksum"] = digest("other")
        else:
            value["action_id"] = "action:invented"
        return value

    f = setup([review_from_wire, reply], facts={"required": True})
    assert run(f, role="review")["status"] == "safe_stop"
    assert f.budget.used == 2
    assert not f.context.installs


@pytest.mark.parametrize("required", [True, False])
@pytest.mark.parametrize("error", ["refusal", "no_result", "authentication", "schema"])
def test_required_failure_stops_optional_returns_baseline(required, error):
    f = setup([Reply(error=error)], facts={"uncertain": True, "required": required})
    assert run(f)["status"] == ("safe_stop" if required else "baseline")
    assert f.budget.used == 1
    assert run(f)["status"] == ("safe_stop" if required else "baseline")
    assert f.budget.used == 1


@pytest.mark.parametrize(
    "change", ["withdraw", "graph", "policy", "actions", "cancel", "deadline", "budget"]
)
def test_return_time_revalidation(change):
    signal = Event()
    holder = {}

    def mutate(_):
        f = holder["f"]
        if change == "withdraw":
            f.reader.withdrawn = True
        elif change == "graph":
            f.reader.state = f.reader.state.model_copy(update={"nodes": ("GOAL:CHANGED",)})
        elif change == "policy":
            f.context.frame = f.frame.model_copy(
                update={"policy": f.frame.policy.model_copy(update={"max_no_progress": 3})}
            )
        elif change == "actions":
            f.context.frame = f.frame.model_copy(
                update={"facts": f.frame.facts.model_copy(update={"conflict": True})}
            )
        elif change == "cancel":
            signal.set()
        elif change == "deadline":
            f.budget.clock = lambda: f.binding.deadline
        else:
            f.budget.max_requests = 0

    f = setup([expert_reply(mutate=mutate)], facts={"uncertain": True})
    holder["f"] = f
    assert run(f, cancel=signal)["status"] == "safe_stop"
    assert not f.context.installs
    assert f.budget.used == 1


def test_legal_alternative_does_not_accept_original_candidate():
    f = setup(facts={"conflict": True})
    other = next(
        a for a in f.frame.actions.actions if a.kind == "observe" and a.id != f.proposed.id
    )
    f.provider.transport.replies = [expert_reply("alternative", other.id)]
    result = run(f)
    assert result["status"] == "alternative"
    assert result["selected_action_id"] == other.id
    assert not f.context.installs


@pytest.mark.parametrize("failure", ["install", "checkpoint"])
def test_successor_confirmation_loss_never_repeats_request_or_acceptance(failure):
    f = setup([review_from_wire])
    setattr(f.context, "fail_" + failure, True)
    with pytest.raises(OSError):
        run(f, role="review")
    count = len(f.documents.audit)
    result = run(f, role="review")
    assert result["status"] == "accepted"
    assert f.budget.used == 1
    assert len(f.context.installs) == 1
    assert len([k for k in f.documents.saved if k.startswith("accepted:")]) == 1
    assert len(f.documents.audit) >= count


def test_frozen_abc_saved_before_single_acceptance():
    f = setup([review_from_wire] * 3)
    result = run(f, role="review", abc=True)
    assert result["status"] == "accepted"
    assert f.budget.used == 3
    comparison = f.service.store.read(f.frame.event.code + ":comparison")[0]
    assert {r["round_id"] for r in comparison["rounds"]} == {"A", "B", "C"}
    assert len({r["material_checksum"] for r in comparison["rounds"]}) == 1
    assert comparison["real_ABC_executed"] is False
    assert len([k for k in f.documents.saved if k.startswith("accepted:")]) == 1
    assert all(r.scope == f.frame.scope for r in f.budget.records())
    assert run(f, role="review", abc=True) == result
    assert f.budget.used == 3


def test_no_progress_stops_without_request():
    f = setup(facts={"no_progress": 2})
    assert run(f)["reason"] == "no_new_information"
    assert f.budget.used == 0


def test_key_conflict_and_unconfirmed_call_do_not_restart():
    f = setup([review_from_wire])
    key = f.frame.event.code
    f.service.store.save(
        key + ":A:call",
        dict(
            schema_version="research-call-claim/1.0.0",
            role="review",
            round="A",
            frame_checksum=digest(f.frame),
        ),
    )
    assert run(f, role="review")["status"] == "safe_stop"
    assert f.budget.used == 0
    with pytest.raises(ValueError, match="conflict"):
        run(f, role="review", abc=True)
    assert any(k.startswith("review:conflict:") for k in f.documents.audit)


def test_html_tampering_is_rejected():
    f = setup([review_from_wire])
    parent = f.reader.state
    result = run(f, role="review")
    report = read_research_report(f.acceptance, key=result["acceptance_key"], parent=parent)
    bad = copy.deepcopy(report)
    bad["scores"]["root"]["coverage"]["slo_image_research"]["acquisition"] = 0
    with pytest.raises(ValueError, match="binding"):
        render_research_html(bad)
