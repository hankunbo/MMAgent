import json
from threading import Event

import pytest

from packages.counterfactual.review import review
from packages.grounding.models import digest
from tests.counterfactual.support import candidate, fixture


def reply(action="accept", action_id="one", hook=None):
    def result(body):
        payload = json.loads(body["messages"][1]["content"][0]["text"])
        if hook:
            hook()
        return dict(
            scope_checksum=digest(payload["material"]["scope"]),
            action=action,
            action_id=action_id,
            evidence=["SOURCE:ONE"],
            reason="Synthetic bounded advice",
        )

    return result


@pytest.mark.parametrize(
    "action,action_id,expected",
    [
        ("accept", "one", "reviewed_candidate"),
        ("alternative", "two", "reviewed_candidate"),
        ("alternative", "invented", "safe_stop"),
        ("safe_stop", None, "reviewed_candidate"),
    ],
)
def test_expert_only_declared_candidates_and_review_separate(action, action_id, expected):
    x = fixture(policy_updates=dict(expert_required=True))
    c = candidate()
    x.service.declare((c, candidate(id="two")))
    x.f.provider.transport.replies.append(reply(action, action_id))
    r = review(x.service, c, ports=x.f.ports)
    assert r["status"] == expected
    assert x.f.budget.used == 2
    assert review(x.service, c, ports=x.f.ports) == r and x.f.budget.used == 2
    assert x.service.read(candidate_id="one")["identity"] == "hypothetical_candidate"
    assert x.f.documents.read("simulation:validation:one")[0]["required_review"]
    assert x.f.budget.summary().api_requests_observed == 0


@pytest.mark.parametrize("change", ["cancel", "withdraw", "deadline", "branches"])
def test_expert_return_cannot_authorize_past_current_control(change):
    x = fixture(policy_updates=dict(expert_required=True))
    c = candidate()
    x.service.declare((c,))
    signal = Event()

    def hook():
        if change == "cancel":
            signal.set()
        elif change == "withdraw":
            x.f.reader.withdrawn = True
        elif change == "deadline":
            x.h.planner.clock = lambda: x.f.binding.deadline
        else:
            state = x.h.planner.read(x.f.binding.run_id)
            x.h.planner.runtime.state = state.model_copy(
                update=dict(branches=state.binding.profile.max_branches)
            )

    x.f.provider.transport.replies.append(reply(hook=hook))
    r = review(x.service, c, ports=x.f.ports, cancel=signal)
    assert r["status"] == "safe_stop" and x.f.budget.used == 2
    assert x.service.read(candidate_id="one", historical=True)
    assert review(x.service, c, ports=x.f.ports, cancel=signal)["status"] == "stopped"
    assert x.f.budget.used == 2


def test_unknown_review_commit_never_reissues_request(monkeypatch):
    x = fixture(policy_updates=dict(expert_required=True))
    c = candidate()
    x.service.declare((c,))
    x.f.provider.transport.replies.append(reply())
    write = x.f.documents.write
    with monkeypatch.context() as m:

        def lost(key, values):
            if key == "simulation:review:one":
                raise OSError("review_io_unknown")
            return write(key, values)

        m.setattr(x.f.documents, "write", lost)
        with pytest.raises(OSError):
            review(x.service, c, ports=x.f.ports)
    result = review(x.service, c, ports=x.f.ports)
    assert result["reason"] == "review_confirmation_unknown" and x.f.budget.used == 2


def test_required_expert_budget_exhaustion_preserves_completed_fact_report():
    x = fixture(policy_updates=dict(expert_required=True))
    c = candidate()
    x.service.declare((c,))
    while x.f.budget.used < x.f.budget.max_requests:
        x.f.budget.reserve(
            x.parent.material.scope, x.f.binding.deadline, digest(x.f.provider.config)
        )
    before = x.f.budget.used
    result = review(x.service, c, ports=x.f.ports)
    assert result["status"] == "safe_stop"
    assert x.f.budget.used == before
    assert x.service.read(candidate_id="one")["clinical_acceptance"] is False


def test_cannot_replace_original_request_budget_with_fresh_same_run_journal():
    from tests.agent_provider.support import budget_for

    x = fixture()
    c = candidate()
    x.service.declare((c,))
    x.f.ports.budget = budget_for(x.f.binding)
    with pytest.raises(ValueError, match="budget_scope"):
        review(x.service, c, ports=x.f.ports)
    assert x.f.budget.used == 1
