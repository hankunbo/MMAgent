from dataclasses import replace
from threading import Event

import pytest

from apps.api.auth import LocalIdentities, Principal
from apps.api.offline_review import OfflineReviewBoard
from packages.grounding.models import digest
from tests.agent_provider.support import review_from_wire
from tests.api.support import HTTP, error
from tests.api_expert.support import configured, receipt
from tests.expert_agent.support import expert_reply


def test_http_register_worker_original_review_and_history():
    s = configured()
    h, path = HTTP(s.dependencies), f"/runs/{s.f.binding.run_id}/expert"
    options = h.get(path + "-options")
    assert options.status_code == 200
    assert options.json()["data"]["idempotency_key"] == s.body.idempotency_key
    assert s.body.proposed_action_id in [a["id"] for a in options.json()["data"]["actions"]]
    initial = s.planner.read(s.f.binding.run_id)
    result = h.post(path, s.body.model_dump(mode="json"))
    assert result.status_code == 200, result.text
    assert not result.json()["data"]["executed"]
    assert s.planner.read(s.f.binding.run_id) == initial and s.f.budget.used == 0
    assert h.post(path, s.body.model_dump(mode="json")).json()["data"] == result.json()["data"]
    after = s.task_owner.advance(receipt(s), cancel=Event())
    assert after.phase == "retrieve" and s.f.budget.used == 1
    assert after.graph != initial.graph
    assert s.task_owner.advance(receipt(s), cancel=Event()) == after
    event = s.planner.begin_research_review(s.f.binding.run_id, s.port.command(s.body))
    read = h.get(path + "/" + event.code + ":result")
    assert read.status_code == 200 and read.json()["data"]["status"] == "accepted"
    assert s.f.budget.used == 1
    assert (
        h.post(path, s.body.model_dump(mode="json")).json()["data"]["current_authorization"]
        is False
    )


@pytest.mark.parametrize("kind", ["role", "revision", "graph", "action", "extra", "old"])
def test_http_rejects_before_write_or_provider(kind):
    s = configured()
    body, role = s.body.model_dump(mode="json"), "researcher"
    if kind == "role":
        role = "read_only"
    if kind == "revision":
        body["expected_revision"] += 1
    if kind == "graph":
        body["expected_version"] = digest("wrong")
    if kind == "action":
        body["proposed_action_id"] = "unknown"
    if kind == "extra":
        body["GraphDelta"] = {}
    if kind == "old":
        body = {
            k: body[k] for k in ("api_version", "case_code", "idempotency_key", "expected_version")
        }
        body["contract_version"] = "ExpertAssessment/1.0"
    before = s.planner.read(s.f.binding.run_id)
    response = HTTP(s.dependencies, role).post(f"/runs/{s.f.binding.run_id}/expert", body)
    assert response.status_code in {403, 409, 422, 503}, response.text
    assert s.f.budget.used == 0 and s.planner.read(s.f.binding.run_id) == before


def test_cancel_after_registration_no_model_and_no_new_reservation():
    s = configured()
    s.port.submit(s.body, actor_code="SOURCE:ACTOR")
    before = s.planner.read(s.f.binding.run_id)
    cancel = Event()
    cancel.set()
    state = s.task_owner.advance(receipt(s), cancel=cancel)
    assert state.cancelled and state.phase == "stopped"
    assert s.f.budget.used == 0 and state.reserved == before.reserved


def test_source_withdrawal_during_queue_and_request_conflicts():
    s = configured()
    s.port.submit(s.body, actor_code="SOURCE:ACTOR")
    with pytest.raises(ValueError, match="expert_key_conflict"):
        s.port.submit(s.body.model_copy(update={"abc": True}), actor_code="SOURCE:ACTOR")
    s.f.reader.current = lambda: (_ for _ in ()).throw(ValueError("source_withdrawn"))
    with pytest.raises(ValueError, match="source_withdrawn"):
        s.task_owner.advance(receipt(s), cancel=Event())
    assert s.f.budget.used == 0
    assert s.port.submit(s.body, actor_code="SOURCE:ACTOR")["current_authorization"] is False


def test_review_checkpoint_lost_reply_recovery():
    s = configured()
    s.port.submit(s.body, actor_code="SOURCE:ACTOR")
    original = s.service.context.checkpoint

    def lost(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("lost_checkpoint_reply")

    s.service.context.checkpoint = lost
    with pytest.raises(OSError):
        s.task_owner.advance(receipt(s), cancel=Event())
    s.service.context.checkpoint = original
    after = s.task_owner.advance(receipt(s), cancel=Event())
    assert after.phase == "retrieve" and s.f.budget.used == 1


def test_policy_changes_on_pending_save_are_not_new_authorization():
    s = configured()
    s.port.submit(s.body, actor_code="SOURCE:ACTOR")
    original = s.planner.begin_research_review

    def changed(*args, **kwargs):
        event = original(*args, **kwargs)
        s.service.context.policy = s.service.context.policy.model_copy(
            update={"comparison_protocol": digest("changed")}
        )
        return event

    s.planner.begin_research_review = changed
    with pytest.raises(ValueError, match="expert_version_changed"):
        s.task_owner.advance(receipt(s), cancel=Event())
    assert s.f.budget.used == 0
    assert s.planner.read(s.f.binding.run_id).pending is not None


@pytest.mark.parametrize("decision", ["accept", "reject", "safe_stop"])
def test_installed_required_expert_uses_original_limited_decision(decision):
    s = configured(replies=[review_from_wire, expert_reply(decision)], facts={"required": True})
    s.port.submit(s.body, actor_code="SOURCE:ACTOR")
    state = s.task_owner.advance(receipt(s), cancel=Event())
    event = s.planner.begin_research_review(s.f.binding.run_id, s.port.command(s.body))
    result = s.service.store.read(event.code + ":result")[0]
    assert result["status"] == ("accepted" if decision == "accept" else decision)
    assert s.f.budget.used == 2 and state.started == 0


def test_required_expert_failure_stops_not_human_wait():
    from packages.llm_provider.transport import Reply

    s = configured(replies=[review_from_wire, Reply(error="refusal")], facts={"required": True})
    s.port.submit(s.body, actor_code="SOURCE:ACTOR")
    state = s.task_owner.advance(receipt(s), cancel=Event())
    assert state.phase == "stopped" and s.f.budget.used == 2


def board(s, reviewers=("SOURCE:R1", "SOURCE:R2")):
    subject = s.f.binding.graph
    port = OfflineReviewBoard(
        run_id=s.f.binding.run_id,
        case_code=s.f.binding.case_code,
        documents=s.f.documents,
        subject=subject,
        read_subject=lambda: subject,
        reviewers=reviewers,
        arbiter="SOURCE:ARBITER",
    )
    identities = LocalIdentities(
        {
            actor: Principal(
                actor,
                "offline_reviewer",
                frozenset([s.f.binding.case_code]),
                frozenset(),
                synthetic=True,
            )
            for actor in (*reviewers, "SOURCE:ARBITER")
        }
    )
    deps = replace(
        s.dependencies, identities=identities, offline_boards={(s.f.binding.run_id, "board"): port}
    )
    body = dict(
        api_version="research-http/1.0.0",
        case_code=s.f.binding.case_code,
        idempotency_key="vote",
        expected_version=subject.checksum,
        decision="approve",
    )
    return port, deps, body


@pytest.mark.parametrize("reviewers", [("SOURCE:R1",), ("SOURCE:R1", "SOURCE:R2")])
def test_offline_single_double_disagreement_has_no_run_effect(reviewers):
    s = configured()
    port, deps, body = board(s, reviewers)
    path = f"/runs/{s.f.binding.run_id}/offline-reviews/board"
    before = s.planner.read(s.f.binding.run_id)
    first = HTTP(deps, "SOURCE:R1").post(path, body)
    assert first.status_code == 200, first.text
    if len(reviewers) == 2:
        error(HTTP(deps, "SOURCE:ARBITER").post(path, body), "STATE_CONFLICT", 409)
        second = HTTP(deps, "SOURCE:R2").post(path, {**body, "decision": "reject"})
        assert second.json()["data"]["status"] == "adjudication_required"
        final = HTTP(deps, "SOURCE:ARBITER").post(path, body)
        assert final.json()["data"]["status"] == "adjudicated"
    else:
        assert first.json()["data"]["status"] == "reviewed"
    assert not port.read()["release_authorization"]
    assert s.planner.read(s.f.binding.run_id) == before and s.f.budget.used == 0
    error(
        HTTP(deps, "SOURCE:R1").post(path, {**body, "decision": "reject"}),
        "IDEMPOTENCY_CONFLICT",
        409,
    )


def test_offline_source_changed_no_vote_and_history_kept():
    s = configured()
    port, deps, body = board(s)
    path = f"/runs/{s.f.binding.run_id}/offline-reviews/board"
    assert HTTP(deps, "SOURCE:R1").post(path, body).status_code == 200
    port.read_subject = lambda: None
    error(HTTP(deps, "SOURCE:R2").post(path, body), "VERSION_CONFLICT", 409)
    assert len(port.read()["votes"]) == 1
