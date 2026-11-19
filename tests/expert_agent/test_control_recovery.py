"""Locked controls and interruption recovery; synthetic owners, no live dispatch."""

from threading import Event

import pytest

from agents.expert_agent.models import read_result
from packages.evidence_details.current_rejection import CurrentResearchRejected
from packages.grounding.models import digest
from packages.llm_provider.transport import Reply
from tests.agent_provider.support import review_from_wire
from tests.expert_agent.support import expert_reply, run, setup
from tests.expert_agent.test_review_boundaries import integrated


def locked(**profile):
    base = setup().binding
    return base.model_copy(update={"profile": base.profile.model_copy(update=profile)})


@pytest.mark.parametrize("reply", [None, "refusal", "accept", "safe_stop"])
@pytest.mark.parametrize("candidate", [True, False])
def test_locked_required_default_facts_for_candidate_and_ordinary_action(reply, candidate):
    replies = [review_from_wire] if candidate else []
    if reply:
        replies.append(Reply(error="refusal") if reply == "refusal" else expert_reply(reply))
    f = setup(replies, binding=locked(expert_required=True))
    h = integrated(f)
    if not candidate:
        action = next(
            a
            for a in h.service.context.read(h.args["event"]).actions.actions
            if a.kind == "observe"
        )
        h.args.update(role=None, proposed_action_id=action.id)
    result = h.service.run(**h.args)
    assert [r.role for r in f.budget.records()] == (["review"] if candidate else []) + ["expert"]
    assert (
        result["status"] == ("accepted" if candidate else "accept")
        if reply == "accept"
        else result["status"] == "safe_stop"
    )
    receipt = h.service.acceptance.read(key=h.args["event"].code + ":domain")
    assert bool(receipt) == (candidate and reply == "accept")


@pytest.mark.parametrize(
    "point",
    [
        "candidate",
        "expert",
        "submission",
        "accept",
        "install",
        "result",
        "accept_lost",
        "install_lost",
        "result_lost",
        "checkpoint_lost",
    ],
)
def test_cancel_all_boundaries_persists_without_repeating_work(monkeypatch, point):
    signal = Event()

    def candidate(body):
        if point == "candidate":
            signal.set()
        return review_from_wire(body)

    def expert(body):
        signal.set()
        return expert_reply()(body)

    f = setup([candidate, expert], binding=locked(expert_required=point == "expert"))
    h = integrated(f)
    key = h.args["event"].code
    snapshots, accept_calls = [], []
    accept = h.service.acceptance.accept

    def counted_accept(**kw):
        accept_calls.append(kw)
        value = accept(**kw)
        snapshots.append(h.service.acceptance.read(key=key + ":domain"))
        return value

    monkeypatch.setattr(h.service.acceptance, "accept", counted_accept)
    if point not in {"candidate", "expert"}:
        target, name = {
            "submission": (h.service.acceptance, "submit"),
            "accept": (h.service.acceptance, "accept"),
            "install": (h.service.context, "installer"),
            "result": (h.service.store, "save"),
            "checkpoint": (h.service.context, "checkpoint"),
        }[point.removesuffix("_lost")]
        original = getattr(target, name)

        def interrupt(*args, **kwargs):
            value = original(*args, **kwargs)
            if name == "save" and not args[0].endswith(":result"):
                return value
            signal.set()
            if point.endswith("_lost"):
                raise OSError("confirmation_lost")
            return value

        monkeypatch.setattr(target, name, interrupt)
    if point.endswith("_lost"):
        with pytest.raises(OSError, match="confirmation_lost"):
            h.service.run(**h.args, cancel=signal)
        monkeypatch.setattr(target, name, original)
    else:
        h.service.run(**h.args, cancel=signal)
    state = h.planner.read(f.binding.run_id)
    assert state.cancelled
    assert any(s.cancelled for s in h.planner.runtime.audit)
    signal.clear()  # Recovery must use durable Control, not this process's Event.
    requests = f.budget.used
    result = h.service.run(**h.args)
    state = h.planner.replay(f.binding.run_id)
    assert state.cancelled and state.phase == "stopped" and state.pending is None
    assert h.service.run(**h.args) == result
    assert f.budget.used == requests == (2 if point == "expert" else 1)
    assert state.started == 0 and state.reserved == 1
    assert len(accept_calls) == (0 if point in {"candidate", "expert", "submission"} else 1)
    if snapshots:
        assert h.service.acceptance.read(key=key + ":domain") == snapshots[0]
        report = h.service.store.read(key + ":report")[0]
        assert report["receipt"] == snapshots[0][-1]
        assert h.planner.report(f.binding.run_id).reason == "cancelled"
        assert state.events[-1].outcome.refs
        if point == "install_lost":
            assert result["successor_status"] == "unconfirmed"
        elif point == "accept_lost" or point == "accept":
            assert result["successor_status"] == "not_installed"
        else:
            assert result["successor_status"] == "installed"


@pytest.mark.parametrize("role", ["candidate", "expert"])
@pytest.mark.parametrize("change", ["cancel", "withdraw", "deadline"])
def test_every_transport_retry_rechecks_current_control(monkeypatch, role, change):
    signal = Event()
    f = setup(
        [review_from_wire] if role == "expert" else [],
        binding=locked(expert_required=role == "expert"),
    )
    h = integrated(f)
    send = f.provider.transport.send
    calls = []

    def transport(*args, **kwargs):
        calls.append(1)
        if role == "expert" and len(calls) == 1:
            return send(*args, **kwargs)
        if change == "cancel":
            signal.set()
        elif change == "withdraw":
            f.reader.withdrawn = True
        else:
            h.planner.clock = lambda: f.binding.deadline
        return Reply(error="timeout")

    monkeypatch.setattr(f.provider.transport, "send", transport)
    result = h.service.run(**h.args, cancel=signal)
    assert result["status"] == "safe_stop"
    assert len(calls) == f.budget.used == (2 if role == "expert" else 1)
    assert h.planner.read(f.binding.run_id).phase == "stopped"


@pytest.mark.parametrize("point", ["accept", "successor", "install", "result_lost"])
@pytest.mark.parametrize("change", ["withdraw", "reset", "policy"])
def test_confirmed_acceptance_current_rejection_is_terminal(monkeypatch, point, change):
    f = setup([review_from_wire])
    h = integrated(f)
    key = h.args["event"].code
    target, name = {
        "accept": (h.service.acceptance, "accept"),
        "successor": (h.service.acceptance, "successor_material"),
        "install": (h.service.context, "installer"),
        "result_lost": (h.service.store, "save"),
    }[point]
    original = getattr(target, name)
    history = []

    def changed(*args, **kwargs):
        value = original(*args, **kwargs)
        if name == "save" and not args[0].endswith(":result"):
            return value
        history.append(h.service.acceptance.read(key=key + ":domain"))
        if change == "withdraw":
            f.reader.withdrawn = True
        elif change == "reset":

            def reset():
                raise CurrentResearchRejected("graph_reset")

            monkeypatch.setattr(f.reader, "current", reset)
        else:
            state = f.reader.state
            f.reader.state = state.model_copy(update={"config_checksum": digest("changed_policy")})
        if point == "result_lost":
            raise OSError("result_lost")
        return value

    monkeypatch.setattr(target, name, changed)
    if point == "result_lost":
        with pytest.raises(OSError, match="result_lost"):
            h.service.run(**h.args)
        monkeypatch.setattr(target, name, original)
        # Policy/config must also be checked at final confirmation, not only graph.
    result = h.service.run(**h.args)
    state = h.planner.replay(f.binding.run_id)
    assert state.phase == "stopped" and state.pending is None
    assert h.service.run(**h.args) == result and f.budget.used == 1
    assert h.service.acceptance.read(key=key + ":domain") == history[0]
    assert h.service.store.read(key + ":report")[0]["receipt"] == history[0][-1]
    assert h.planner.report(f.binding.run_id).result_refs


@pytest.mark.parametrize("failure", [OSError("object_failure"), ValueError("object_checksum")])
def test_unknown_install_failure_keeps_pending_and_same_key(monkeypatch, failure):
    f = setup([review_from_wire])
    h = integrated(f)
    install = h.service.context.installer

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(h.service.context, "installer", fail)
    with pytest.raises(type(failure), match=str(failure)):
        h.service.run(**h.args)
    assert h.planner.read(f.binding.run_id).pending is not None
    assert h.service.store.read(h.args["event"].code + ":result") is None
    saved = h.service.acceptance.read(key=h.args["event"].code + ":domain")
    monkeypatch.setattr(h.service.context, "installer", install)
    assert h.service.run(**h.args)["status"] == "accepted"
    assert h.service.acceptance.read(key=h.args["event"].code + ":domain") == saved
    assert f.budget.used == 1


def test_exhausted_steps_same_key_and_accepted_deadline_recovery(monkeypatch):
    f = setup([review_from_wire], binding=locked(max_steps=8))
    h = integrated(f)
    result = h.service.run(**h.args)
    assert result["reason"] == "step_budget" and f.budget.used == 0
    assert h.service.run(**h.args) == result
    assert h.planner.read(f.binding.run_id).steps == 9  # A stop event, not a new request.

    f = setup([review_from_wire])
    h = integrated(f)
    accept = h.service.acceptance.accept

    def lost(**kwargs):
        accept(**kwargs)
        raise OSError("accepted_lost")

    monkeypatch.setattr(h.service.acceptance, "accept", lost)
    with pytest.raises(OSError):
        h.service.run(**h.args)
    saved = h.service.acceptance.read(key=h.args["event"].code + ":domain")
    h.planner.clock = f.budget.clock = lambda: f.binding.deadline
    result = h.service.run(**h.args)
    assert result["domain_status"] == "accepted" and result["reason"] == "deadline"
    assert h.service.acceptance.read(key=h.args["event"].code + ":domain") == saved
    assert h.service.run(**h.args) == result and f.budget.used == 1


def test_v1_result_exact_readback_after_new_control_stop(monkeypatch):
    f = setup([review_from_wire])
    h = integrated(f)
    result = h.service.run(**h.args)
    old = {
        k: v
        for k, v in result.items()
        if k not in {"domain_status", "successor_status", "continuation", "cancelled"}
    }
    old["schema_version"] = "research-review-result/1.0.0"
    assert read_result(old) is old
    assert read_result(result) is result
    # A synthetic pre-upgrade document: the historical reader must neither add
    # default fields nor reauthorize its owner. No old on-disk evidence is edited.
    legacy = setup([review_from_wire])
    old = run(legacy, role="review")
    for field in (
        "domain_status",
        "successor_status",
        "continuation",
        "cancelled",
        "clinical_acceptance",
        "new_medical_review",
    ):
        old.pop(field)
    old["schema_version"] = "research-review-result/1.0.0"
    key = "review:" + legacy.frame.event.code + ":result"
    legacy.documents.saved[key] = (old,)
    legacy.context.checkpoints = [old]
    legacy.reader.withdrawn = True
    assert run(legacy, role="review") == old
    assert legacy.documents.saved[key] == (old,) and legacy.budget.used == 1


def test_cancel_confirmation_claim_lost_is_resumed_without_signal(monkeypatch):
    f = setup([review_from_wire])
    h = integrated(f)
    result = h.service.run(**h.args)
    save = h.planner.runtime.save

    def lost(old, new):
        value = save(old, new)
        if new.pending and new.pending.command.action == "cancel":
            raise OSError("cancel_claim_lost")
        return value

    signal = Event()
    signal.set()
    monkeypatch.setattr(h.planner.runtime, "save", lost)
    with pytest.raises(OSError, match="cancel_claim_lost"):
        h.service.context.checkpoint(h.args["event"], result, cancel=signal)
    monkeypatch.setattr(h.planner.runtime, "save", save)
    state = h.planner.read(f.binding.run_id)
    assert state.cancelled and state.pending.command.action == "cancel"
    assert h.service.run(**h.args) == result
    state = h.planner.replay(f.binding.run_id)
    assert state.cancelled and state.pending is None and state.phase == "stopped"
    assert f.budget.used == 1


def test_required_expert_cannot_use_baseline_at_last_request_slot():
    f = setup([review_from_wire], binding=locked(expert_required=True))
    for _ in range(f.budget.max_requests - 1):
        f.budget.reserve(f.frame.scope, f.binding.deadline, digest(f.provider.config))
    h = integrated(f)
    result = h.service.run(**h.args)
    assert result["status"] == "safe_stop"
    assert f.budget.used == f.budget.max_requests
    assert h.service.acceptance.read(key=h.args["event"].code + ":domain") is None


@pytest.mark.parametrize("point", ["selected", "decision"])
@pytest.mark.parametrize("change", ["withdraw", "policy", "deadline", "cancel"])
def test_saved_advice_stages_recheck_before_request_or_return(monkeypatch, point, change):
    f = setup([review_from_wire, expert_reply()], binding=locked(expert_required=True))
    h = integrated(f)
    write = f.documents.write

    def lost(key, values):
        write(key, values)
        if key.endswith(":" + point):
            raise OSError("stage_lost")

    monkeypatch.setattr(f.documents, "write", lost)
    with pytest.raises(OSError, match="stage_lost"):
        h.service.run(**h.args)
    monkeypatch.setattr(f.documents, "write", write)
    requests = f.budget.used
    signal = Event()
    if change == "withdraw":
        f.reader.withdrawn = True
    elif change == "policy":
        h.service.context.policy = h.service.context.policy.model_copy(
            update={"max_no_progress": 3}
        )
    elif change == "deadline":
        h.planner.clock = lambda: f.binding.deadline
    else:
        signal.set()
    result = h.service.run(**h.args, cancel=signal)
    assert result["status"] == "safe_stop" and f.budget.used == requests
    assert h.planner.read(f.binding.run_id).phase == "stopped"


def test_document_fault_before_acceptance_is_not_forged_terminal(monkeypatch):
    f = setup([review_from_wire])
    h = integrated(f)
    write = f.documents.write

    def fail(key, values):
        if key.endswith(":selected"):
            raise ValueError("object_checksum")
        write(key, values)

    monkeypatch.setattr(f.documents, "write", fail)
    with pytest.raises(ValueError, match="object_checksum"):
        h.service.run(**h.args)
    assert h.planner.read(f.binding.run_id).pending is not None
    monkeypatch.setattr(f.documents, "write", write)
    assert h.service.run(**h.args)["status"] == "accepted"
    assert f.budget.used == 1
