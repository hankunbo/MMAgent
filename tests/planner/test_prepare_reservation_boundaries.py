"""Last-slot prepare recovery keeps admission, budget ownership and idempotency boundaries."""

from datetime import timedelta
from threading import Event

import pytest

from agents.planner.service import Planner
from packages.grounding.models import digest
from packages.planner_control.models import Phase, Profile
from tests.planner.support import advance, command, setup
from tests.planner.test_actions import Suggest, select, setup_ranked
from tests.planner.test_prepare_reservation_recovery import inject_save_failure
from tests.planner.test_prepare_reservation_recovery import no_tool_execution as no_tool_execution


def interrupted(*, stage="after_reservation", max_actions=1):
    agent = Suggest()
    p, r, owner, source = setup_ranked(agent=agent, profile=Profile(max_actions=max_actions))
    select(p)
    cmd = command(p, "prepare")
    original = inject_save_failure(r, stage)
    with pytest.raises(ConnectionError):
        p.step(r.state.binding.run_id, cmd)
    r.save = original
    return p, r, owner, source, agent, cmd


@pytest.mark.parametrize("stage", ["after_reservation", "before_confirmation"])
def test_last_slot_of_multiple_reservations(stage):
    agent = Suggest()
    p, r, owner, _ = setup_ranked(agent=agent, profile=Profile(max_actions=2))
    select(p)
    owner.failure = "prepare"
    assert p.step(r.state.binding.run_id, command(p, "prepare")).after == Phase.FALLBACK
    owner.failure = None
    p.step(r.state.binding.run_id, command(p, "fallback"))
    select(p)
    assert r.state.selected.id == "A" and r.state.reserved == 1
    asks = agent.calls
    cmd = command(p, "prepare")
    original = inject_save_failure(r, stage)
    with pytest.raises(ConnectionError):
        p.step(r.state.binding.run_id, cmd)
    r.save = original
    assert r.state.reserved == 2 and r.state.pending.reserved
    p = Planner(runtime=r, owner=owner, ranking=p.ranking)
    assert p.step(r.state.binding.run_id, cmd).after == Phase.EXECUTE
    assert r.state.reserved == 2 and agent.calls == asks and "execute" not in owner.calls


@pytest.mark.parametrize("stage", ["after_reservation", "before_confirmation"])
@pytest.mark.parametrize(
    "change", ["safety", "input", "withdrawn", "source", "graph", "cancel", "deadline", "version"]
)
def test_recovery_does_not_bypass_current_facts_or_stop_signals(stage, change):
    p, r, owner, source, agent, cmd = interrupted(stage=stage)
    prepares = owner.calls.count("prepare")
    signal = Event()
    if change == "safety":
        source.safety = digest("revoked")
    elif change == "input":
        source.availability = digest("input_changed")
    elif change == "withdrawn":
        source.blocked.add("B")
    elif change == "source":

        def rejected(_):
            raise ValueError("source_rejected")

        owner.current = rejected
    elif change == "graph":
        owner.graph = owner.graph.model_copy(update={"checksum": digest("graph_changed")})
    elif change == "cancel":
        signal.set()
    elif change == "deadline":
        p.clock = lambda: r.state.binding.deadline + timedelta(seconds=1)
    else:
        p.ranking.policy = p.ranking.policy.model_copy(update={"suggestion_seconds": 29})
    event = p.step(r.state.binding.run_id, cmd, cancel=signal)
    assert event.after == (Phase.RETRIEVE if change == "graph" else Phase.STOPPED)
    assert r.state.reserved == 1 and owner.calls.count("prepare") == prepares
    assert "execute" not in owner.calls and agent.calls == 1
    assert not r.state.pending and (r.state.cancelled if change == "cancel" else True)
    state, audits = r.state, tuple(r.audit)
    assert p.step(state.binding.run_id, cmd) == event
    assert r.state == state and tuple(r.audit) == audits


@pytest.mark.parametrize("mutation", ["key", "input_refs", "revision"])
def test_recovery_requires_exact_original_command(mutation):
    p, r, owner, _, agent, cmd = interrupted()
    changed = cmd.model_copy(
        update={"key": "other_key"}
        if mutation == "key"
        else {"input_refs": (r.state.binding.locks[0],)}
        if mutation == "input_refs"
        else {"expected_revision": cmd.expected_revision + 1}
    )
    before = r.state
    with pytest.raises(ValueError, match="pending_step_requires_same_command"):
        p.step(r.state.binding.run_id, changed)
    assert r.state == before and "prepare" not in owner.calls and agent.calls == 1
    assert p.step(r.state.binding.run_id, cmd).after == Phase.EXECUTE


@pytest.mark.parametrize(
    "mutation",
    ["flag", "pending", "key", "execution_key", "action", "attempted", "candidate", "branch"],
)
def test_only_reserved_selected_action_can_use_capacity_exception(mutation):
    p, r, _, _, _, _ = interrupted()
    state = r.state
    if mutation == "flag":
        state = state.model_copy(
            update={"pending": state.pending.model_copy(update={"reserved": False})}
        )
    elif mutation == "pending":
        state = state.model_copy(update={"pending": None})
    elif mutation == "key":
        state = state.model_copy(
            update={
                "pending": state.pending.model_copy(
                    update={"command": state.pending.command.model_copy(update={"key": "wrong"})}
                )
            }
        )
    elif mutation == "execution_key":
        state = state.model_copy(
            update={"pending": state.pending.model_copy(update={"execution_key": "wrong"})}
        )
    elif mutation == "action":
        state = state.model_copy(
            update={
                "pending": state.pending.model_copy(
                    update={
                        "command": state.pending.command.model_copy(update={"action": "select"})
                    }
                )
            }
        )
    elif mutation == "attempted":
        state = state.model_copy(update={"attempted": ("B",)})
    else:
        wrong = (
            state.candidates[0]
            if mutation == "candidate"
            else state.selected.model_copy(update={"branch": "A"})
        )
        with pytest.raises(ValueError):
            r.save(
                state, state.model_copy(update={"revision": state.revision + 1, "selected": wrong})
            )
        assert r.state == state
        return
    assert not any(a.candidate for a in p.ranking.current(state).actions)


def test_owned_reservation_never_enables_other_candidate_or_new_select():
    p, r, _, _, _, _ = interrupted()
    actions = p.ranking.current(r.state)
    assert [a.candidate.id for a in actions.actions if a.candidate] == ["B"]
    state = r.state.model_copy(update={"phase": Phase.SELECT, "pending": None, "selected": None})
    assert not any(a.candidate for a in p.ranking.current(state).actions)


def test_caller_reserved_flag_is_not_a_command_and_checkpoint_needs_counter():
    p, r, _, _, _, cmd = interrupted()
    with pytest.raises(ValueError):
        p.step(
            r.state.binding.run_id,
            cmd.model_copy(update={"reserved": True}),
        )
    # A reservation flag alone cannot be saved on a newly pending prepare.
    p, r, _, _ = setup_ranked(profile=Profile(max_actions=1))
    select(p)
    original = r.save

    def lose_pending(old, new):
        original(old, new)
        if new.pending and not new.pending.reserved:
            raise ConnectionError()
        return new

    r.save = lose_pending
    with pytest.raises(ConnectionError):
        p.step(r.state.binding.run_id, command(p, "prepare"))
    r.save = original
    old = r.state
    with pytest.raises(ValueError, match="reservation_count"):
        r.save(
            old,
            old.model_copy(
                update={
                    "revision": old.revision + 1,
                    "pending": old.pending.model_copy(update={"reserved": True}),
                }
            ),
        )
    assert r.state == old


def test_unranked_08a_recovery_remains_compatible():
    p, r, owner = setup(profile=Profile(max_actions=1))
    advance(p, until="prepare")
    cmd = command(p, "prepare")
    original = inject_save_failure(r, "after_reservation")
    with pytest.raises(ConnectionError):
        p.step(r.state.binding.run_id, cmd)
    r.save = original
    assert p.step(r.state.binding.run_id, cmd).after == Phase.EXECUTE
    assert r.state.reserved == 1 and owner.calls.count("prepare") == 1
