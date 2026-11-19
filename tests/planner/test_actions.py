"""08B adversarial contracts and select integration; every external process is forbidden."""

from datetime import timedelta
from threading import Event

import pytest

from agents.planner.service import Planner
from packages.grounding.models import digest, validate
from packages.grounding.query_models import Component
from packages.grounding.synthetic import ref
from packages.planner_control.actions import (
    RankedActions,
    action,
    baseline,
    build_set,
    unknown_components,
)
from packages.planner_control.models import (
    ActionDecision,
    Candidate,
    LegalActionSet,
    Phase,
    Profile,
    RankingPolicy,
    Reference,
    TraceRanking,
)
from tests.planner.support import FakeOwner, MemoryRuntime, advance, binding, command


@pytest.fixture(autouse=True)
def forbid_process(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("08B must not start tool or LLM processes")

    monkeypatch.setattr("subprocess.Popen", forbidden)
    monkeypatch.setattr("os.system", forbidden)


def ranked_binding(policy, **changes):
    b = binding(**changes)
    return b.model_copy(
        update={
            "locks": tuple(
                Reference(
                    role="ranking", code=r.code, version=policy.version, checksum=digest(policy)
                )
                if r.role == "ranking"
                else r
                for r in b.locks
            )
        }
    )


class Source:
    """Synthetic installed owner, deliberately independent from state's legal flags."""

    def __init__(self, b):
        self.b = b
        self.safety = digest("current_safety")
        self.availability = digest("current_inputs")
        self.choices = tuple(
            Candidate(id=n, branch=n, operation="observe", and_ready=True) for n in ("A", "B")
        )
        self.reverse = False
        self.blocked = set()
        self.metrics = {}
        self.failures = {}

    def read(self, state, policy):
        choices = []
        for c in self.choices:
            if c.branch in self.blocked:
                continue
            components = list(unknown_components())
            for name, number in self.metrics.get(c.branch, {}).items():
                i = next(i for i, v in enumerate(components) if v.name == name)
                components[i] = Component(
                    name=name,
                    state="known",
                    value=number,
                    source_ref=ref("source_version", "SOURCE:SYNTHETIC:ESTIMATE"),
                )
            choices.append(
                action(
                    kind="observe",
                    target=state.graph,
                    branch=c.branch,
                    candidate=c,
                    candidate_ref=digest(c.branch),
                    required_inputs=(),
                    entrypoint="ResearchPlannerSteps.invoke",
                    preconditions=("owner_verified",),
                    reasons=("synthetic_existing",),
                    components=tuple(components),
                    failure_count=self.failures.get(c.branch, 0),
                    failure_history=tuple(
                        digest(dict(branch=c.branch, failure=n))
                        for n in range(self.failures.get(c.branch, 0))
                    ),
                    expires_at=state.binding.deadline,
                )
            )
        return build_set(
            state,
            policy,
            choices=tuple(reversed(choices)) if self.reverse else choices,
            safety=self.safety,
            availability=self.availability,
        )


class Suggest:
    def __init__(self, change=None):
        self.calls = 0
        self.change = change

    def rank(self, s, *, deadline):
        self.calls += 1
        a = next(a for a in s.actions if a.branch == "B")
        ranking = TraceRanking(
            set_checksum=s.checksum,
            policy_checksum=digest(s.policy),
            ranked_ids=(a.id,),
            reason="B has the preferred existing evidence",
        )
        return self.change(s, ranking) if self.change else ranking


def setup_ranked(*, policy=None, agent=None, profile=None):
    policy = policy or RankingPolicy()
    b = ranked_binding(policy, **({"profile": profile} if profile else {}))
    r, o, source = MemoryRuntime(), FakeOwner(b), Source(b)
    o.choices = source.choices
    p = Planner(runtime=r, owner=o, ranking=RankedActions(source, policy=policy, agent=agent))
    p.begin(b)
    advance(p, until="select")
    return p, r, o, source


def select(p):
    return p.step(p.runtime.state.binding.run_id, command(p, "select"))


def test_advised_b_is_selected_persisted_and_same_key_recovery():
    agent = Suggest()
    p, r, o, s = setup_ranked(agent=agent)
    cmd = command(p, "select")
    event = p.step(r.state.binding.run_id, cmd)
    assert r.state.selected.id == "B"
    d = event.outcome.decision
    assert d.strategy == "trace-ranking/1.0.0" and d.selected.candidate.id == "B"
    assert d.ranking.set_checksum == d.action_set.checksum
    assert d.action_set.binding == r.state.binding
    snapshot = list(r.audit)
    # Recreate Planner: persisted event, not an in-memory query cache, answers the retry.
    p = Planner(runtime=r, owner=o, ranking=p.ranking)
    assert p.step(r.state.binding.run_id, cmd) == event and r.audit == snapshot
    assert agent.calls == 1 and "execute" not in o.calls


def test_semantic_baseline_b_and_candidate_id_cannot_change_choice():
    p, r, _, s = setup_ranked()
    s.metrics = {"A": {"directness": 0.1}, "B": {"directness": 0.9}}
    initial = p.ranking.current(r.state)
    assert next(a for a in initial.actions if a.id == baseline(initial).ranked_ids[0]).branch == "B"
    s.choices = tuple(
        c.model_copy(update={"id": "ZZ" if c.branch == "A" else "00"}) for c in s.choices
    )
    state = r.state.model_copy(update={"candidates": s.choices})
    changed = p.ranking.current(state)
    assert baseline(initial).ranked_ids == baseline(changed).ranked_ids
    # Actual select integration for the deterministic new policy, with original owner values.
    s.choices = r.state.candidates
    assert select(p).outcome.decision.selected.candidate.id == "B"


def test_set_order_equivalent_ties_unknown_and_failure_penalty():
    p, r, _, s = setup_ranked()
    first = p.ranking.current(r.state)
    s.reverse = True
    assert p.ranking.current(r.state) == first
    assert all(v.value is None for a in first.actions for v in a.components)
    assert baseline(first).ranked_ids == baseline(p.ranking.current(r.state)).ranked_ids
    s.metrics = {"B": {"cost": 0.0}}
    assert (
        next(
            a
            for a in p.ranking.current(r.state).actions
            if a.id == baseline(p.ranking.current(r.state)).ranked_ids[0]
        ).branch
        == "B"
    )
    s.failures = {"B": 1}
    assert select(p).outcome.decision.selected.branch == "A"


@pytest.mark.parametrize(
    "mutation", ["unknown", "duplicate", "checksum", "extra", "version", "construct"]
)
def test_invalid_advice_falls_back_once(mutation):
    def corrupt(s, r):
        if mutation == "unknown":
            return r.model_copy(update={"ranked_ids": ("invented",)})
        if mutation == "duplicate":
            return r.model_copy(update={"ranked_ids": r.ranked_ids * 2})
        if mutation == "checksum":
            return r.model_copy(update={"set_checksum": digest("forged")})
        if mutation == "extra":
            return r.model_copy(update={"argv": ["bad"]})
        if mutation == "version":
            return r.model_copy(update={"schema_version": "trace-ranking/99.0.0"})
        return TraceRanking.model_construct(**{**r.model_dump(), "reason": ""})

    agent = Suggest(corrupt)
    p, r, _, _ = setup_ranked(policy=RankingPolicy(max_agent_attempts=3), agent=agent)
    d = select(p).outcome.decision
    assert d.strategy == "semantic-baseline/1.0.0" and r.state.selected.id == "A"
    assert d.rejected_advice == ("invalid_advice",) and agent.calls == 1


def test_unavailable_agent_bounded_retry_and_explicit_legacy_fallback():
    def unavailable(*_):
        raise ConnectionError()

    agent = Suggest(unavailable)
    p, _, _, s = setup_ranked(
        policy=RankingPolicy(max_agent_attempts=2, fallback="legacy_id_baseline"), agent=agent
    )
    s.metrics = {"B": {"directness": 1.0}}
    d = select(p).outcome.decision
    assert d.strategy == "legacy-id/1.0.0" and d.selected.branch == "A" and agent.calls == 2


@pytest.mark.parametrize("change", ["graph", "input", "safety", "withdrawn", "and"])
def test_inflight_changes_reject_old_advice(change):
    agent = Suggest()
    p, r, o, source = setup_ranked(agent=agent)

    def mutate(s, advice):
        if change == "graph":
            o.graph = o.graph.model_copy(update={"checksum": digest("changed")})
        elif change == "input":
            source.availability = digest("changed")
        elif change == "safety":
            source.safety = digest("revoked")
        else:
            source.blocked.add("B")
        return advice

    agent.change = mutate
    e = select(p)
    assert e.outcome.status == "rejected" and r.state.selected is None
    assert r.state.phase == Phase.STOPPED and "prepare" not in o.calls


def test_slow_advice_fallback_and_absolute_deadline_stop():
    for expire_run in (False, True):
        agent = Suggest()
        p, r, _, _ = setup_ranked(agent=agent)
        now = [r.state.binding.started_at]
        p.clock = lambda now=now: now[0]

        def delayed(s, advice, now=now, expire_run=expire_run):
            now[0] += timedelta(seconds=301 if expire_run else 31)
            return advice

        agent.change = delayed
        event = select(p)
        if expire_run:
            assert event.after == Phase.STOPPED and r.state.selected is None
        else:
            assert event.outcome.decision.strategy == "semantic-baseline/1.0.0"
            assert "advice_timeout" in event.outcome.decision.rejected_advice


def test_cancel_during_advice_stops_without_selected_work():
    cancellation, agent = Event(), Suggest()
    p, r, _, _ = setup_ranked(agent=agent)
    agent.change = lambda s, r: (cancellation.set(), r)[1]
    p.step(r.state.binding.run_id, command(p, "select"), cancel=cancellation)
    assert r.state.cancelled and r.state.phase == Phase.STOPPED and r.state.selected is None


@pytest.mark.parametrize("stage", ["prepare", "execute"])
def test_readmission_revocation_before_prepare_or_launch(stage):
    p, r, o, s = setup_ranked(agent=Suggest())
    select(p)
    if stage == "execute":
        p.step(r.state.binding.run_id, command(p, "prepare"))
    s.safety = digest("revoked")
    e = p.step(r.state.binding.run_id, command(p, stage))
    assert e.after == Phase.STOPPED and "execute" not in o.calls
    assert r.state.started == 0


def test_reserved_last_action_remains_valid_for_launch_and_unknown_not_retried():
    p, r, o, _ = setup_ranked(agent=Suggest(), profile=Profile(max_actions=1))
    select(p)
    p.step(r.state.binding.run_id, command(p, "prepare"))
    assert r.state.phase == Phase.EXECUTE and r.state.reserved == 1
    o.failure = "execute"
    c = command(p, "execute")
    for _ in range(4):
        p.step(r.state.binding.run_id, c)
    assert r.state.pending.launch_claimed and o.calls.count("execute") == 1


def test_empty_budget_and_insufficient_and_safely_stop():
    for profile in (Profile(max_actions=0), Profile()):
        p, r, _, s = setup_ranked(profile=profile)
        if profile.max_actions:
            s.blocked.update(("A", "B"))
        d = select(p).outcome.decision
        assert d.selected.kind == "stop" and r.state.phase == Phase.STOPPED
        assert "empty" in {x.category for x in d.action_set.rejections}
        if profile.max_actions == 0:
            assert "budget" in {x.category for x in d.action_set.rejections}


@pytest.mark.parametrize(
    "mutation", ["id", "duplicate", "order", "checksum", "expiry", "construct"]
)
def test_action_set_revalidates_copies_and_construct(mutation):
    p, r, _, _ = setup_ranked()
    s = p.ranking.current(r.state)
    if mutation == "id":
        s = s.model_copy(
            update={
                "actions": (s.actions[0].model_copy(update={"id": "arbitrary"}), *s.actions[1:])
            }
        )
    elif mutation == "duplicate":
        s = s.model_copy(update={"actions": s.actions + s.actions[:1]})
    elif mutation == "order":
        s = s.model_copy(update={"actions": tuple(reversed(s.actions))})
    elif mutation == "checksum":
        s = s.model_copy(update={"checksum": digest("tampered")})
    elif mutation == "expiry":
        s = s.model_copy(update={"expires_at": s.expires_at + timedelta(seconds=1)})
    else:
        s = LegalActionSet.model_construct(**{**s.model_dump(), "schema_version": "wrong"})
    with pytest.raises(ValueError):
        validate(LegalActionSet, s)


def test_decision_cannot_reorder_baseline_or_selected_or_scope():
    p, _, _, _ = setup_ranked()
    d = select(p).outcome.decision
    with pytest.raises(ValueError):
        validate(ActionDecision, d.model_copy(update={"selected_id": "fake"}))
    ranked = d.ranking.model_copy(update={"ranked_ids": tuple(reversed(d.ranking.ranked_ids))})
    with pytest.raises(ValueError, match="baseline_order"):
        validate(
            ActionDecision,
            d.model_copy(update={"ranking": ranked, "selected_id": ranked.ranked_ids[0]}),
        )


def test_exact_policy_lock_required():
    p, r, _, _ = setup_ranked()
    p.ranking.policy = RankingPolicy(order=("cost",))
    assert select(p).after == Phase.STOPPED and r.state.selected is None


def test_selection_confirmation_lost_reply_is_same_event():
    p, r, _, _ = setup_ranked(agent=Suggest())
    c = command(p, "select")
    # Pending already saved: fail after the final atomic event commit.
    original = r.save

    def save(old, new):
        if len(new.events) > len(old.events):
            r.fail = "lost_reply"
        return original(old, new)

    r.save = save
    with pytest.raises(RuntimeError):
        p.step(r.state.binding.run_id, c)
    assert r.state.selected.id == "B"
    assert p.step(r.state.binding.run_id, c) == r.state.events[-1]


def test_pending_selection_survives_confirmation_failure_without_another_advice():
    agent = Suggest()
    p, r, _, _ = setup_ranked(agent=agent)
    c = command(p, "select")
    original = r.save

    def fail_confirmation(old, new):
        if len(new.events) > len(old.events):
            raise RuntimeError("confirmation_transaction_failed")
        return original(old, new)

    r.save = fail_confirmation
    with pytest.raises(RuntimeError):
        p.step(r.state.binding.run_id, c)
    assert r.state.pending.decision.selected.branch == "B"
    r.save = original
    e = p.step(r.state.binding.run_id, c)
    assert e.outcome.decision.selected.branch == "B" and agent.calls == 1


def test_pending_selection_revocation_refused_without_reasking_agent():
    agent = Suggest()
    p, r, _, s = setup_ranked(agent=agent)
    c = command(p, "select")
    original = r.save

    def fail_confirmation(old, new):
        if len(new.events) > len(old.events):
            raise RuntimeError()
        return original(old, new)

    r.save = fail_confirmation
    with pytest.raises(RuntimeError):
        p.step(r.state.binding.run_id, c)
    r.save = original
    s.blocked.add("B")
    assert p.step(r.state.binding.run_id, c).after == Phase.STOPPED
    assert r.state.selected is None and agent.calls == 1


def test_technical_source_error_distinct_from_empty_legal_set():
    p, r, _, s = setup_ranked()

    def unavailable(*args):
        raise ConnectionError("do not expose raw detail")

    s.read = unavailable
    event = select(p)
    assert event.outcome.gaps[0].kind == "technical"
    assert event.outcome.reason == "action_source_technical_error"
    assert r.state.selected is None


def test_information_gain_disabled_is_retained_but_does_not_dominate():
    p, r, _, source = setup_ranked(policy=RankingPolicy(order=("information_gain", "directness")))
    source.metrics = {"A": {"information_gain": 10.0}, "B": {"directness": 1.0}}
    d = select(p).outcome.decision
    assert d.selected.branch == "B"
    assert d.action_set.actions[0].components[0].value == 10.0


def test_saved_selection_state_tampering_rejected_on_read():
    p, r, _, _ = setup_ranked(agent=Suggest())
    select(p)
    from packages.planner_control.models import Control

    with pytest.raises(ValueError, match="selected_state_mismatch"):
        validate(Control, r.state.model_copy(update={"selected": r.state.candidates[0]}))


def test_baseline_cannot_revive_missing_and_or_failed_branch():
    p, r, _, s = setup_ranked(agent=Suggest())
    s.blocked.add("B")
    d = select(p).outcome.decision
    assert d.selected.branch == "A" and d.rejected_advice == ("invalid_advice",)
    p, r, _, s = setup_ranked()
    state = r.state.model_copy(update={"attempted": ("A",)})
    actions = p.ranking.current(state)
    assert not any(a.branch == "A" for a in actions.actions)
    assert next(a for a in actions.actions if a.id == baseline(actions).ranked_ids[0]).branch == "B"


def test_permuted_identified_set_has_identical_binding():
    p, r, _, s = setup_ranked()
    a = p.ranking.current(r.state)
    state = r.state.model_copy(update={"candidates": tuple(reversed(r.state.candidates))})
    s.reverse = True
    assert p.ranking.current(state) == a
