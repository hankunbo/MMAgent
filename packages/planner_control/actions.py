"""08B action assembly/ranking. Authority stays with installed 03/05/07 readers.

No query cache: every selection and execution preparation reads the owner again.
"""

from datetime import timedelta
from typing import Protocol

from packages.grounding.models import digest, seal, validate
from packages.grounding.query_models import METRICS, Component

from .models import (
    ActionBudget,
    ActionDecision,
    ActionRejection,
    Candidate,
    Control,
    Graph,
    LegalAction,
    LegalActionSet,
    Phase,
    RankingPolicy,
    Reference,
    SelectionOutcome,
    TraceRanking,
    request_checksum,
)


class TraceAgent(Protocol):
    def rank(self, actions: LegalActionSet, *, deadline) -> TraceRanking: ...


class ActionSource(Protocol):
    def read(self, state: Control, policy: RankingPolicy) -> LegalActionSet: ...


def unknown_components():
    return tuple(Component(name=n, state="unknown", value=None, source_ref=None) for n in METRICS)


def action(**fields):
    def reference(value):
        if isinstance(value, Graph):
            return Reference(
                role=value.role, code=value.code, version=value.version, checksum=value.checksum
            )
        return value

    fields["target"] = reference(fields["target"])
    fields["required_inputs"] = tuple(
        sorted((reference(r) for r in fields["required_inputs"]), key=digest)
    )
    # Identity can be calculated without bypassing the public validation boundary.
    identity = dict(
        kind=fields["kind"],
        target=fields["target"],
        branch=fields["branch"],
        candidate_ref=fields.get("candidate_ref"),
        inputs=fields["required_inputs"],
        entrypoint=fields["entrypoint"],
    )
    return LegalAction(id="action:" + digest(identity)[7:], **fields)


def state_checksum(state):
    # Pending bookkeeping/revision do not change the semantic selection request.
    return digest(
        dict(
            binding=state.binding,
            graph=state.graph,
            steps=state.steps,
            candidates=sorted(state.candidates, key=lambda c: c.id),
            attempted=sorted(state.attempted),
            reserved=state.reserved,
            branches=state.branches,
            no_progress=state.no_progress,
            cancelled=state.cancelled,
            plans=state.result_refs,
        )
    )


def _owns_prepare_reservation(state, action):
    """Recognize only the current Runtime checkpoint's selected prepare reservation.

    Planner supplies a stored Control under its session lock, never caller reservation
    flags. Runtime CAS already binds the pending command and accounts for its +1.
    This exception changes only capacity filtering; source/member checks still apply.
    """
    pending = state.pending
    if (
        state.phase != Phase.PREPARE
        or pending is None
        or pending.command.action != "prepare"
        or not pending.reserved
        or pending.launch_claimed
        or state.selected != action.candidate
        or not state.events
    ):
        return False
    event = state.events[-1]
    if (
        event.command.action != "select"
        or event.after != Phase.PREPARE
        or not isinstance(event.outcome, SelectionOutcome)
        or event.outcome.status != "ok"
    ):
        return False
    decision = event.outcome.decision
    return (
        decision.selected == action
        and decision.action_set.binding == state.binding
        and decision.action_set.graph == state.graph
        and state.reserved == decision.action_set.budget.reserved + 1
        and pending.request_checksum == request_checksum(state.binding, pending.command)
        and pending.execution_key
        == "planner:" + digest(dict(run=state.binding.run_id, key=pending.command.key))[7:55]
        and pending.command.key not in {e.command.key for e in state.events}
    )


def build_set(
    state,
    policy,
    *,
    choices,
    safety,
    availability,
    sources=(),
    rejections=(),
    enumeration="complete",
):
    state, policy = validate(Control, state), validate(RankingPolicy, policy)
    ranking_lock = next(r for r in state.binding.locks if r.role == "ranking")
    if ranking_lock.version != policy.version or ranking_lock.checksum != digest(policy):
        raise ValueError("ranking_policy_lock_mismatch")
    retained, rejected = [], list(rejections)
    for a in choices:
        a = validate(LegalAction, a)
        branch, failures = None, list(a.failure_history)
        for event in state.events:
            if isinstance(event.outcome, SelectionOutcome):
                branch = event.outcome.decision.selected.branch
            elif (
                branch == a.branch
                and event.command.action in {"prepare", "execute", "verify"}
                and event.outcome.status != "ok"
            ):
                failures.append(digest(event))
        a = validate(
            LegalAction,
            a.model_copy(
                update={"failure_history": tuple(failures), "failure_count": len(failures)}
            ),
        )
        reason = None
        if a.candidate is not None:
            if a.candidate not in state.candidates:
                reason = ("safety", "candidate_not_owner_identified")
            elif a.branch in state.attempted:
                reason = ("already_attempted", "branch_already_attempted")
            elif state.reserved >= state.binding.profile.max_actions and not (
                (state.phase == Phase.EXECUTE and state.selected == a.candidate)
                or _owns_prepare_reservation(state, a)
            ):
                reason = ("budget", "action_budget")
            elif state.no_progress >= state.binding.profile.max_no_progress:
                reason = ("budget", "no_progress_budget")
        if reason:
            rejected.append(
                ActionRejection(target=a.target.code, category=reason[0], reason=reason[1])
            )
        else:
            retained.append(a)
    stop_target = Reference(
        role="control", code=state.binding.run_id, version="1.0.0", checksum=digest(state.binding)
    )
    retained.append(
        action(
            kind="stop",
            target=stop_target,
            branch="stop",
            candidate=None,
            candidate_ref=None,
            required_inputs=(),
            entrypoint="Planner.step",
            preconditions=("active_control",),
            reasons=("safe_stop_available",),
            components=unknown_components(),
            expires_at=state.binding.deadline,
        )
    )
    if (
        state.attempted
        and state.branches < state.binding.profile.max_branches
        and any(a.candidate for a in retained)
    ):
        retained.append(
            action(
                kind="fallback",
                target=stop_target,
                branch="fallback",
                candidate=None,
                candidate_ref=None,
                required_inputs=(),
                entrypoint="Planner.step",
                preconditions=("independent_or_available",),
                reasons=("finite_or",),
                components=unknown_components(),
                expires_at=state.binding.deadline,
            )
        )
    if not any(a.candidate for a in retained):
        rejected.append(
            ActionRejection(target=state.graph.code, category="empty", reason="empty_legal_work")
        )
    return seal(
        LegalActionSet,
        binding=state.binding,
        graph=state.graph,
        state_checksum=state_checksum(state),
        budget=ActionBudget(**{k: getattr(state, k) for k in ActionBudget.model_fields}),
        policy=policy,
        safety_checksum=safety,
        availability_checksum=availability,
        source_checksums=tuple(sorted(set(sources))),
        plan_refs=tuple(r for r in state.result_refs if r.role == "plan"),
        actions=tuple(sorted(retained, key=lambda a: a.semantic_key())),
        rejections=tuple(sorted(rejected, key=digest)),
        enumeration=enumeration,
        expires_at=state.binding.deadline,
    )


def baseline(actions, *, legacy=False):
    actions = validate(LegalActionSet, actions)

    def key(a):
        # Controls follow useful work in the baseline; Trace may explicitly choose stop.
        prefix = (a.candidate is None, a.failure_count)
        if legacy:
            return (*prefix, a.candidate.id if a.candidate else a.kind, a.semantic_key())
        values = []
        components = {c.name: c for c in a.components}
        for name in actions.policy.order:
            if name == "information_gain" and not actions.policy.information_gain_enabled:
                continue
            c = components[name]
            # The first tuple element makes unknown a separate class, never numeric zero.
            values.append(
                (
                    c.state != "known",
                    (-(c.value) if name in {"information_gain", "directness"} else c.value)
                    if c.state == "known"
                    else 0,
                )
            )
        return (*prefix, *values, a.semantic_key())

    return TraceRanking(
        set_checksum=actions.checksum,
        policy_checksum=digest(actions.policy),
        ranked_ids=tuple(a.id for a in sorted(actions.actions, key=key)),
        reason="versioned_deterministic_baseline",
    )


class RankedActions:
    """Installed source grants membership; an optional agent only orders its IDs."""

    def __init__(self, source, *, policy=None, agent=None):
        self.source = source
        self.policy = validate(RankingPolicy, policy or RankingPolicy())
        self.agent = agent

    def current(self, state):
        result = validate(LegalActionSet, self.source.read(state, self.policy))
        if (result.binding, result.graph, result.state_checksum, result.policy) != (
            state.binding,
            state.graph,
            state_checksum(state),
            self.policy,
        ):
            raise ValueError("action_scope_mismatch")
        if result.budget != ActionBudget(
            **{k: getattr(state, k) for k in ActionBudget.model_fields}
        ):
            raise ValueError("action_budget_snapshot_mismatch")
        return result

    def select(self, state, clock):
        actions = self.current(state)
        deadline = min(
            actions.expires_at, clock() + timedelta(seconds=self.policy.suggestion_seconds)
        )
        errors, attempts, ranking = [], 0, None
        if self.agent is not None:
            for _ in range(self.policy.max_agent_attempts):
                if clock() >= deadline:
                    break
                attempts += 1
                try:
                    proposed = validate(TraceRanking, self.agent.rank(actions, deadline=deadline))
                    if (proposed.set_checksum, proposed.policy_checksum) != (
                        actions.checksum,
                        digest(self.policy),
                    ) or not set(proposed.ranked_ids) <= {a.id for a in actions.actions}:
                        raise ValueError("illegal_ranking")
                    ranking = proposed
                    break
                except (TimeoutError, ConnectionError):
                    errors.append("agent_unavailable")
                except Exception:
                    # Never ask the same invalid proposal again.
                    errors.append("invalid_advice")
                    break
        fresh = self.current(state)
        if fresh != actions:
            raise ValueError("stale_action_set")
        now = clock()
        if now >= actions.expires_at:
            raise ValueError("action_deadline")
        if now >= deadline:
            ranking = None
            errors.append("advice_timeout")
        if ranking is None:
            legacy = self.policy.fallback == "legacy_id_baseline"
            ranking = baseline(fresh, legacy=legacy)
            strategy = "legacy-id/1.0.0" if legacy else "semantic-baseline/1.0.0"
        else:
            strategy = "trace-ranking/1.0.0"
        return validate(
            ActionDecision,
            ActionDecision(
                action_set=fresh,
                ranking=ranking,
                strategy=strategy,
                selected_id=ranking.ranked_ids[0],
                attempts=attempts,
                rejected_advice=tuple(errors),
                decided_at=now,
            ),
        )

    def resume_selection(self, state, decision, clock):
        decision = validate(ActionDecision, decision)
        if self.current(state) != decision.action_set or clock() >= decision.action_set.expires_at:
            raise ValueError("stale_pending_selection")
        return decision

    def recheck(self, state, decision, clock):
        decision = validate(ActionDecision, decision)
        fresh = self.current(state)
        old = decision.action_set
        if clock() >= old.expires_at or (
            fresh.binding,
            fresh.graph,
            fresh.safety_checksum,
            fresh.availability_checksum,
            fresh.source_checksums,
        ) != (
            old.binding,
            old.graph,
            old.safety_checksum,
            old.availability_checksum,
            old.source_checksums,
        ):
            raise ValueError("selection_source_changed")
        # Initial prepare needs capacity; resumed prepare and execute may own it already.
        # Neither case bypasses the fresh source or exact selected-action checks.
        selected = decision.selected
        if selected not in fresh.actions or selected.candidate != state.selected:
            raise ValueError("selection_no_longer_legal")


class SavedResearchActions:
    """Only the installed 07 saved-snapshot observer; no caller self-attested legality."""

    def __init__(self, snapshots):
        self.snapshots = snapshots

    def read(self, state, policy):
        return self.read_with_domain(state, policy)[0]

    def read_with_domain(self, state, policy):
        from packages.research_runtime.live_snapshots import LiveSnapshots

        if type(self.snapshots) is LiveSnapshots:
            graph, domain = self.snapshots.current_with_domain(state.binding)
        else:
            graph, domain = self.snapshots.current(state.binding), None
        if graph != state.graph:
            raise ValueError("research_snapshot_changed")
        c = Candidate(
            id="installed_observe", branch="installed", operation="observe", and_ready=True
        )
        a = action(
            kind="observe",
            target=state.graph,
            branch=c.branch,
            candidate=c,
            candidate_ref=digest(state.graph),
            required_inputs=(state.graph,),
            entrypoint="ResearchPlannerSteps.invoke",
            preconditions=("saved_snapshot_valid",),
            reasons=("read_existing_and_recompute_saved_v5",),
            components=unknown_components(),
            expires_at=state.binding.deadline,
        )
        rejected = tuple(
            ActionRejection(target=t, category="missing_implementation", reason=r)
            for t, r in (
                ("controlled_retrieval", "08C_search_step_not_installed"),
                ("demand_candidate", "08C_candidate_submission_not_installed"),
                ("visual_observation", "08C_visual_step_not_installed"),
                ("medical_review", "08C_review_acceptance_not_installed"),
                ("expert_review", "08D_expert_not_installed"),
                ("export", "export_step_not_installed"),
            )
        )
        actions = build_set(
            state,
            policy,
            choices=(a,),
            safety=digest(state.binding.locks),
            availability=state.binding.visible_checksum,
            sources=(state.graph.checksum,),
            rejections=rejected,
        )
        return actions, domain
