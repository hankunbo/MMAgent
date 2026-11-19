"""Synthetic graph routes and Fake execution facts; no real tool or model calls."""

from types import SimpleNamespace

from agents.expert_agent.models import ReviewFacts, ReviewPolicy
from agents.planner import Planner
from apps.expert_agent_assembly import assemble_research_review
from packages.grounding.models import digest
from packages.planner_control.actions import RankedActions
from packages.planner_control.models import Command, Gap, Outcome, RankingPolicy
from packages.validation_recovery import (
    BranchRoute,
    Failure,
    NearestLegalActions,
    RecoveryPolicy,
    ValidationRecovery,
)
from tests.agent_provider.support import ref
from tests.expert_agent.support import protocol, setup
from tests.planner.support import FakeOwner, MemoryRuntime


def fixture(
    *, replies=(), pg=None, required=False, category="execution_failed", blocked=(), profile=None
):
    f = setup(
        replies,
        **(
            dict(
                binding=pg.binding,
                reader=pg.reader,
                documents=pg.documents,
                budget=pg.budget,
                config=pg.config,
            )
            if pg
            else {}
        ),
    )
    if profile is not None:
        f.binding = f.binding.model_copy(update={"profile": profile})
    policy = RecoveryPolicy(
        routes=(
            BranchRoute(branch="A", forks=("root", "near")),
            BranchRoute(branch="B", forks=("root", "near")),
        ),
        source=next(r for r in f.binding.locks if r.role == "rule"),
    )
    gap = (
        Gap(
            kind="capability",
            target="GOAL:SYNTHETIC",
            reason="resolved_missing_implementation",
            capability_code="CAPABILITY:SYNTHETIC",
            requirement_ref=ref("requirement", "REQUIREMENT:SYNTHETIC"),
        )
        if category == "capability_gap"
        else None
    )
    failure = Failure(
        category=category,
        target="GOAL:SYNTHETIC",
        reason="synthetic_failure",
        sources=(ref("observation", "OBS:SYNTHETIC"),),
        gap=gap,
    )

    class Owner(FakeOwner):
        def current(self, binding):
            return f.reader.current().material.scope.graph

        def invoke(self, state, pending, *, cancel):
            self.calls.append(pending.command.action)
            if pending.command.action == "identify":
                return Outcome(candidates=self.choices)
            if pending.command.action == "execute":
                return Outcome(
                    status="rejected" if state.selected.branch == "A" else "ok",
                    reason="synthetic_execution_failed"
                    if state.selected.branch == "A"
                    else "synthetic_observed",
                    refs=(ref("observation", "OBS:" + state.selected.branch),),
                    gaps=(gap,) if gap and state.selected.branch == "A" else (),
                    progress=state.selected.branch != "A",
                )
            return Outcome()

    owner = Owner(f.binding)
    owner.choices = f.source.choices
    runtime = pg.documents.sessions if pg else MemoryRuntime()
    nearest = NearestLegalActions(f.source, policy)
    planner = Planner(runtime=runtime, owner=owner, ranking=RankedActions(nearest))
    planner.begin(f.binding)

    def command(action):
        state = planner.read(f.binding.run_id)
        return Command(
            key=f"step-{state.steps}-{action}", action=action, expected_revision=state.revision
        )

    for action in ("load", "retrieve", "instantiate", "identify", "select", "prepare", "execute"):
        planner.step(f.binding.run_id, command(action))
    f.source.blocked.update(blocked)
    review = assemble_research_review(
        planner=planner,
        run_id=f.binding.run_id,
        domain_owner=f.reader,
        documents=f.documents,
        ports=f.ports,
        action_source=nearest,
        ranking_policy=RankingPolicy(),
        review_policy=ReviewPolicy(comparison_protocol=digest(protocol())),
        facts=lambda _: ReviewFacts(required=required),
        install=f.context.install,
        protocol=protocol(),
    )
    service = ValidationRecovery(
        planner=planner,
        review=review,
        documents=f.documents,
        policy=policy,
        failure_reader=lambda _: failure,
    )
    return SimpleNamespace(
        f=f,
        planner=planner,
        owner=owner,
        runtime=runtime,
        policy=policy,
        service=service,
        command=command,
        run_id=f.binding.run_id,
        failure=failure,
    )
