from types import SimpleNamespace

from agents.expert_agent.actions import ResearchActionSource
from agents.expert_agent.models import ReviewFacts, ReviewPolicy
from agents.planner.service import Planner
from apps.api.assembly import Dependencies
from apps.expert_agent_assembly import assemble_research_review
from packages.evidence_details.successor_owner import ResearchSuccessorOwner
from packages.grounding.models import digest
from packages.planner_control.models import Command, Outcome, RankingPolicy
from packages.task_runtime.expert import ExpertDispatch
from packages.task_runtime.expert_models import ExpertDispatchRequest
from packages.task_runtime.owner import PlannerTaskOwner
from tests.agent_provider.support import ref, review_from_wire
from tests.api.support import identities
from tests.expert_agent.support import protocol, setup
from tests.planner.support import FakeOwner, MemoryRuntime


def configured(pg=None, *, replies=None, facts=None):
    args = (
        dict(
            binding=pg.binding,
            reader=pg.reader,
            documents=pg.documents,
            budget=pg.budget,
            config=pg.config,
        )
        if pg
        else {}
    )
    f = setup([review_from_wire] if replies is None else replies, **args)
    domain, install = f.reader, f.context.install
    if pg:
        initial = pg.reader.state

        def verify(state):
            assert state.material.sources == initial.material.sources
            assert state.policy == initial.policy

        domain = ResearchSuccessorOwner(
            initial=initial,
            documents=pg.documents,
            verify_dependencies=verify,
            read_image=lambda _: b"",
            config=pg.config,
            role="review",
        )
        install = domain.install
        for agent in (
            f.ports.trace,
            f.ports.expert,
            f.ports.source,
            f.ports.vision,
            f.ports.review,
        ):
            agent.materials = domain

    class Steps(FakeOwner):
        def current(self, binding):
            return domain.current().material.scope.graph

        def invoke(self, state, pending, *, cancel):
            if pending.command.action == "execute":
                return Outcome(
                    reason="synthetic_no_tool",
                    refs=(
                        ref("material", "ASSET:READ", checksum=digest(domain.current().material)),
                    ),
                )
            return super().invoke(state, pending, cancel=cancel)

    runtime = pg.documents.sessions if pg else MemoryRuntime()
    steps = Steps(f.binding)
    steps.choices = f.source.choices
    planner = Planner(runtime=runtime, owner=steps)
    planner.begin(f.binding)
    for action in (
        "load",
        "retrieve",
        "instantiate",
        "identify",
        "select",
        "prepare",
        "execute",
        "verify",
    ):
        state = planner.read(f.binding.run_id)
        planner.step(
            f.binding.run_id,
            Command(
                key=f"setup-{state.revision}-{action}",
                expected_revision=state.revision,
                action=action,
            ),
        )
    service = assemble_research_review(
        planner=planner,
        run_id=f.binding.run_id,
        domain_owner=domain,
        documents=f.documents,
        ports=f.ports,
        action_source=ResearchActionSource(f.source, domain),
        ranking_policy=RankingPolicy(),
        review_policy=ReviewPolicy(comparison_protocol=digest(protocol())),
        facts=lambda state: ReviewFacts(**(facts or {})),
        install=install,
        protocol=protocol(),
    )
    if pg:
        from db.runtime.repository.research_api import ResearchAPIRepository

        api_runtime = ResearchAPIRepository(pg.target)
    else:
        context = SimpleNamespace(
            synthetic=True,
            run_id=f.binding.run_id,
            checksum=f.binding.run_context_checksum,
            request=SimpleNamespace(scope=SimpleNamespace(case_code=f.binding.case_code)),
        )
        api_runtime = SimpleNamespace(
            read_run=lambda _: SimpleNamespace(context=context),
            get_case_bundle=lambda _: SimpleNamespace(case_code=f.binding.case_code),
        )
    port = ExpertDispatch(service=service, runtime=api_runtime)
    task_owner = PlannerTaskOwner(
        planners={f.binding.run_id: planner}, runtime=api_runtime, reviews={f.binding.run_id: port}
    )
    dependencies = Dependencies(
        identities=identities(cases=[f.binding.case_code]),
        runtime=api_runtime,
        planners={f.binding.run_id: planner},
        expert_requests={f.binding.run_id: port},
        expert_history={f.binding.run_id: service.store},
    )
    state = planner.read(f.binding.run_id)
    actions = service.context.source.read(state, service.context.ranking)
    selected = next(a for a in actions.actions if a.branch == "research_review")
    body = ExpertDispatchRequest(
        api_version="research-http/1.0.0",
        contract_version="research-expert-request/2.0.0",
        idempotency_key=f"dispatch-{state.revision}-review",
        case_code=f.binding.case_code,
        context_checksum=f.binding.run_context_checksum,
        expected_revision=state.revision,
        expected_version=state.graph.checksum,
        proposed_action_id=selected.id,
        role="review",
    )
    return SimpleNamespace(
        f=f,
        domain=domain,
        planner=planner,
        runtime=runtime,
        service=service,
        port=port,
        task_owner=task_owner,
        dependencies=dependencies,
        body=body,
    )


def receipt(s):
    from packages.task_runtime.models import DispatchRequest

    request = DispatchRequest(
        **s.body.model_dump(
            include={"case_code", "context_checksum", "expected_revision", "idempotency_key"}
        ),
        run_id=s.f.binding.run_id,
    )
    return SimpleNamespace(request=request, command=s.port.command(s.body))
