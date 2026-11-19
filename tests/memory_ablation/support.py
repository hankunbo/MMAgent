"""Fresh synthetic sources through original owners; old exports are never imported."""

from datetime import timedelta
from types import SimpleNamespace

from apps.memory_execution_sources import ExecutionSources
from apps.memory_sources import ResearchSources
from packages.contracts import Artifact
from packages.grounding.models import canonical, digest
from packages.llm_provider import RequestBudget
from packages.memory_review.models import Binding
from packages.memory_writer import MemoryWriter
from packages.object_store import checksum
from packages.planner_control.models import Graph, Profile, RankingPolicy
from packages.research_runtime.agent_records import RuntimeAgentJournal, RuntimeDocuments
from tests.agent_provider.support import materials
from tests.counterfactual.support import candidate, fixture, run
from tests.memory.support import second_case
from tests.memory.test_writer import clone, cmd
from tests.memory_review.support import conditions, policy
from tests.memory_review.test_postgres import approve as approve_legacy
from tests.planner.test_actions import ranked_binding
from tests.research_task_completion.support import installed
from tests.research_task_completion.test_postgres import approve as approve_task
from tests.research_task_completion.test_postgres import bound


def query_binding(*, start, **identity):
    """Validate the declared query budget together with its absolute deadline."""
    profile = Profile(seconds=3600)
    return ranked_binding(
        RankingPolicy(),
        profile=profile,
        started_at=start,
        deadline=start + timedelta(seconds=profile.seconds),
        **identity,
    )


def prepare_query(x):
    """Prebind the same task as supports without executing the query tool."""
    from db.runtime.repository.case_evidence import CaseEvidenceRepository
    from db.runtime.repository.planner import PlannerRepository

    driver, request, _ = installed(x.root / "query", ready=x.ready, identity="query-held-out")
    driver.research.create(request.code, request.version, request.checksum, "query-context")
    context = driver.research.read(request.lock.run_id)
    plan = driver.research.compile(context.run_id, driver.choice)
    executor, scores = driver.bind_plan(plan, "query-task:tool")
    definition = driver.task_completion.definition(
        context, plan, executor, scores, "query-task:tool"
    )
    assert definition.profile == x.supports[0].proof.definition.profile
    repo = CaseEvidenceRepository(driver.runtime._target)
    sessions = PlannerRepository(driver.runtime._target)
    start, sha = sessions.origin(context.run_id)
    original = driver.runtime.get_case_bundle(request.scope.case_code).artifacts[0]
    raw = canonical(dict(schema_version="case-evidence-graph/1.19.0", synthetic=True))
    store = x.ready.f.objects
    body = original.model_dump(mode="json")
    body.update(
        asset_code="ASSET:OID_ABLATION_ROOT",
        artifact_type_code="ARTIFACT:CASE_EVIDENCE_GRAPH",
        uri=store.promote(store.stage(raw), checksum(raw)),
        checksum=checksum(raw),
        media_type="application/json",
        derived_from=[original.asset_code],
        dimensions=[],
        measurements=[],
        research_only=True,
    )
    body["provenance"].update(
        input_checksums=[original.checksum], source_codes=[original.asset_code]
    )
    root = Artifact.model_validate(body)
    repo.register_observation_bundle((root,), "TASK:OID_ABLATION_ROOT")
    binding = query_binding(
        case_code=request.scope.case_code,
        run_id=context.run_id,
        run_context_checksum=sha,
        start=start,
        graph=Graph(
            role="graph",
            code=root.asset_code,
            version="case-evidence-graph/1.19.0",
            checksum=root.checksum,
            kind="research_artifact",
        ),
    )
    config = x.cf_base.config
    from agents.provider_agents import provider_locks

    locks = {ref.role: ref for ref in provider_locks(config)}
    binding = binding.model_copy(
        update={"locks": tuple(locks.get(r.role, r) for r in binding.locks)}
    )
    reader, _ = materials(binding, config=config)
    documents = RuntimeDocuments(
        repo=repo,
        contexts=x.contexts,
        sessions=sessions,
        store=store,
        run_id=context.run_id,
        root=root,
    )
    return SimpleNamespace(
        binding=binding,
        documents=documents,
        repo=repo,
        root=root,
        store=store,
        reader=reader,
        config=config,
        driver=driver,
        definition=definition,
        budget=RequestBudget(
            run_id=context.run_id,
            deadline=binding.deadline,
            max_requests=config.max_requests,
            journal=RuntimeAgentJournal(documents),
        ),
    )


def align_source_binding(pg):
    """Restore the unused source Planner to its original durable request budget."""
    if pg.budget.used or pg.budget.records():
        raise ValueError("source_review_budget_already_used")
    seconds = (pg.budget.deadline - pg.binding.started_at).total_seconds()
    assert seconds.is_integer() and seconds > 0
    pg.binding = pg.binding.model_copy(
        update={
            "deadline": pg.budget.deadline,
            "profile": pg.binding.profile.model_copy(update={"seconds": int(seconds)}),
        }
    )


def prepare_other_sources(x):
    """Two actual 08 acceptance/09 simulations, plus original preparation failures."""
    pgs = [x.cf_base, second_case(x.cf_base)]
    cf_episodes, recovery_episodes, reports = [], [], []
    x.simulation_fake_calls = 0
    for index, pg in enumerate(pgs):
        align_source_binding(pg)
        sim = fixture(pg=pg)
        x.simulation_fake_calls += len(sim.f.provider.transport.calls)
        intervention = candidate(targets=tuple(i.id for i in sim.parent.policy.items), id="remove")
        report = run(sim, intervention)["report"]
        reports.append(report)
        sim.h.planner.step(pg.binding.run_id, cmd(sim.h.planner, pg.binding.run_id, "stop"))
        research = ResearchSources(
            fact_reader=sim.reader,
            applicability=("USE:SYNTHETIC",),
            simulation=sim.service,
            documents=pg.documents,
        )
        source = ExecutionSources(
            runtime=x.contexts,
            run_id=pg.binding.run_id,
            applicability=("USE:SYNTHETIC",),
            planner=sim.h.planner,
            research=research,
        )
        x.registry.owners[pg.binding.run_id] = source
        writer = MemoryWriter(repository=x.content_writer, sources=source)
        cf_episodes.append(
            writer.write(
                run_id=pg.binding.run_id,
                request_key=f"cf-{index}",
                supplemental=(("simulation", "remove", "counterfactual_report"),),
            )
        )
        host = SimpleNamespace(
            m=SimpleNamespace(pg=pg, writer=x.content_writer, registry=x.registry),
            runtime=x.contexts,
            rid=pg.binding.run_id,
        )
        failed = clone(host, f"ablation-recovery-{index}")
        x.contexts.reject_preparation(
            failed.context, f"failure-{index}", digest("synthetic_failure")
        )
        recovery_episodes.append(
            failed.writer.write(run_id=failed.rid, request_key=f"recovery-{index}")
        )
    x.simulation_reports = reports
    return {
        "success": x.candidate,
        "recovery": x.distiller.write(
            request_key="ablation-recovery",
            episode_keys=tuple(e.record.key for e in recovery_episodes),
        ),
        "counterfactual": x.distiller.write(
            request_key="ablation-cf", episode_keys=tuple(e.record.key for e in cf_episodes)
        ),
    }


def publish(x, group, record):
    if group == "success":
        binding = bound(x)
    else:
        binding = Binding(
            target=record.record.key,
            target_checksum=record.checksum,
            content_version=record.record.schema_version,
            conditions=conditions(),
            group=group,
            preferred_branches=("B", "ILLEGAL"),
            terms=("task_complete",),
        )
    fake_review(x, group, binding)
    if group == "success":
        p, opinion, publication = approve_task(x, "ablation-success", binding)
    else:
        p = policy("ablation-" + group + "-policy", x.reviewer.target.user).model_copy(
            update={"test_context": "10e-task-owned"}
        )
        p, opinion, publication = approve_legacy(x, "ablation-" + group, bound=binding, p=p)
    x.publisher.publish(publication)
    return publication


def fake_review(x, group, binding):
    import json
    from datetime import UTC, datetime

    from packages.llm_provider import DeterministicFakeProvider
    from packages.memory_review.models import Advice, ReviewInput
    from packages.memory_review.service import MemoryReviewAgent
    from packages.memory_review.task_models import TaskBinding, TaskReviewInput

    pg = x.query_base
    scope = pg.reader.current().material.scope
    model = TaskReviewInput.project if isinstance(binding, TaskBinding) else ReviewInput
    payload = model(
        scope=scope.model_dump(mode="json"),
        binding=binding,
        sources=tuple(sorted(digest(s) for s in x.reader.read(binding.target).record.sources)),
    )
    advice = Advice(
        target_checksum=binding.target_checksum,
        binding_checksum=digest(binding),
        decision="candidate_supported",
        reasons=("synthetic_engineering_only",),
    )
    # One deterministic reply for this exact input; no group-conditioned answers.
    x.fake_mapping[digest(payload)] = advice.model_dump(mode="json")

    def response(body):
        actual = json.loads(body["messages"][1]["content"][0]["text"])
        return x.fake_mapping[digest(actual)]

    provider = DeterministicFakeProvider(pg.config, replies=(response,))
    agent = MemoryReviewAgent(
        repository=x.writer,
        provider=provider,
        budget=pg.budget,
        documents=pg.documents,
        scope=scope,
    )
    now = datetime.now(UTC)
    opinion = agent.review(
        key="ablation-fake-" + group,
        binding=binding,
        valid_from=now,
        expires_at=now + timedelta(minutes=30),
    )
    assert opinion.kind == "agent" and len(provider.transport.calls) == 1
    x.fake_calls += len(provider.transport.calls)


def query_branch(x, group):
    pg = x.query_base
    host = SimpleNamespace(
        m=SimpleNamespace(pg=pg, writer=x.content_writer, registry=x.registry),
        runtime=x.contexts,
        rid=pg.binding.run_id,
    )
    branch = clone(host, "ablation-query-" + group, planner=True, profile={"seconds": 3600})
    request = branch.context.request
    x.ready.f.owners.requests[(request.code, request.version, request.checksum)] = request
    # Recompute the query task profile from the actual registered current plan.
    driver = pg.driver
    plan = driver.research.compile(branch.rid, driver.choice)
    executor, scores = driver.bind_plan(plan, "query-" + group + ":tool")
    definition = driver.task_completion.definition(
        branch.context, plan, executor, scores, "query-" + group + ":tool"
    )
    assert definition.profile == pg.definition.profile
    branch.definition = definition
    branch.documents = RuntimeDocuments(
        repo=pg.repo,
        contexts=x.contexts,
        sessions=pg.documents.sessions,
        store=pg.store,
        run_id=branch.rid,
        root=pg.root,
    )
    return branch
