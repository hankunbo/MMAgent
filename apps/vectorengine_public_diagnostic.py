"""Independent public compatibility stage, original Runtime/Planner ownership only.

The debug context uses the existing local CASE storage root, but installs no
research-live grant or patient action pipeline. Only CASE:PUBLIC_PROBE materials
can reach the Provider. Its stopped journal qualifies one later case allocation.
"""

import json
from datetime import timedelta
from pathlib import Path

from agents.planner.service import Planner
from agents.provider_agents import StructuredRoleAgent
from apps.vectorengine_live_assembly import make_binding, stop_failed_run
from apps.vectorengine_round import (
    check_round_gate,
    evidence_for_round,
    load_round,
    read_deployment,
)
from apps.vectorengine_single_case import open_case, write_once
from db.runtime.repository.planner import PlannerRepository
from db.runtime.repository.research import ResearchRunRepository
from packages.evidence_scoring.accepted import build_accepted
from packages.grounding.models import digest, seal, validate
from packages.llm_provider.bounded_live import BoundedLiveBudget
from packages.llm_provider.live_allocation import StageAllocation
from packages.llm_provider.live_probes import PublicProbeMaterials
from packages.llm_provider.provider import ProviderError, StructuredProvider
from packages.llm_provider.public_shape import PublicReviewMaterials
from packages.planner_control.models import Binding, Command
from packages.research_runtime.agent_records import RuntimeAgentJournal, RuntimeDocuments
from packages.research_runtime.models import ResearchRunContext
from packages.tool_admission.models import ResearchRequest


class PublicDiagnosticSnapshot:
    """Original qualified local graph only; no patient action or acceptance ports."""

    def __init__(self, evidence, initial):
        self.evidence, self.initial = evidence, initial

    def current(self, binding):
        material = self.initial.material
        if (binding.run_id, binding.input_checksum, binding.visible_checksum) != (
            material.scope.run_id,
            material.scope.input_checksum,
            material.scope.visible_checksum,
        ):
            raise ValueError("public_control_scope")
        self.evidence.verify(self.initial)
        return material.scope.graph


def original_documents(evidence, target, run_id):
    return RuntimeDocuments(
        repo=evidence.runtime,
        contexts=ResearchRunRepository(target),
        sessions=PlannerRepository(target),
        store=evidence.objects,
        run_id=run_id,
        root=evidence.root,
    )


def public_step(*, name, role, provider, budget, documents, materials):
    key = "compat:" + name
    saved = documents.read(key + ":result")
    if saved:
        if saved[0]["status"] != "passed":
            raise ProviderError(saved[0]["error"])
        return saved[0]
    with documents.session():
        budget.before_reserve(digest(provider.config))
        if documents.read(key + ":claim"):
            raise ValueError("public_attempt_unknown_no_repeat")
        documents.write(
            key + ":claim", (dict(purpose=name, config_checksum=digest(provider.config)),)
        )
    try:
        candidate = StructuredRoleAgent(
            role, provider=provider, budget=budget, materials=materials
        ).call(
            materials.state.material.scope,
            deadline=budget.deadline,
            round_id="A" if role == "review" else None,
        )
        if role == "vision":
            if candidate.status != "observed" or not any(
                t in candidate.finding.lower() for t in ("red", "红")
            ):
                raise ProviderError("scope")
        else:
            build_accepted(materials.state, candidate)
        result = dict(
            status="passed",
            purpose=name,
            role=role,
            candidate=candidate,
            request_id=budget.records()[-1].request_id,
            config_checksum=digest(provider.config),
            image_bytes=len(materials.raw) if role == "vision" else 0,
            task_count=len(materials.state.policy.items) if role == "review" else 0,
            remote_constraint="strict_json_schema",
            patient_material=False,
        )
    except Exception as error:
        result = dict(
            status="failed",
            purpose=name,
            role=role,
            error=error.code if isinstance(error, ProviderError) else type(error).__name__,
            request_id=budget.records()[-1].request_id if budget.records() else None,
        )
        with documents.session():
            documents.write(key + ":result", (result,))
        raise
    with documents.session():
        documents.write(key + ":result", (result,))
    return result


def main_diagnostic(round_config, config, *, execute):
    original, target = open_case()
    evidence = evidence_for_round(original, round_config)
    deployment = read_deployment(evidence, target, round_config.run_id)
    initial, _ = evidence.project(run_id=round_config.run_id, config=config)
    output = Path(round_config.output)
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_once(output / ("deployment-" + digest(deployment)[7:23] + ".json"), deployment)
    if not execute:
        write_once(
            output / "public-state.json",
            PublicReviewMaterials(
                round_config.run_id,
                config,
                task_count=config.review_batching.batch_size
                if getattr(config, "review_batching", None)
                else 45,
            ).state,
        )
        if getattr(config, "review_batching", None):
            from packages.llm_provider.public_batches import public_plan

            write_once(output / "public-batch-plan.json", public_plan(round_config.run_id, config))
        print(json.dumps(dict(status="public_prepared_no_requests", run_id=round_config.run_id)))
        return 0
    check_round_gate(round_config)
    contexts, sessions = ResearchRunRepository(target), PlannerRepository(target)
    current = sessions.read(round_config.run_id)
    if current is not None:
        Planner(runtime=sessions, owner=None).require_agent_work(current.binding)
    old = contexts.read_research_run(evidence.view["case_evidence"]["fairclip"]["run_id"])
    raw = old.request.model_dump(mode="json", exclude={"checksum"})
    raw["code"], raw["lock"]["run_id"], raw["purpose"] = (
        round_config.request_code,
        round_config.run_id,
        "debug",
    )
    request = seal(ResearchRequest, **raw)
    context = seal(
        ResearchRunContext, request=request, synthetic=False, scope_checksum=digest(request.scope)
    )
    contexts.create_research_run(context, round_config.request_key)
    started, context_sha = sessions.origin(context.run_id)
    binding, _ = make_binding(
        initial=initial,
        config=config,
        started=started,
        context_checksum=context_sha,
        catalog=request.lock.catalog_ref,
    )
    binding = validate(
        Binding,
        binding.model_copy(
            update={
                "profile": binding.profile.model_copy(update={"seconds": 1200}),
                "deadline": started + timedelta(seconds=1200),
            }
        ),
    )
    documents = original_documents(evidence, target, context.run_id)
    plan = None
    if getattr(config, "wire_policy", None) and not getattr(config, "review_batching", None):
        from packages.llm_provider.public_compatibility import load_plan

        plan = load_plan(round_config)
    allocation = StageAllocation(
        session_id=round_config.session_id,
        stage="public",
        public_run_id=context.run_id,
        public_used=0,
        public_journal_checksum=None,
        configuration_checksums=tuple(c.config_checksum for c in plan.candidates)
        if plan
        else (digest(config),),
    )
    if round_config.schema_version in {
        "bounded-live-round/6.0.0",
        "bounded-live-round/7.0.0",
        "bounded-live-round/10.0.0",
        "bounded-live-round/11.0.0",
    }:
        from apps.vectorengine_integrated_gate import historical_usage
        from packages.llm_provider.live_allocation import StageAllocationV3

        allocation = StageAllocationV3(
            **allocation.model_dump(exclude={"schema_version"}),
            history=historical_usage(round_config, evidence, target),
        )
    budget = BoundedLiveBudget(
        binding=binding,
        config=config,
        journal=RuntimeAgentJournal(documents),
        allocation=allocation,
        diagnostic_plan=plan,
    )
    provider = StructuredProvider(config, live_opt_in=True)
    planner = Planner(runtime=sessions, owner=PublicDiagnosticSnapshot(evidence, initial))
    import signal
    from threading import Event

    cancel = Event()
    budget.cancel_signal = cancel
    handlers = {
        s: signal.signal(s, lambda *_: cancel.set()) for s in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        if current is None:
            planner.begin(binding)
        if getattr(config, "review_batching", None):
            from packages.llm_provider.public_batches import run_public_batches

            qualification = run_public_batches(config, budget, documents, digest(round_config))
        elif plan is not None:
            from packages.llm_provider.public_compatibility import run_comparisons

            qualification = run_comparisons(plan, budget, documents, digest(round_config))
        else:
            results = []
            for name, role, materials in (
                ("full_vision", "vision", PublicProbeMaterials(context.run_id, config)),
                ("review_45", "review", PublicReviewMaterials(context.run_id, config)),
            ):
                results.append(
                    public_step(
                        name=name,
                        role=role,
                        provider=provider,
                        budget=budget,
                        documents=documents,
                        materials=materials,
                    )
                )
            qualification = dict(
                schema_version="bounded-public-qualification/1.0.0",
                session_id=round_config.session_id,
                run_id=context.run_id,
                round_checksum=digest(round_config),
                config_checksum=digest(config),
                roles=results,
                source_reused="docs/acceptance/12-live-round3-v1/actual/result-summary.json",
                source_wire_unchanged=True,
                patient_requests=0,
            )
            documents.write("public-qualification", (qualification,))
        state = planner.read(context.run_id)
        planner.step(
            context.run_id,
            Command(key="public-qualified-stop", action="stop", expected_revision=state.revision),
        )
        write_once(output / "qualification.json", qualification)
        print(
            json.dumps(dict(status="public_qualified", run_id=context.run_id, requests=budget.used))
        )
        return 0
    except Exception as error:
        failure = stop_failed_run(
            planner=planner,
            sessions=sessions,
            binding=binding,
            budget=budget,
            error=error,
            cancel=cancel,
        )
        write_once(output / "stop.json", failure)
        print(
            json.dumps(
                dict(status="public_bounded_stop", reason=failure["reason"], requests=budget.used)
            )
        )
        return 2
    finally:
        for s, h in handlers.items():
            signal.signal(s, h)
        facts = dict(
            records=budget.records(),
            summary=budget.summary(),
            transport_counts=budget.transport_counts(),
            journal=budget.journal.read(),
        )
        write_once(output / ("requests-" + digest(facts)[7:23] + ".json"), facts)


def consume_qualification(round_config, evidence, target, initial=None):
    prior_usages = ()
    history = ()
    public_config = load_round(round_config.public_config)
    if public_config.stage != "public" or public_config.session_id != round_config.session_id:
        raise ValueError("public_session_binding")
    if round_config.schema_version in {
        "bounded-live-round/6.0.0",
        "bounded-live-round/7.0.0",
        "bounded-live-round/8.0.0",
        "bounded-live-round/10.0.0",
        "bounded-live-round/11.0.0",
    }:
        from apps.vectorengine_integrated_gate import historical_usage

        if round_config.schema_version == "bounded-live-round/8.0.0":
            from apps.vectorengine_performance_gate import check_public_baseline

            check_public_baseline(public_config)
        else:
            check_round_gate(public_config)
        history = historical_usage(round_config, evidence, target)
    elif round_config.schema_version == "bounded-live-round/5.0.0":
        from apps.vectorengine_case_quote_gate import check_public_baseline, prior_case_usage
        from apps.vectorengine_case_target_gate import check_prior_case

        check_public_baseline(public_config)
        prior = load_round(round_config.prior_case_config)
        original = load_round(prior.prior_case_config)
        check_prior_case(original_documents(evidence, target, original.run_id))
        prior_usages = (
            prior_case_usage(original_documents(evidence, target, prior.run_id), prior),
        )
    elif round_config.schema_version == "bounded-live-round/4.0.0":
        from apps.vectorengine_case_target_gate import check_prior_case, check_public_baseline

        check_public_baseline(public_config)
        prior = load_round(round_config.prior_case_config)
        check_prior_case(original_documents(evidence, target, prior.run_id))
    else:
        check_round_gate(public_config)
    documents = original_documents(evidence, target, public_config.run_id)
    state = documents.sessions.read(public_config.run_id)
    saved = documents.read("public-qualification")
    journal = RuntimeAgentJournal(documents).read()
    if state is None or state.phase != "stopped" or state.pending or not saved:
        raise ValueError("public_qualification_missing")
    qualification = saved[0]
    if qualification.get("schema_version") == "bounded-public-qualification/3.0.0":
        from apps.vectorengine_single_case import load_config
        from packages.evidence_details.review_tasks import project_review_tasks
        from packages.llm_provider.provider import RequestBudget
        from packages.llm_provider.public_batches import case_estimate, check_public_batches

        config = load_config(round_config.provider_config)
        if initial is None or digest(config) != public_config.provider_checksum:
            raise ValueError("public_batch_case_configuration")
        public_budget = RequestBudget(
            run_id=public_config.run_id,
            deadline=state.binding.deadline,
            max_requests=40,
            journal=RuntimeAgentJournal(documents),
        )
        durations = check_public_batches(
            config,
            public_budget,
            documents,
            qualification,
            digest(public_config),
            require_live=True,
        )
        estimate = case_estimate(
            len(project_review_tasks(initial).tasks),
            config,
            public_budget.used + sum(p.used for p in prior_usages + history),
            durations,
        )
        if not estimate["feasible"]:
            raise ValueError("batch_case_budget_or_duration_not_feasible")
    elif qualification.get("schema_version") == "bounded-public-qualification/2.0.0":
        from packages.llm_provider.public_compatibility import check_qualification, load_plan

        plan = load_plan(public_config)
        if qualification["round_checksum"] != digest(public_config):
            raise ValueError("public_qualification_round")
        check_qualification(plan, qualification, documents, round_config.provider_checksum)
    else:
        if (
            qualification["config_checksum"] != round_config.provider_checksum
            or qualification["round_checksum"] != digest(public_config)
            or [r["role"] for r in qualification["roles"]] != ["vision", "review"]
            or any(r["status"] != "passed" for r in qualification["roles"])
        ):
            raise ValueError("public_qualification_binding")
    used = sum(e.get("phase") == "reserved" for e in journal)
    from packages.llm_provider.live_allocation import StageAllocationV2

    allocation = (StageAllocationV2 if prior_usages else StageAllocation)(
        **(dict(prior_cases=prior_usages) if prior_usages else {}),
        session_id=round_config.session_id,
        stage="case",
        public_run_id=public_config.run_id,
        public_used=used,
        public_journal_checksum=digest(journal),
        configuration_checksums=(round_config.provider_checksum,),
    )
    if history:
        from packages.llm_provider.live_allocation import StageAllocationV3

        allocation = StageAllocationV3(
            **allocation.model_dump(exclude={"schema_version"}), history=history
        )
    result = dict(qualification=qualification, public_requests_reused_not_sent_in_case=used)
    if prior_usages:
        result["prior_case_requests_charged_not_replayed"] = sum(p.used for p in prior_usages)
    return result, allocation
