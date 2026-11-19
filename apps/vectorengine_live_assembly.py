"""Direct bounded assembly of original 07/08 owners, without servers or tool execution."""

import json
import signal
from datetime import timedelta
from pathlib import Path
from threading import Event

from agents.expert_agent.actions import ResearchActionSource
from agents.expert_agent.models import ReviewFacts, ReviewPolicy
from agents.planner.service import Planner
from agents.provider_agents import provider_locks
from apps.agent_provider_assembly import assemble_provider_roles
from apps.expert_agent_assembly import assemble_research_review
from apps.vectorengine_case_materials import ref
from apps.vectorengine_single_case import write_once
from packages.evidence_details.acceptance_models import Submission
from packages.evidence_details.live_permission import ResearchLiveGrant, ResearchLivePermission
from packages.evidence_details.successor_owner import ResearchSuccessorOwner
from packages.grounding.models import digest, seal, validate
from packages.llm_provider.bounded_live import BoundedLiveBudget
from packages.llm_provider.live_probes import run_probes
from packages.planner_control.actions import RankedActions, SavedResearchActions
from packages.planner_control.models import Binding, Command, Outcome, Profile, RankingPolicy
from packages.research_runtime.agent_records import RuntimeAgentJournal, RuntimeDocuments
from packages.research_runtime.live_snapshots import LiveSnapshots
from packages.research_runtime.planner_steps import ResearchPlannerSteps


class LiveResearchSteps(ResearchPlannerSteps):
    def invoke(self, state, pending, *, cancel):
        result = super().invoke(state, pending, cancel=cancel)
        if pending.command.action == "identify":
            # The additive ResearchActionSource below really installs these candidate ports.
            return validate(Outcome, result.model_copy(update={"gaps": ()}))
        return result


def make_binding(*, initial, config, started, context_checksum, catalog):
    ranking = RankingPolicy(max_agent_attempts=1, suggestion_seconds=120)
    domain = initial.material
    locks = dict(
        knowledge=ref("knowledge", "POLICY:RESEARCH_SOURCES", digest(domain.sources)),
        template=ref("template", domain.policy_ref.code, domain.policy_ref.checksum),
        rule=ref("rule", "POLICY:RESEARCH_TASKS", digest(initial.policy)),
        catalog=ref("catalog", catalog.code, catalog.checksum, catalog.version),
        mapping=ref("mapping", "POLICY:REUSED_OBSERVATIONS", digest(initial.resources)),
        execution=ref("execution", "POLICY:NO_TOOL_EXECUTION", digest(dict(tool_execution=False))),
        ranking=ref("ranking", "POLICY:RANKING", digest(ranking), ranking.version),
        sources=ref("sources", "POLICY:RESEARCH_SOURCES", digest(domain.sources)),
    )
    locks.update({r.role: r for r in provider_locks(config)})
    return Binding(
        case_code=domain.scope.case_code,
        run_id=domain.scope.run_id,
        run_context_checksum=context_checksum,
        input_checksum=domain.scope.input_checksum,
        visible_checksum=domain.scope.visible_checksum,
        graph=domain.scope.graph,
        locks=tuple(locks.values()),
        profile=Profile(
            max_steps=20,
            max_actions=2,
            max_branches=2,
            max_retries=1,
            max_no_progress=2,
            seconds=2700,
        ),
        started_at=started,
        deadline=started + timedelta(seconds=2700),
    ), ranking


def assemble(
    *,
    evidence,
    initial,
    config,
    documents,
    binding,
    ranking,
    allocation=None,
    source_quote_policy="research-quote-policy/literal/1.0.0",
    budget_factory=None,
):
    journal = RuntimeAgentJournal(documents)
    budget = (budget_factory or BoundedLiveBudget)(
        binding=binding, config=config, journal=journal, allocation=allocation
    )
    grant = ResearchLiveGrant(
        enabled=True,
        case_code=binding.case_code,
        run_id=binding.run_id,
        input_checksum=binding.input_checksum,
        visible_checksum=binding.visible_checksum,
        policy_checksum=initial.material.scope.policy_checksum,
        config_checksum=digest(config),
        root_checksum=documents.root.checksum,
        purpose="bounded_single_case_research",
        deadline=binding.deadline,
        roles=("source", "vision", "review"),
    )
    permission = ResearchLivePermission(
        grant=grant, config=config, documents=documents, budget=budget
    )
    owner = ResearchSuccessorOwner(
        source_quote_policy=source_quote_policy,
        initial=initial,
        documents=documents,
        verify_dependencies=evidence.verify,
        read_image=evidence.image,
        config=config,
        role="vision",
        live_permission=permission,
    )
    if callable(getattr(evidence, "wrap_owner", None)):
        owner = evidence.wrap_owner(owner)
    if getattr(evidence, "targeted_acquisition", False):
        from apps.active_evidence_materials import ActiveMaterialReader

        owner = ActiveMaterialReader(owner, evidence, documents)
    ports = assemble_provider_roles(
        binding=binding,
        materials=owner,
        journal=journal,
        config=config,
        live_opt_in=True,
        budget=budget,
    )
    snapshots = LiveSnapshots(owner)
    source = ResearchActionSource(
        SavedResearchActions(snapshots),
        owner,
        roles=("source", "vision", "review")
        if getattr(config, "review_batching", None)
        else ("vision", "review"),
        phases=("select", "review"),
    )
    steps = LiveResearchSteps(snapshots=snapshots)
    if getattr(evidence, "targeted_acquisition", False):
        from packages.evidence_acquisition.planner import AcquisitionPlannerOwner

        steps = AcquisitionPlannerOwner(steps, owner, evidence.requirements)
    planner = Planner(
        runtime=documents.sessions,
        owner=steps,
        ranking=RankedActions(source, policy=ranking, agent=ports.trace),
    )
    protocol = json.loads(Path("configs/research/evidence_review_protocol_v1.json").read_text())
    review = assemble_research_review(
        planner=planner,
        run_id=binding.run_id,
        domain_owner=owner,
        documents=documents,
        ports=ports,
        action_source=source,
        ranking_policy=ranking,
        review_policy=ReviewPolicy(max_no_progress=2, comparison_protocol=digest(protocol)),
        facts=lambda s: ReviewFacts(),
        install=owner.install,
        protocol=protocol,
        live_permission=permission,
    )
    return planner, owner, ports, review


def check_regression_gate():
    from packages.object_store import checksum

    path = Path("docs/acceptance/12-live-vectorengine-v1/execution-gate.json")
    gate = json.loads(path.read_text())
    if gate.get("synthetic_regression_passed") is not True:
        raise ValueError("live_regression_gate_required")
    for name, sha in gate["implementation"].items():
        if checksum(Path(name).read_bytes()) != sha:
            raise ValueError("live_regression_gate_drift")


def stop_failed_run(*, planner, sessions, binding, budget, error, cancel):
    """Keep the primary failure even when the installed cleanup owner is unavailable."""
    from packages.llm_provider.provider import ProviderError

    current = None
    cleanup_error = None
    try:
        current = sessions.read(binding.run_id)
        if current is not None and current.pending is None and current.phase != "stopped":
            planner.step(
                binding.run_id,
                Command(key="live-error-stop", expected_revision=current.revision, action="stop"),
                cancel=cancel,
            )
    except Exception as cleanup:
        cleanup_error = type(cleanup).__name__
    return dict(
        schema_version="bounded-live-stop/1.0.0",
        run_id=binding.run_id,
        reason=error.code
        if isinstance(error, ProviderError)
        else "owner_or_confirmation_unresolved",
        error_type=type(error).__name__,
        cleanup_error_type=cleanup_error,
        requests=budget.summary(),
        pending_preserved=bool(current and current.pending),
        planner_read_confirmed=cleanup_error is None,
        clinical_acceptance=False,
        new_tool_executions=0,
    )


def export_stage_report(*, review, documents, output, role, event, result, verify_existing=False):
    """Historical export is repeatable even if its earlier file acknowledgement was lost."""
    from packages.multimodal_research.report_io import publish_html, publish_json

    if result.get("acceptance_key"):
        from agents.expert_agent.reporting import read_research_report, render_research_html

        intent = documents.read("review:" + event.code + ":intent")[0]
        from packages.evidence_details.acceptance_models import DomainState

        report = read_research_report(
            review.acceptance,
            key=result["acceptance_key"],
            parent=validate(DomainState, intent["state"]),
        )
        publish_json(output / (role + "-report.json"), report)
        html_path = output / (role + "-report.html")
        if (
            verify_existing
            and html_path.exists()
            and html_path.read_text() != render_research_html(report)
        ):
            raise ValueError("research_html_export_conflict")
        if not html_path.exists():
            publish_html(html_path, render_research_html(report))


def run_case_stages(
    *,
    planner,
    owner,
    ports,
    review,
    documents,
    binding,
    config,
    output,
    signal_cancel,
    selected_roles=None,
    before_step=None,
):
    """Original bounded Source -> Vision -> Review continuation; shared by CLI and PG gate."""
    completed = []
    roles = (
        ("source", "vision", "review")
        if getattr(config, "review_batching", None)
        else ("vision", "review")
    )
    if selected_roles is not None:
        if (
            not selected_roles
            or len(set(selected_roles)) != len(selected_roles)
            or any(role not in roles for role in selected_roles)
        ):
            raise ValueError("live_selected_role_scope")
        roles = tuple(selected_roles)
    for role in roles:
        saved = documents.read("live-stage:" + role)
        if saved:
            if getattr(config, "review_batching", None):
                from packages.planner_control.models import Reference

                intent = documents.read("live-stage-intent:" + role)
                if intent is None:
                    raise ValueError("saved_stage_intent_missing")
                export_stage_report(
                    review=review,
                    documents=documents,
                    output=output,
                    role=role,
                    event=validate(Reference, intent[0]["event"]),
                    result=saved[0],
                    verify_existing=True,
                )
            completed.append(saved[0])
            if saved[0]["status"] != "accepted":
                break
            continue
        owner.role = role
        if hasattr(owner, "acquisition_checkpoint"):
            write_once(
                output / ("acquisition-before-" + role + ".json"),
                owner.acquisition_checkpoint("before-" + role),
            )
        saved_intent = (
            documents.read("live-stage-intent:" + role)
            if getattr(config, "review_batching", None)
            else None
        )
        if saved_intent:
            from packages.planner_control.models import Reference

            intent = saved_intent[0]
            if (intent["role"], intent["config_checksum"]) != (role, digest(config)):
                raise ValueError("live_stage_resume_binding")
            event = validate(Reference, intent["event"])
            proposed_action_id = intent["proposed_action_id"]
        else:
            state = planner.read(binding.run_id)
            actions = {
                "initialize": "load",
                "retrieve": "retrieve",
                "instantiate": "instantiate",
                "identify": "identify",
                "select": "select",
                "prepare": "prepare",
                "execute": "execute",
                "verify": "verify",
            }
            while state.phase not in {"select", "review", "stopped"}:
                if before_step is not None:
                    before_step()
                cmd = (
                    state.pending.command
                    if state.pending
                    else Command(
                        key="live-step:" + str(state.revision),
                        expected_revision=state.revision,
                        action=actions[state.phase],
                    )
                )
                planner.step(binding.run_id, cmd, cancel=signal_cancel)
                state = planner.read(binding.run_id)
                if state.pending:
                    raise ValueError("planner_unknown_requires_original_recovery")
            if state.phase == "stopped":
                break
            if before_step is not None:
                before_step()
            cmd = (
                state.pending.command
                if state.pending
                else Command(
                    key="live-review:" + role, expected_revision=state.revision, action="review"
                )
            )
            event = planner.begin_recovery_review(binding.run_id, cmd, cancel=signal_cancel)
            frame = review.context.read(event)
            action = next(a for a in frame.actions.actions if a.branch == "research_" + role)
            if getattr(config, "review_batching", None):
                documents.write(
                    "live-stage-intent:" + role,
                    (
                        dict(
                            schema_version="live-stage-intent/1.0.0",
                            role=role,
                            config_checksum=digest(config),
                            event=event.model_dump(mode="json"),
                            proposed_action_id=action.id,
                        ),
                    ),
                )
            proposed_action_id = action.id
        if before_step is not None:
            before_step()
        result = review.run(
            event=event,
            proposed_action_id=proposed_action_id,
            role=role,
            abc=False,
            cancel=signal_cancel,
        )
        documents.write("live-stage:" + role, (result,))
        completed.append(result)
        if hasattr(owner, "acquisition_checkpoint"):
            write_once(
                output / ("acquisition-after-" + role + ".json"),
                owner.acquisition_checkpoint("after-" + role),
            )
        export_stage_report(
            review=review,
            documents=documents,
            output=output,
            role=role,
            event=event,
            result=result,
            verify_existing=bool(getattr(config, "review_batching", None)),
        )
        if result["status"] != "accepted":
            break
    return completed, roles


def execute(*, evidence, target, initial, config, output, round_config=None):
    from apps.vectorengine_case_preflight import check_case_admissibility

    # Public callers cannot bypass preparation by directly invoking the live assembler.
    # No deployed owner, qualification query, run creation or request before this check.
    check_case_admissibility(initial, config)
    if round_config is None:
        check_regression_gate()
    else:
        from apps.vectorengine_round import check_round_gate

        if initial.material.scope.run_id != round_config.run_id:
            raise ValueError("round_run_binding")
        check_round_gate(round_config, initial)
    from db.runtime.repository.planner import PlannerRepository
    from db.runtime.repository.research import ResearchRunRepository
    from packages.research_runtime.models import ResearchRunContext
    from packages.tool_admission.models import ResearchRequest

    qualification, allocation = None, None
    if round_config is not None and getattr(round_config, "stage", None) == "case":
        from apps.vectorengine_public_diagnostic import consume_qualification

        qualification, allocation = consume_qualification(round_config, evidence, target, initial)
    contexts, sessions = ResearchRunRepository(target), PlannerRepository(target)
    # Detect a missing deployed 08A owner before creating a run or making any API call.
    current = sessions.read(initial.material.scope.run_id)
    if current is not None:
        Planner(runtime=sessions, owner=None).require_agent_work(current.binding)
    old = contexts.read_research_run(evidence.view["case_evidence"]["fairclip"]["run_id"])
    request = old.request.model_dump(mode="json", exclude={"checksum"})
    request["code"] = (
        "RESEARCH:VECTOR_SINGLE_V1" if round_config is None else round_config.request_code
    )
    request["lock"]["run_id"] = initial.material.scope.run_id
    # Reference the original visible-input qualification only. No 06 executor/compiler is installed.
    request = seal(ResearchRequest, **request)
    context = seal(
        ResearchRunContext, request=request, synthetic=False, scope_checksum=digest(request.scope)
    )
    contexts.create_research_run(
        context, "vectorengine-single-case-v1" if round_config is None else round_config.request_key
    )
    started, context_checksum = sessions.origin(context.run_id)
    if current is not None:
        # A saved earlier absolute limit wins; recovering never grants a later deadline.
        started = min(started, current.binding.started_at)
    binding, ranking = make_binding(
        initial=initial,
        config=config,
        started=started,
        context_checksum=context_checksum,
        catalog=request.lock.catalog_ref,
    )
    documents = RuntimeDocuments(
        repo=evidence.runtime,
        contexts=contexts,
        sessions=sessions,
        store=evidence.objects,
        run_id=context.run_id,
        root=evidence.root,
    )
    planner, owner, ports, review = assemble(
        evidence=evidence,
        initial=initial,
        config=config,
        documents=documents,
        binding=binding,
        ranking=ranking,
        allocation=allocation,
        source_quote_policy=getattr(
            round_config, "source_quote_policy", "research-quote-policy/literal/1.0.0"
        ),
    )
    signal_cancel = Event()
    ports.budget.cancel_signal = signal_cancel
    previous = {
        s: signal.signal(s, lambda *_: signal_cancel.set()) for s in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        # The original write owner must also work before the first network reservation.
        if sessions.read(binding.run_id) is None:
            planner.begin(binding)
        probes = (
            qualification
            if qualification is not None
            else run_probes(
                provider=ports.expert.provider, budget=ports.budget, documents=documents
            )
        )
        write_once(output / "capability-probes.json", probes)
        # Independent source submission/revalidation does not make it a medical fact.
        source_saved = documents.read("live-source-candidate")
        if source_saved is None and not getattr(config, "review_batching", None):
            if documents.read("live-source-claim"):
                raise ValueError("source_unknown_no_repeat")
            ports.budget.before_reserve(digest(config))
            documents.write("live-source-claim", (dict(config_checksum=digest(config)),))
            owner.role = "source"
            state = owner.current()
            c = ports.source.call(state.material.scope, deadline=binding.deadline)
            r = ports.budget.records()[-1]
            sub = Submission(
                scope=state.material.scope,
                actor="PLANNER",
                correlation="SOURCE:AGENT_RESEARCH",
                request_id=r.request_id,
                prompt_checksum=r.prompt_checksum,
                config_checksum=r.config_checksum,
                observation_checksums=tuple(o.reference for o in state.material.observations),
                source_checksums=tuple(s.reference for s in state.material.sources),
                candidate=c,
            )
            from packages.knowledge_claims.research_candidates import submit_research_candidate

            result = submit_research_candidate(review.acceptance, sub, key="live-source")
            documents.write("live-source-candidate", (dict(submission=sub, receipt=result),))
        completed, roles = run_case_stages(
            planner=planner,
            owner=owner,
            ports=ports,
            review=review,
            documents=documents,
            binding=binding,
            config=config,
            output=output,
            signal_cancel=signal_cancel,
        )
        final_state = planner.read(binding.run_id)
        if final_state.phase != "stopped" and final_state.pending is None:
            planner.step(
                binding.run_id,
                Command(
                    key="live-bounded-stop", expected_revision=final_state.revision, action="stop"
                ),
                cancel=signal_cancel,
            )
        report = dict(
            schema_version="bounded-live-run-result/1.0.0",
            run_id=binding.run_id,
            capabilities=probes,
            stages=completed,
            request_summary=ports.budget.summary(),
            transport_counts=ports.budget.transport_counts(),
            planner=planner.report(binding.run_id),
            active_tool_evidence=False,
            new_tool_executions=0,
            clinical_acceptance=False,
        )
        write_once(output / "run-result.json", report)
        print(
            json.dumps(
                dict(
                    status="completed"
                    if len(completed) == len(roles)
                    and all(r["status"] == "accepted" for r in completed)
                    else "bounded_stop",
                    run_id=binding.run_id,
                    output=str(output),
                )
            )
        )
        return (
            0
            if len(completed) == len(roles) and all(r["status"] == "accepted" for r in completed)
            else 2
        )
    except Exception as error:
        failure = stop_failed_run(
            planner=planner,
            sessions=sessions,
            binding=binding,
            budget=ports.budget,
            error=error,
            cancel=signal_cancel,
        )
        write_once(output / ("stop-" + digest(failure)[7:23] + ".json"), failure)
        print(
            json.dumps(
                dict(
                    status="bounded_stop",
                    run_id=binding.run_id,
                    reason=failure["reason"],
                    output=str(output),
                )
            )
        )
        return 2
    finally:
        if signal_cancel.is_set():
            with ports.budget.journal.session():
                if not any(e.get("phase") == "live_cancelled" for e in ports.budget.journal.read()):
                    ports.budget.journal.append(dict(phase="live_cancelled"))
        for s, handler in previous.items():
            signal.signal(s, handler)
        # Observed request/usage facts survive any later owner or report failure.
        requests = dict(
            records=ports.budget.records(),
            summary=ports.budget.summary(),
            transport_counts=ports.budget.transport_counts(),
        )
        write_once(output / ("requests-" + digest(requests)[7:23] + ".json"), requests)
