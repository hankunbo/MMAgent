"""Original Generic executor composition for a capability-selected increment.

New dispatch requires a fresh, frozen successor and the original exclusive
parent claim. Saved results use the original zero-request report recovery.
"""

from apps.vectorengine_single_case import write_once
from packages.grounding.models import digest

from .generic_capability_successor import (
    CONTINUATION_CLAIM,
    CONTINUATION_VERSIONS,
    ROUTED_VERSION,
    VERSIONS,
    CapabilitySuccessor,
    origin,
    selection_for,
)
from .generic_definition_successor import CLAIM, read
from .generic_input import directory


def load_selection(key, *, history_software=None, _seen=()):
    if key in _seen or len(_seen) >= 4:
        raise ValueError("generic_capability_history_cycle_or_depth")
    root = directory(key)
    # Restoration may not silently create a missing execution window.
    for path in ("input.json", "binding.json", "started.json", "evidence/protocol.json"):
        if not (root / path).is_file():
            raise ValueError("generic_capability_saved_execution_required")
    value = read(root / "input.json")
    if value.get("schema_version") not in VERSIONS or value.get("key") != key:
        raise ValueError("generic_capability_selection_version")
    chain = None
    if history_software is not None:
        from .generic_capability_recovery import software_chain

        windows = software_chain(history_software)
        if windows is not None:
            entry = windows.get(key)
            if entry is None or entry["selection_checksum"] != digest(value):
                raise ValueError("generic_capability_history_chain_selection")
            chain, history_software = history_software, entry["software_reference"]
    continued = value["schema_version"] in CONTINUATION_VERSIONS
    if continued:
        if history_software is not None and chain is None:
            raise ValueError("generic_capability_nested_history_transition_requires_chain")
        from .generic_capability_recovery import continuation_origin

        c = value["continuation"]
        checked = continuation_origin(
            c["predecessor_key"],
            c["software_reference"],
            c["acknowledged_requests"],
            history_chain=chain,
            _seen=(*_seen, key),
        )
    routed = "route_transition" in value if continued else value["schema_version"] == ROUTED_VERSION
    if routed != ("route_transition" in value):
        raise ValueError("generic_capability_selection_route_version")
    if not continued:
        checked = origin(
            value["parent_key"],
            own_claim=value,
            **({"network_route": value["route_transition"]["path"]} if routed else {}),
        )
    parent = directory(value["parent_key"])
    root_claim = value["continuation"]["root_claim"] if continued else value
    if (
        continued
        and read(directory(value["continuation"]["predecessor_key"]) / CONTINUATION_CLAIM) != value
    ):
        raise ValueError("generic_capability_continuation_claim_binding")
    if (
        value != selection_for(checked, key)
        or read(parent / CLAIM) != root_claim
        or read(root / "binding.json") != read(parent / "binding.json")
    ):
        raise ValueError("generic_capability_selection_binding")
    started = read(root / "started.json")
    if started.get("input_checksum") != digest(value) or started.get("key") != key:
        raise ValueError("generic_capability_saved_start_binding")
    campaign = CapabilitySuccessor(checked, key)
    # Original freeze compares all dependencies and preserves the saved deadline.
    # Existence was checked above, so this cannot create a new protocol.
    if history_software is None:
        campaign.freeze()
    else:
        from .generic_capability_recovery import historical_protocol

        historical_protocol(campaign, history_software)
    return campaign


def run(key, successor_key, network_route=None):
    from db.runtime.repository.case_pipeline import CasePipelineRepository
    from packages.multimodal_research.unknown_acquisition import check_network

    from .generic_control import start
    from .generic_execution_lock import target
    from .generic_run import finish_export, persist_export

    root, out = directory(key), directory(successor_key)
    lease = "generic-case-execution:" + digest(str(root.resolve()))[7:]
    with CasePipelineRepository(target()).guarded_session("CASE:GENERIC_EXECUTION", lease) as guard:
        guard()
        existing = read(root / CLAIM) if (root / CLAIM).exists() else None
        checked = origin(
            key, own_claim=existing, **({"network_route": network_route} if network_route else {})
        )
        value = selection_for(checked, successor_key)
        if existing is not None and existing != value:
            raise ValueError("generic_capability_successor_conflict")
        if (out / "input.json").exists() and read(out / "input.json") != value:
            raise ValueError("generic_capability_output_conflict")
        if (out / "export-reference.json").exists():
            return dict(status="already_recorded", exports=finish_export(successor_key))
        if (out / "started.json").exists():
            raise ValueError("generic_capability_started_use_export_not_resend")
        check_network(getattr(checked, "execution_config", checked.campaign.config))
        guard()
        out.mkdir(mode=0o700, parents=True, exist_ok=True)
        write_once(root / CLAIM, value)
        write_once(out / "input.json", value)
        write_once(out / "binding.json", read(root / "binding.json"))
        start(value)
        campaign = CapabilitySuccessor(checked, successor_key)
        campaign._parent_execution_guard = guard
        result = campaign.execute()
        guard()
        return persist_export(successor_key, campaign, result)


def report_metadata(campaign, parent):
    """Read exact saved producer metadata; never infer medical scope in HTML."""
    docs = campaign.documents(parent.material.scope.run_id)
    projection = docs.read("generic-capability-demand-projection")
    if (
        not projection
        or len(projection) != 1
        or projection[0]["phase_input_checksum"] != digest(parent)
        or projection[0]["selection_checksum"] != campaign.selection["selection_checksum"]
        or projection[0]["correspondence_checksum"] != campaign.selection["correspondence_checksum"]
    ):
        raise ValueError("generic_capability_report_saved_projection_binding")
    cutoff = None
    path = campaign.output / "acquisition-cutoff.json" if hasattr(campaign, "output") else None
    if path is not None and path.exists():
        from .generic_capability_recovery import phase_inventory
        from .generic_capability_successor import COMPLETION_VERSION

        cutoff = read(path)
        if (
            cutoff.get("schema_version") != COMPLETION_VERSION
            or cutoff.get("protocol_checksum") != digest(campaign.protocol)
            or cutoff.get("phases")
            != [phase_inventory(campaign, t) for t in campaign.visual_targets]
        ):
            raise ValueError("generic_capability_report_cutoff_binding")
    return dict(
        **({"acquisition_completion": cutoff} if cutoff is not None else {}),
        schema_version="generic-capability-increment-report-scope/1.0.0",
        selection=campaign.selection,
        projection=projection[0],
        whole_case_aggregation_required=True,
        historical_scores_modified=False,
        independent_evidence_count_claimed=False,
    )


def continue_run(
    key, successor_key, history_software, acknowledged_requests=(), *, check_only=False
):
    """One explicitly linked window through the same CapabilitySuccessor executor."""
    from pathlib import Path

    from db.runtime.repository.case_pipeline import CasePipelineRepository
    from packages.multimodal_research.unknown_acquisition import check_network
    from packages.object_store import checksum

    from .generic_capability_recovery import continuation_origin
    from .generic_control import start
    from .generic_execution_lock import target
    from .generic_run import persist_export

    previous, out = directory(key), directory(successor_key)
    value = read(previous / "input.json")
    if value.get("schema_version") not in VERSIONS:
        raise ValueError("generic_capability_continuation_original_version")
    if successor_key in {key, value["parent_key"]}:
        raise ValueError("generic_capability_distinct_successor_identity")
    proof_path = Path(history_software).resolve()
    reference = dict(path=str(proof_path), checksum=checksum(proof_path.read_bytes()))
    root = directory(value["parent_key"])
    lease = "generic-case-execution:" + digest(str(root.resolve()))[7:]
    with CasePipelineRepository(target()).guarded_session("CASE:GENERIC_EXECUTION", lease) as guard:
        guard()
        checked = continuation_origin(key, reference, acknowledged_requests)
        selected = selection_for(checked, successor_key)
        claim = previous / CONTINUATION_CLAIM
        if claim.exists() and read(claim) != selected:
            raise ValueError("generic_capability_continuation_already_claimed")
        if (out / "input.json").exists() and read(out / "input.json") != selected:
            raise ValueError("generic_capability_output_conflict")
        if (out / "started.json").exists():
            raise ValueError("generic_capability_started_use_export_not_resend")
        check_network(checked.execution_config)
        guard()
        if check_only:
            return dict(
                status="continuation_preflight_verified",
                selection=selected,
                api_requests=0,
                new_window_created=False,
                execution_authorized=False,
            )
        out.mkdir(mode=0o700, parents=True, exist_ok=True)
        write_once(claim, selected)
        write_once(out / "input.json", selected)
        write_once(out / "binding.json", read(root / "binding.json"))
        start(selected)
        campaign = CapabilitySuccessor(checked, successor_key)
        campaign._parent_execution_guard = guard
        result = campaign.execute()
        guard()
        return persist_export(successor_key, campaign, result)


def close_unsent_review(key):
    """Zero-dispatch cleanup of a saved deadline failure via the original Planner.

    This is not resume: no provider or campaign construction, no new deadline,
    no changed result. The parent execution lease excludes the original runner.
    """
    import json
    from pathlib import Path

    from agents.planner.service import Planner
    from db.runtime.repository.case_pipeline import CasePipelineRepository
    from db.runtime.repository.planner import PlannerRepository
    from db.runtime.repository.research import ResearchRunRepository
    from packages.research_runtime.agent_records import RuntimeAgentJournal, RuntimeDocuments

    from .case_execution import owners
    from .evidence_media import acquisition_root
    from .generic_execution_lock import target
    from .generic_run import export

    folder = directory(key)
    selection = read(folder / "input.json")
    if selection.get("schema_version") not in VERSIONS or selection.get("key") != key:
        raise ValueError("generic_capability_closeout_selection")
    parent = directory(selection["parent_key"])
    lease = "generic-case-execution:" + digest(str(parent.resolve()))[7:]
    channel = target()
    with CasePipelineRepository(channel).guarded_session("CASE:GENERIC_EXECUTION", lease) as guard:
        saved = export(key)
        report = json.loads(Path(saved["json"]).read_bytes())
        _, binding = acquisition_root(report, key)  # Exact saved root and successor claims.
        request = read(folder / "evidence/review/request.json")
        run = request["lock"]["run_id"]
        contexts, sessions = ResearchRunRepository(channel), PlannerRepository(channel)
        context, control = contexts.read_research_run(run), sessions.read(run)
        if (
            context is None
            or control is None
            or context.request.model_dump(mode="json") != request
            or context.request.scope.case_code != binding["case_code"]
            or control.binding.run_id != run
            or control.binding.case_code != binding["case_code"]
        ):
            raise ValueError("generic_capability_closeout_context")
        runtime, objects, _ = owners()
        docs = RuntimeDocuments(
            repo=runtime,
            contexts=contexts,
            sessions=sessions,
            store=objects,
            run_id=run,
            root=runtime.read_case_artifact(binding["case_code"], control.binding.graph.code),
        )
        phase = docs.read("dynamic-phase-result")
        if (
            not phase
            or len(phase) != 1
            or phase[0] != read(folder / "evidence/review/result.json")
            or phase[0].get("accepted")
            or phase[0].get("acceptance_key")
            or phase[0].get("error") != "deadline"
            or phase[0].get("run_id") != run
            or report["result"]["phases"][-1] != phase[0]
        ):
            raise ValueError("generic_capability_closeout_saved_failure")
        journal = digest(RuntimeAgentJournal(docs).read())
        guard()
        stopped = close_expired_unsent_review(Planner(runtime=sessions, owner=None), docs)
        if stopped.phase != "stopped" or stopped.pending is not None:
            raise ValueError("generic_capability_closeout_not_stopped")
        if (
            journal != digest(RuntimeAgentJournal(docs).read())
            or docs.read("dynamic-phase-result") != phase
        ):
            raise ValueError("generic_capability_closeout_history_changed")
        guard()
        result = dict(
            schema_version="generic-capability-unsent-review-closeout/1.0.0",
            key=key,
            run_id=run,
            original_phase_checksum=digest(phase[0]),
            control_checksum=digest(stopped),
            journal_checksum=journal,
            status="stopped_without_dispatch",
            api_requests=0,
            old_result_modified=False,
            dispatch_authorized=False,
        )
        write_once(folder / "unsent-review-closeout.json", result)
        return result


def close_expired_unsent_review(planner, documents):
    """Finish the original pending command only with proof of no Review dispatch.

    No provider/medical owner is called. UNKNOWN reservations and committed
    domain facts require their own recovery. A batch intent alone may be closed
    only when it matches the saved parent and no request was ever reserved.
    The original expired Planner step records the stop; no Control is edited here.
    """
    from packages.evidence_details.acceptance_models import DomainState
    from packages.evidence_details.review_batches import (
        batch_payload,
        check_plan,
        read_plan,
        response_key,
    )
    from packages.grounding.models import validate
    from packages.research_runtime.agent_records import RuntimeAgentJournal, RuntimeDocuments

    if type(documents) is not RuntimeDocuments or planner.runtime is not documents.sessions:
        raise ValueError("unsent_review_original_owner_required")
    with documents.session():
        state = planner.read(documents.run_id)
        if state.cancelled or planner.clock() < state.binding.deadline:
            raise ValueError("unsent_review_not_expired")
        if any(
            e.get("phase") in {"reserved", "completed", "response_confirmation"}
            for e in RuntimeAgentJournal(documents).read()
        ):
            raise ValueError("unsent_review_request_exists")
        intent = documents.read("live-stage-intent:review")
        if not intent or len(intent) != 1 or intent[0].get("role") != "review":
            raise ValueError("unsent_review_intent_required")
        event = intent[0]["event"]
        key = event["code"] + ":domain"
        if event["code"] != "review:" + event["checksum"][7:]:
            raise ValueError("unsent_review_event_binding")
        for name in (
            "live-stage:review",
            "submission:" + key,
            "accepted:" + key,
            "review:" + event["code"] + ":result",
            "review:" + event["code"] + ":selected",
        ):
            if documents.read(name):
                raise ValueError("unsent_review_domain_fact_exists")
        for round_id in ("A", "B", "C"):
            candidate_key = event["code"] + ":" + round_id
            if documents.read("review:" + candidate_key + ":candidate"):
                raise ValueError("unsent_review_candidate_exists")
            prefix = "review-batches:" + candidate_key
            saved = documents.read(prefix + ":plan")
            if saved:
                if len(saved) != 1:
                    raise ValueError("unsent_review_plan_binding")
                plan = read_plan(saved[0])
                if plan.key != candidate_key or plan.scope.run_id != documents.run_id:
                    raise ValueError("unsent_review_plan_binding")
                parent = documents.read("dynamic-phase-input")
                if not parent or len(parent) != 1:
                    raise ValueError("unsent_review_parent_required")
                domain = validate(DomainState, parent[0])
                check_plan(domain, plan)
                for batch in plan.batches:
                    slot = prefix + ":" + batch.id
                    if any(
                        documents.read(name)
                        for name in (slot + ":result", response_key(plan, batch))
                    ):
                        raise ValueError("unsent_review_batch_work_exists")
                    claim = documents.read(slot + ":claim")
                    if claim and list(claim) != [
                        dict(
                            plan_checksum=digest(plan),
                            input_checksum=digest(batch_payload(domain, plan, batch)),
                        )
                    ]:
                        raise ValueError("unsent_review_batch_claim_binding")
        if state.phase == "stopped" and state.pending is None:
            if not any(
                e.request_checksum == event["checksum"] and e.outcome.reason == "deadline"
                for e in state.events
            ):
                raise ValueError("unsent_review_stop_binding")
            return state
        if (
            state.pending is None
            or state.pending.command.action != "review"
            or state.pending.launch_claimed
            or state.pending.request_checksum != event["checksum"]
        ):
            raise ValueError("unsent_review_pending_binding")
        planner.step(documents.run_id, state.pending.command)
        return planner.read(documents.run_id)
