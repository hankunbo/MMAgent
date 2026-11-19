"""Terminal execution and zero-request restoration for a definition successor.

All Source/Review work uses DefinitionReview's original campaign executor. This
module owns only explicit CLI composition, frozen identity and local recovery.
It never changes an old complete result, replays UNKNOWN or dispatches tools.
"""

from pathlib import Path
from types import SimpleNamespace

from agents.expert_agent.committed_review import read_committed_review
from agents.expert_agent.reporting import read_research_report
from apps.vectorengine_single_case import write_once
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_reuse.definition_sources import KnowledgeDefinitionSources
from packages.grounding.models import digest
from packages.research_runtime.agent_records import RuntimeAgentJournal

from .generic_definition_stage import GenericDefinitionStage
from .generic_definition_successor import (
    CLAIM,
    VERSION,
    DefinitionReview,
    origin,
    read,
    selection_for,
)
from .generic_input import directory


def source_owner(checked):
    original, p = checked.campaign.base, checked.policy
    return KnowledgeDefinitionSources(
        repository=original.knowledge,
        store=original.knowledge_objects,
        index_keys=p.index_keys,
        failure_policy=p.search_failure_policy,
        material_policy=p.material_policy,
        passage_plans=p.passage_plans,
    )


def load_selection(key):
    root = directory(key)
    value = read(root / "input.json")
    if value.get("schema_version") in {
        "generic-capability-successor/1.0.0",
        "generic-capability-successor/1.1.0",
        "generic-capability-successor/1.2.0",
        "generic-capability-successor/1.3.0",
    }:
        from .generic_capability_execution import load_selection as load_capability

        return load_capability(key)
    if value.get("schema_version") != VERSION or value.get("key") != key:
        raise ValueError("generic_definition_selection_version")
    checked = origin(value["parent_key"], value["definition_policy"], own_claim=value)
    if (
        value != selection_for(checked, key)
        or read(directory(value["parent_key"]) / CLAIM) != value
        or read(root / "binding.json") != read(directory(value["parent_key"]) / "binding.json")
    ):
        raise ValueError("generic_definition_selection_binding")
    campaign = DefinitionReview(checked, key, sources=source_owner(checked))
    campaign.protocol = read(campaign.output / "protocol.json")
    p = campaign.protocol
    started = read(root / "started.json")
    if (
        p["schema_version"] != "generic-evidence-campaign/1.0.0"
        or p["input"] != digest(campaign.spec)
        or p["provider_checksum"] != digest(campaign.config)
        or p["previous"] != campaign.prior()
        or p["observations"] != campaign.freeze_observations()
        or p["max_new_requests"] != value["max_new_requests"]
        or p["max_cumulative_graph_requests"] != value["max_cumulative_requests"]
        or p["seconds"] != value["seconds"]
        or started["input_checksum"] != digest(value)
        or started["key"] != key
    ):
        raise ValueError("generic_definition_saved_protocol_binding")
    campaign.definition_stage = GenericDefinitionStage(
        campaign, checked.view, value["definition_policy"], sources=campaign.sources
    )
    campaign.definition_stage.restore(report_only=True)
    campaign.bind_review_phase()
    return campaign


def committed_state(campaign):
    """Current facts plus exact original Review commit, not a file's accepted flag."""
    request_path = campaign.phase_path(None) / "request.json"
    if not request_path.is_file():
        raise ValueError("generic_definition_no_review_commit")
    request = read(request_path)
    run_id = request["lock"]["run_id"]
    ctx = campaign.contexts.read_research_run(run_id)
    control = campaign.sessions.read(run_id)
    docs = campaign.documents(run_id)
    if (
        ctx is None
        or ctx.request.model_dump(mode="json") != request
        or ctx.run_id != run_id
        or control is None
        or control.phase != "stopped"
        or control.pending
        or docs.read("dynamic-campaign") != (campaign.protocol,)
    ):
        raise ValueError("generic_definition_original_review_binding")
    evidence = campaign.phase_class(
        campaign.base,
        observation_reader=campaign.accepted_observations,
        model_reader=campaign.read_model,
    )
    state, _ = evidence.project(run_id=run_id, config=campaign.config)
    evidence.verify(state)
    if docs.read("dynamic-phase-input") != (state.model_dump(mode="json"),):
        raise ValueError("generic_definition_current_review_material_changed")
    manifest = getattr(evidence, "definition_revision_manifest", None)
    if manifest is not None and docs.read("generic-definition-review-projection") != (manifest,):
        raise ValueError("generic_definition_current_projection_changed")
    capability = getattr(evidence, "capability_projection", None)
    if capability is not None and docs.read("generic-capability-demand-projection") != (
        capability,
    ):
        raise ValueError("generic_capability_current_projection_changed")
    phase = docs.read("dynamic-phase-result")
    phase = phase[0] if phase else None
    proof = read_committed_review(docs, parent=state, config=campaign.config)
    if proof is None:
        # The original zero-evidence closure is valid only without an adopted
        # definition and without a Review reservation. It is not a fake score.
        zero = docs.read("dynamic-zero-evidence")
        if (
            campaign.definition_keys
            or not zero
            or state.policy.items
            or not phase
            or phase.get("accepted")
            or not phase.get("evidence_limited")
            or control.cancelled
            or not zero_review_journal(campaign, docs, control)
        ):
            raise ValueError("generic_definition_no_confirmed_review_or_zero_closure")
        from packages.evidence_scoring.accepted import build_accepted

        expected_zero = dict(
            status="evidence_limited",
            scores=build_accepted(state, None),
            requirements=state.policy.requirements,
            no_review_requests=True,
        )
        if (
            digest(zero) != digest((expected_zero,))
            or phase.get("run_id") != run_id
            or phase.get("target") is not None
            or phase.get("acceptance_key") is not None
            or phase.get("error") is not None
        ):
            raise ValueError("generic_definition_zero_closure_material_binding")
        return docs, state, phase, None
    if phase and (phase.get("run_id") != run_id or phase.get("target") is not None):
        raise ValueError("generic_definition_original_phase_binding")
    if phase and phase.get("accepted") and phase.get("acceptance_key") != proof["acceptance_key"]:
        raise ValueError("generic_definition_original_acceptance_key_changed")
    acceptance = ResearchAcceptance(
        owner=SimpleNamespace(current=lambda: state), documents=docs, requests=None
    )
    detail = read_research_report(acceptance, key=proof["acceptance_key"], parent=state)
    if digest(detail) != proof["report_checksum"]:
        raise ValueError("generic_definition_committed_report_changed")
    return docs, state, phase, (proof, detail)


def zero_review_journal(campaign, documents, control):
    """Initialization metadata is not a dispatch; any other event fails closed.

    This is a read-only confirmation of an existing zero-evidence closure. It
    neither opens a new budget nor grants execution after cancellation/expiry.
    """
    events = RuntimeAgentJournal(documents).read()
    if not events:
        return True  # Preserve the previously accepted empty-journal format.
    binding = control.binding
    seconds = binding.profile.seconds
    if not 0 < seconds <= 2700:
        return False
    expected = (
        dict(
            schema_version="agent-budget/1.0.0",
            binding=dict(
                run_id=documents.run_id, deadline=binding.deadline.isoformat(), max_requests=40
            ),
        ),
        dict(
            schema_version="bounded-live-limits/1.0.0"
            if seconds == 2700
            else "bounded-live-limits/1.1.0",
            config_checksum=digest(campaign.config),
            max_steps=20,
            max_no_progress=2,
            max_searches=8,
            max_concurrency=1,
            max_retries=1,
            seconds=seconds,
            deadline=binding.deadline.isoformat(),
        ),
    )
    return events == expected


def closure_prefix(campaign):
    from .generic_capability_successor import CapabilitySuccessor

    return "generic-capability" if type(campaign) is CapabilitySuccessor else "generic-definition"


def export_saved(key):
    """Only a durable commit may produce local output; no secret or provider load."""
    from .generic_run import finish_export, persist_export

    root = directory(key)
    if (root / "export-reference.json").exists():
        # Exact historical owner export, never current execution authority.
        return dict(status="already_recorded", exports=finish_export(key))
    campaign = load_selection(key)
    prefix = closure_prefix(campaign)
    docs, state, phase, committed = committed_state(campaign)
    used = campaign.usage()
    previous = campaign.protocol["previous"]["reservations"]
    if used > campaign.max_new_requests or used + previous > 40:
        raise ValueError("generic_definition_saved_budget_binding")
    saved = docs.read("dynamic-case-result")
    if saved:
        result = saved[0]
        if (
            result["phases"] != [phase]
            or result["new_requests"] != used
            or result["prior_graph_requests"] != previous
        ):
            raise ValueError("generic_definition_saved_case_result_binding")
        if committed and not phase.get("accepted"):
            saved = None  # Keep failed result, create separate local closure below.
    if not saved:
        proof = committed[0] if committed else None
        final = (
            phase
            if proof is None
            else dict(
                phase or {},
                schema_version="dynamic-phase-local-closure/1.0.0",
                run_id=docs.run_id,
                target=None,
                accepted=True,
                acceptance_key=proof["acceptance_key"],
                evidence_limited=False,
                closure=dict(
                    original_phase=phase, original_phase_checksum=digest(phase), confirmation=proof
                ),
            )
        )
        result = dict(
            schema_version=prefix + "-local-case-result/1.0.0",
            sample=campaign.sample,
            case_code=campaign.base.binding["case_code"],
            phases=[final],
            complete=campaign.phase_completed(final) if prefix == "generic-capability" else True,
            new_requests=used,
            prior_graph_requests=previous,
            reused_visual_observations=len(campaign.accepted_observations()),
            clinical_acceptance=False,
            hidden_labels_used=False,
            local_closure=dict(
                original_case_result_checksum=digest(docs.read("dynamic-case-result")),
                original_phase_checksum=digest(phase),
                api_requests=0,
            ),
        )
        campaign.report_key = prefix + "-local-case-report"
        campaign.export_suffix += "_本地恢复"
        with docs.session():
            docs.write(prefix + "-local-case-result", (result,))
        write_once(campaign.output / "local-closure-result.json", result)
    else:
        write_once(campaign.output / "result.json", result)
    if committed:
        write_once(campaign.phase_path(None) / "review-report.json", committed[1])
    # Original report production and immutable bundle owner, not a new renderer.
    report_key = getattr(campaign, "report_key", "dynamic-case-report")
    saved_report = docs.read(report_key)
    if saved_report is None:
        campaign.export_result(result)
    elif (
        len(saved_report) != 2
        or saved_report[0].get("result") != result
        or saved_report[1].get("report_checksum") != digest(saved_report[0])
        or (committed and saved_report[0].get("scores") != committed[1]["scores"])
    ):
        raise ValueError("generic_definition_existing_report_binding")
    return persist_export(key, campaign, result)


def run(key, review_key, policy_path):
    from db.runtime.repository.case_pipeline import CasePipelineRepository
    from packages.multimodal_research.unknown_acquisition import check_network

    from .generic_control import start
    from .generic_execution_lock import target
    from .generic_run import finish_export, persist_export

    policy_path = Path(policy_path).resolve()
    repository = Path(__file__).resolve().parents[2]
    if not policy_path.is_relative_to(repository) or policy_path.suffix != ".json":
        raise ValueError("generic_definition_public_policy_path")
    root, out = directory(key), directory(review_key)
    lease = "generic-case-execution:" + digest(str(root.resolve()))[7:]
    with CasePipelineRepository(target()).guarded_session("CASE:GENERIC_EXECUTION", lease) as guard:
        existing = read(root / CLAIM) if (root / CLAIM).exists() else None
        checked = origin(key, read(policy_path), own_claim=existing)
        value = selection_for(checked, review_key)
        if existing is not None and existing != value:
            raise ValueError("generic_definition_successor_conflict")
        if (out / "input.json").exists() and read(out / "input.json") != value:
            raise ValueError("generic_definition_output_conflict")
        if (out / "export-reference.json").exists():
            return dict(status="already_recorded", exports=finish_export(review_key))
        if (out / "started.json").exists():
            raise ValueError("generic_definition_started_use_export_not_resend")
        check_network(checked.campaign.config)
        guard()
        out.mkdir(mode=0o700, parents=True, exist_ok=True)
        write_once(root / CLAIM, value)
        write_once(out / "input.json", value)
        write_once(out / "binding.json", read(root / "binding.json"))
        start(value)
        campaign = DefinitionReview(checked, review_key, sources=source_owner(checked))
        campaign._parent_execution_guard = guard
        result = campaign.execute()
        guard()
        return persist_export(review_key, campaign, result)


def restore(path, *, history_software=None):
    """Read a saved successor for original aggregation; never dispatch or repair."""
    path = Path(path).absolute()
    if path != directory(path.name):
        raise ValueError("generic_definition_restore_path")
    if history_software is not None:
        from .generic_capability_execution import load_selection as load_capability

        campaign = load_capability(path.name, history_software=history_software)
    else:
        campaign = load_selection(path.name)
    prefix = closure_prefix(campaign)
    docs, _, phase, committed = committed_state(campaign)
    if committed is None:
        raise ValueError("generic_definition_aggregation_requires_new_review")
    local = campaign.output / "local-closure-result.json"
    result_path = local if local.exists() else campaign.output / "result.json"
    result = read(result_path)
    result_key = prefix + "-local-case-result" if local.exists() else "dynamic-case-result"
    report_key = prefix + "-local-case-report" if local.exists() else "dynamic-case-report"
    report = docs.read(report_key)
    final = result["phases"][-1]
    proof, detail = committed
    if (
        docs.read(result_key) != (result,)
        or not result["complete"]
        or len(result["phases"]) != 1
        or not final["accepted"]
        or final["target"] is not None
        or final["run_id"] != docs.run_id
        or final["acceptance_key"] != proof["acceptance_key"]
        or result["new_requests"] != campaign.usage()
        or result["prior_graph_requests"] != campaign.protocol["previous"]["reservations"]
        or result["new_requests"] > campaign.max_new_requests
        or result["new_requests"] + result["prior_graph_requests"] > 40
        or not report
        or len(report) != 2
        or report[0]["result"] != result
        or report[0]["scores"] != detail["scores"]
        or report[1]["report_checksum"] != digest(report[0])
        or (not local.exists() and final != phase)
        or (local.exists() and final.get("closure", {}).get("confirmation") != proof)
    ):
        raise ValueError("generic_definition_restore_original_result")
    return campaign, result
