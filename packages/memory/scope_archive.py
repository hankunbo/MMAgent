"""Historical input/method context for the original multimodal Memory archive.

Read committed Runtime facts, not local mirrors. This never grants current
execution, imports old patient values, or turns an input survey into a vote.
"""

from db.runtime.repository.planner import PlannerRepository
from db.runtime.repository.research import ResearchRunRepository
from packages.evidence_details.acceptance_models import DomainState
from packages.grounding.models import digest, validate
from packages.multimodal_research.input_scope import accepted_fact
from packages.multimodal_research.scope_selection import storage_identity

VERSIONS = {
    "multimodal-scope-case-result/1.0.0",
    "multimodal-scoped-method-case-result/1.0.0",
}


def check_phase_input(initial, context):
    """CaseScope identifies the patient; material Scope also binds run and bytes."""
    request = context.request
    expected = dict(
        case_code=request.scope.case_code,
        run_id=context.run_id,
        input_checksum=request.input_checksum,
        visible_checksum=request.visible_checksum,
    )
    if any(getattr(initial.material.scope, k) != value for k, value in expected.items()):
        raise ValueError("memory_scope_phase_input_binding")


def survey_origin(read, protocol, entry, saved, *, depth=0):
    """Exact versioned historical recovery lineage, never a dispatch permission."""
    if depth > 16:
        raise ValueError("memory_scope_reuse_depth")
    if protocol.get("schema_version") in {
        "multimodal-campaign-renewal/1.84.0",
        "multimodal-campaign-renewal/1.85.0",
        "multimodal-campaign-renewal/1.86.0",
        "multimodal-campaign-renewal/1.87.0",
    }:
        proof = protocol.get("current_scope_reuse")
        matching = [r for r in proof["reused"] if r["target"] == entry["target"]] if proof else []
        if matching:
            parent = read("dynamic-campaign", source_run=proof["parent_run_id"])
            result = read("multimodal-case-result", source_run=proof["parent_run_id"])
            if (
                len(parent) != 1
                or len(result) != 1
                or digest(parent[0]) != proof["parent_protocol_checksum"]
                or digest(result[0]) != proof["parent_result_checksum"]
                or parent[0]["input"] != protocol["input"]
                or parent[0]["task"] != protocol["task"]
                or matching
                != [dict(target=entry["target"], fact=saved, fact_checksum=digest(saved))]
                or saved not in result[0]["input_scopes"]
                or entry not in parent[0]["capability_transition"]["scope_plan"]["entries"]
            ):
                raise ValueError("memory_scope_current_reuse_binding")
            return survey_origin(read, parent[0], entry, saved, depth=depth + 1)
    phase = saved["phase"]
    run_id = phase["run_id"]
    original_phase = read("dynamic-phase-result", source_run=run_id)
    original_protocol = read("dynamic-campaign", source_run=run_id)
    if phase.get("schema_version") != "recorded-input-scope-result/1.0.0":
        if original_phase != [phase] or original_protocol != [protocol]:
            raise ValueError("memory_scope_phase_binding")
        return protocol["input_scope"]["provider_checksum"]
    from packages.evidence_details.scope_finalization import ACCEPTANCE_KEY, GRANT_KEY
    from packages.multimodal_research.capability_window import claim
    from packages.multimodal_research.scope_successor import VERSIONS, claim_key

    proof = protocol.get("scope_reuse", {})
    matching = [r for r in proof.get("reused", []) if r["target"] == entry["target"]]
    recovery = read("recorded-input-scope:result:1", source_run=run_id)
    grant = read(GRANT_KEY, source_run=run_id)
    if (
        protocol.get("schema_version") not in VERSIONS
        or len(matching) != 1
        or matching[0] != dict(target=entry["target"], fact=saved, fact_checksum=digest(saved))
        or recovery != [phase]
        or original_phase != [phase.get("original_phase")]
        or phase.get("original_phase_checksum") != digest(original_phase[0])
        or len(original_protocol) != 1
        or digest(original_protocol[0]) != proof.get("protocol_checksum")
        or phase.get("acceptance_key") != ACCEPTANCE_KEY
        or len(grant) != 1
        or digest(grant[0]) != phase.get("grant_checksum")
        or grant[0]["original_request_checksum"] != phase.get("original_request_checksum")
        or grant[0]["run_id"] != run_id
        or phase.get("new_api_requests") != 0
        or phase.get("new_disease_scores") != 0
        or phase.get("campaign_resumed") is not False
        or read(claim_key(proof), source_run=proof["reused"][0]["fact"]["phase"]["run_id"])
        != [claim(protocol, protocol["output"])]
    ):
        raise ValueError("memory_scope_recovery_lineage")
    old_entries = original_protocol[0]["capability_transition"]["scope_plan"]["entries"]
    if entry not in old_entries:
        raise ValueError("memory_scope_recovery_task_or_view")
    return original_protocol[0]["input_scope"]["provider_checksum"]


def read_context(owner, read, refs, result, report, parent):
    """Rebuild immutable ancestry; Memory current-source checks remain in force."""
    protocol = read("dynamic-campaign")[0]
    assessment = report["input_scope_assessment"]
    plan = assessment["plan"]
    scope = protocol["input_scope"]
    contexts = ResearchRunRepository(owner.runtime._target)
    sessions = PlannerRepository(owner.runtime._target)

    def context(run_id):
        value = contexts.read_research_run(run_id)
        request = value.request
        if (
            request.scope.case_code != result["case_code"]
            or request.task.model_dump(mode="json") != result["task"]
            or request.task.checksum != protocol["task"]
            or request.input_checksum != protocol["input"]
            or request.visible_checksum != protocol["visible"]
        ):
            raise ValueError("memory_scope_original_input_binding")
        for expected in request.artifacts:
            asset, _ = owner.artifact(result["case_code"], expected.asset_code)
            if asset.checksum != expected.checksum:
                raise ValueError("memory_scope_original_artifact_binding")
            refs[asset.asset_code] = asset
        return value

    final_context = context(result["phases"][-1]["run_id"])
    if (
        assessment["schema_version"] != "input-scope-report/1.0.0"
        or plan["schema_version"] != "input-scope-phase-plan/1.0.0"
        or assessment["disease_votes"] != 0
        or plan["disease_votes"] != 0
        or plan["input_checksum"] != protocol["input"]
        or scope["plan_checksum"] != digest(plan)
        or result["input_scope_plan_checksum"] != digest(plan)
        or assessment["records"] != result["input_scopes"]
        or scope["targets"] != [entry["target"] for entry in plan["entries"]]
        or parent["config_checksum"] != protocol["provider"]
        or len(result["input_scopes"]) != len(plan["entries"])
    ):
        raise ValueError("memory_scope_report_binding")
    facts = []
    for entry, saved in zip(plan["entries"], result["input_scopes"], strict=True):
        phase = saved["phase"]
        run_id = phase["run_id"]
        original = context(run_id)
        initial = validate(DomainState, read("dynamic-phase-input", source_run=run_id)[0])
        check_phase_input(initial, original)
        expected_provider = survey_origin(read, protocol, entry, saved)
        if (
            phase["target"] != entry["target"]
            or initial.config_checksum != expected_provider
            or original.synthetic != final_context.synthetic
            or entry["task"]["input_checksum"] != protocol["input"]
            or entry["task"]["component_id"] not in original.request.selected_components
        ):
            raise ValueError("memory_scope_phase_binding")
        values = read("accepted:" + phase["acceptance_key"], source_run=run_id)
        fact = accepted_fact(values, initial, phase, entry["task"], sessions.read(run_id))
        if fact != saved:
            raise ValueError("memory_scope_accepted_fact_changed")
        for resource in initial.resources:
            asset, _ = owner.artifact(result["case_code"], resource.code)
            if asset.checksum != resource.checksum:
                raise ValueError("memory_scope_resource_binding")
            refs[asset.asset_code] = asset
        facts.append(fact)

    method = result.get("scope_method_plan")
    location = None
    if result["schema_version"] == "multimodal-scoped-method-case-result/1.0.0":
        if not method:
            raise ValueError("memory_scope_method_missing")
        run_id, key = storage_identity(method, protocol["campaign_id"])
        original = context(run_id)
        if (
            read(key, source_run=run_id) != [method]
            or method != assessment.get("method_plan")
            or method["input_checksum"] != protocol["input"]
            or method["target"] != result["task"]["target_code"]
            or method["scope_plan_checksum"] != digest(plan)
            or method["scope_records"]
            != [{k: f[k] for k in ("phase", "records_checksum", "receipt_checksum")} for f in facts]
            or method["added_disease_votes"] != 0
            or method["physical_scale_authorized"] is not False
            or original.synthetic != final_context.synthetic
        ):
            raise ValueError("memory_scope_method_binding")
        if not facts and original.request.policy_ref.code != "POLICY:INPUT_CAPABILITY_METHOD_PLAN":
            raise ValueError("memory_scope_numeric_context_binding")
        location = dict(run_id=run_id, key=key, checksum=digest(method))
    elif method is not None or assessment.get("method_plan") is not None:
        raise ValueError("memory_scope_unversioned_method")
    return dict(
        schema_version="memory-input-method-context/1.0.0",
        scope_plan_checksum=digest(plan),
        scope_record_checksums=[f["records_checksum"] for f in facts],
        method_plan=location,
        disease_votes=0,
        current_use="requires_current_input_scope_tool_and_source_owners",
    )
