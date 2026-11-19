"""Read-only intermediate mirrors, explicitly not authorization or final audit.

No model, tool, database or recovery call. Final owner readback remains separate.
"""

import json
from collections import Counter
from pathlib import Path

from packages.grounding.models import digest, validate
from packages.object_store import checksum
from packages.research_runtime.task_context import TaskRunRequest
from packages.tool_admission.models import ResearchRequest

MAX_BYTES = 16 * 1024 * 1024
EXPORT_ROOT = Path(__file__).resolve().parents[2] / "docs" / "exports"
TEXT_SUCCESSOR_VERSIONS = {
    "multimodal-campaign-renewal/1.35.0",
    "multimodal-campaign-renewal/1.36.0",
    "multimodal-campaign-renewal/1.38.0",
    "multimodal-campaign-renewal/1.39.0",
    "multimodal-campaign-renewal/1.40.0",
    "multimodal-campaign-renewal/1.42.0",
    "multimodal-campaign-renewal/1.43.0",
    "multimodal-campaign-renewal/1.45.0",
    "multimodal-campaign-renewal/1.46.0",
}
RETAINED_ACQUISITION_VERSION = "multimodal-campaign-renewal/1.37.0"
RETAINED_ACQUISITION_VERSIONS = {
    RETAINED_ACQUISITION_VERSION,
    "multimodal-campaign-renewal/1.47.0",
    "multimodal-campaign-renewal/1.48.0",
    "multimodal-campaign-renewal/1.49.0",
    "multimodal-campaign-renewal/1.50.0",
    "multimodal-campaign-renewal/1.51.0",
    "multimodal-campaign-renewal/1.66.0",
    "multimodal-campaign-renewal/1.67.0",
    "multimodal-campaign-renewal/1.70.0",
}


def successor_request_path(directory, protocol):
    """Locate a current successor's sealed mirror, never follow ancestor runs."""
    if protocol.get("schema_version") in RETAINED_ACQUISITION_VERSIONS:
        targets = protocol.get("acquisition_targets")
        if (
            protocol.get("mode") != "acquisition"
            or not isinstance(targets, list)
            or not 1 <= len(targets) <= 3
            or any(not isinstance(value, str) or not value for value in targets)
            or len(set(targets)) != len(targets)
        ):
            raise ValueError("cohort_progress_renewal_binding")
        folders = [directory / "review"] + [
            directory / ("vision-" + digest(target)[7:19]) for target in targets
        ]
    else:
        folders = [directory / "review"]
    return next(
        (folder / "request.json" for folder in folders if (folder / "request.json").exists()),
        None,
    )


def bounded(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("cohort_progress_symlink")
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("cohort_progress_size")
    return json.loads(raw), checksum(raw)


def fair(directory):
    """Project original Fair receipts, without database access or dispatch."""
    directory = Path(directory)
    if not (directory / "input.json").exists():
        return dict(status="not_prepared", authorization=False, diagnostic_api_requests=0)
    spec, spec_sha = bounded(directory / "input.json")
    if spec.get("key") != directory.name:
        raise ValueError("cohort_fair_input_identity")
    evidence = directory / "evidence"
    if not (evidence / "protocol.json").exists():
        return dict(status="campaign_not_started", authorization=False, diagnostic_api_requests=0)
    protocol, protocol_sha = bounded(evidence / "protocol.json")
    binding, _ = bounded(directory / "binding.json")
    plan, plan_sha = bounded(evidence / "task-plan.json")
    if (
        protocol.get("schema_version") != "generic-evidence-campaign/1.0.0"
        or protocol["input"] != digest(spec)
        or protocol["case_code"] != binding["case_code"]
        or protocol["observations"]["plan"] != digest(plan)
    ):
        raise ValueError("cohort_fair_protocol_binding")
    actions = {
        "review_measurement",
        "observe_partial",
        "record_input_gap",
        "record_definition_gap",
        "retain_attempt",
    }
    if (
        not isinstance(plan, list)
        or any(not isinstance(p, dict) or p.get("action") not in actions for p in plan)
        or len({p["id"] for p in plan}) != len(plan)
    ):
        raise ValueError("cohort_fair_task_plan")
    visual_ids = {p["id"] for p in plan if p["action"] == "observe_partial"}
    phases, errors, seen = [], [], set()
    for folder in sorted(evidence.glob("vision-*")) + [evidence / "review"]:
        if not folder.is_dir():
            continue
        try:
            raw, request_sha = bounded(folder / "request.json")
            request = validate(ResearchRequest, raw)
            image = binding["image"]
            if (
                request.scope.case_code != binding["case_code"]
                or not any(
                    a.asset_code == image["asset_code"] and a.checksum == image["checksum"]
                    for a in request.artifacts
                )
                or folder.name not in {"review", *("vision-" + digest(t)[7:19] for t in visual_ids)}
            ):
                raise ValueError("cohort_fair_request_binding")
            if not (folder / "result.json").exists():
                phases.append(
                    dict(
                        path=folder.name,
                        status="prepared_or_inflight_not_confirmed",
                        run_id=request.lock.run_id,
                        request_checksum=request_sha,
                    )
                )
                continue
            result, result_sha = bounded(folder / "result.json")
            target, summary = result["target"], result["requests"]
            expected = "vision-" + digest(target)[7:19] if target is not None else "review"
            if (
                folder.name != expected
                or (target is not None and target not in visual_ids)
                or result["schema_version"] != "dynamic-phase-result/1.0.0"
                or result["run_id"] != request.lock.run_id
                or summary["run_id"] != request.lock.run_id
                or result["run_id"] in seen
                or type(result["accepted"]) is not bool
            ):
                raise ValueError("cohort_fair_phase_binding")
            for field in (
                "api_requests_observed",
                "api_requests_unknown",
                "reservations",
                "completed",
            ):
                if type(summary[field]) is not int or summary[field] < 0:
                    raise ValueError("cohort_fair_phase_count")
            seen.add(result["run_id"])
            phases.append(
                dict(
                    path=folder.name,
                    target=target,
                    status="saved_phase_mirror",
                    run_id=result["run_id"],
                    accepted=result["accepted"],
                    error=result["error"],
                    acceptance_key=result["acceptance_key"],
                    request_checksum=request_sha,
                    phase_checksum=result_sha,
                    requests=summary,
                    transport=result.get("transport"),
                )
            )
        except (ValueError, KeyError, TypeError, OSError) as error:
            errors.append(
                dict(
                    path=folder.name,
                    status="mirror_not_readable",
                    error_type=type(error).__name__,
                )
            )
    saved = [p for p in phases if p["status"] == "saved_phase_mirror"]
    return dict(
        schema_version="cohort-fair-progress/1.0.0",
        status="local_mirrors_only",
        input_checksum=spec_sha,
        protocol_checksum=protocol_sha,
        task_plan_checksum=plan_sha,
        deadline=protocol["deadline"],
        plan_action_counts=dict(Counter(p["action"] for p in plan)),
        phases=phases,
        errors=errors,
        saved_accepted_visual_phases=sum(p["accepted"] and p["target"] is not None for p in saved),
        saved_accepted_review_phases=sum(p["accepted"] and p["target"] is None for p in saved),
        phase_api_requests_lower_bound=sum(p["requests"]["api_requests_observed"] for p in saved),
        completion=fair_completion(directory, binding, saved),
        source_or_unfinished_requests_not_inferred=True,
        clinical_validation=False,
        authorization=False,
        automatic_replay=False,
        diagnostic_api_requests=0,
        database_writes=0,
    )


def fair_completion(directory, binding, phases):
    """Saved report coverage is not an independent owner or medical audit."""
    unknown = dict(
        status="final_report_not_saved",
        pending_review_ids=None,
        all_obtained_reviewed=None,
        whole_case_success_not_inferred=True,
        authorization=False,
    )
    result_path = directory / "evidence" / "result.json"
    report_path = EXPORT_ROOT / ("FairVLMed_" + directory.name + ".json")
    if not result_path.exists() or not report_path.exists():
        return unknown
    try:
        result, result_sha = bounded(result_path)
        report, report_sha = bounded(report_path)
        if result["case_code"] != binding["case_code"] or report["result"] != result:
            raise ValueError("cohort_fair_report_binding")
        final = [p for p in phases if p["target"] is None]
        if len(final) != 1 or len(result["phases"]) != 1:
            raise ValueError("cohort_fair_final_phase")
        if any(
            final[0][field] != result["phases"][0][field]
            for field in ("run_id", "accepted", "acceptance_key", "target", "requests")
        ):
            raise ValueError("cohort_fair_final_phase_binding")
        fields = report["completion"]
        for field in ("report_saved", "domain_accepted", "all_obtained_reviewed"):
            if type(fields[field]) is not bool:
                raise ValueError("cohort_fair_completion_boolean")
        pending = fields["pending_review_ids"]
        if (
            not isinstance(pending, list)
            or any(not isinstance(v, str) or not v for v in pending)
            or len(set(pending)) != len(pending)
            or (pending and fields["all_obtained_reviewed"])
            or fields["domain_accepted"] != final[0]["accepted"]
        ):
            raise ValueError("cohort_fair_completion_conflict")
        counts = report["counts"]
        for field in ("requirements", "acquired", "reviewed", "directed"):
            if type(counts[field]) is not int or counts[field] < 0:
                raise ValueError("cohort_fair_completion_count")
        return dict(
            status="saved_report_mirror",
            result_checksum=result_sha,
            report_checksum=report_sha,
            pending_review_ids=pending,
            all_obtained_reviewed=fields["all_obtained_reviewed"],
            report_saved=fields["report_saved"],
            domain_accepted=fields["domain_accepted"],
            counts=counts,
            whole_case_success_not_inferred=True,
            authorization=False,
        )
    except (KeyError, TypeError, ValueError, OSError) as error:
        return dict(unknown, status="final_report_not_readable", error_type=type(error).__name__)


def multimodal(directory):
    directory = Path(directory)
    if not (directory / "protocol.json").exists():
        return dict(status="not_started", authorization=False)
    protocol, protocol_sha = bounded(directory / "protocol.json")
    if (
        protocol.get("schema_version") in TEXT_SUCCESSOR_VERSIONS | RETAINED_ACQUISITION_VERSIONS
        and successor_request_path(directory, protocol) is None
    ):
        # Missing local mirrors never prove zero dispatch or authorize a retry.
        return dict(
            status="renewal_request_mirror_not_available",
            protocol_checksum=protocol_sha,
            request_state="unknown",
            authorization=False,
            automatic_replay=False,
            diagnostic_api_requests=0,
        )
    entry, entry_sha, identity_source = input_identity(directory, protocol)
    phases, errors, seen = [], [], set()
    for folder in sorted(directory.glob("vision-*")) + [directory / "review"]:
        if not folder.is_dir():
            continue
        try:
            if protocol.get("schema_version") in RETAINED_ACQUISITION_VERSIONS and (
                folder.name
                not in {
                    "review",
                    *(
                        "vision-" + digest(target)[7:19]
                        for target in protocol["acquisition_targets"]
                    ),
                }
            ):
                raise ValueError("cohort_progress_phase_binding")
            request, request_sha = bounded(folder / "request.json")
            request = validate(TaskRunRequest, request)
            if (
                request.input_checksum != entry["input_checksum"]
                or request.task.model_dump(mode="json") != entry["intent"]
            ):
                raise ValueError("cohort_progress_request_binding")
            if not (folder / "result.json").exists():
                phases.append(
                    dict(
                        path=folder.name,
                        status="prepared_or_inflight_not_confirmed",
                        run_id=request.lock.run_id,
                        request_checksum=request_sha,
                    )
                )
                continue
            result, sha = bounded(folder / "result.json")
            expected = "vision-" + digest(result["target"])[7:19] if result["target"] else "review"
            if (
                folder.name != expected
                or result["schema_version"] != "dynamic-phase-result/1.0.0"
                or result["run_id"] != request.lock.run_id
                or result["run_id"] in seen
                or result["requests"]["run_id"] != result["run_id"]
            ):
                raise ValueError("cohort_progress_phase_binding")
            seen.add(result["run_id"])
            summary = result["requests"]
            for field in (
                "api_requests_observed",
                "api_requests_unknown",
                "reservations",
                "completed",
            ):
                if type(summary[field]) is not int or summary[field] < 0:
                    raise ValueError("cohort_progress_request_count")
            phases.append(
                dict(
                    path=folder.name,
                    target=result["target"],
                    status="saved_phase_mirror",
                    run_id=result["run_id"],
                    accepted=result["accepted"],
                    error=result["error"],
                    acceptance_key=result["acceptance_key"],
                    request_checksum=request_sha,
                    phase_checksum=sha,
                    requests=summary,
                    transport=result.get("transport"),
                )
            )
        except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
            errors.append(
                dict(path=folder.name, status="mirror_not_readable", error_type=type(exc).__name__)
            )
    complete = [p for p in phases if p["status"] == "saved_phase_mirror"]
    return dict(
        schema_version="cohort-case-progress/1.1.0",
        status="local_mirrors_only",
        protocol_checksum=protocol_sha,
        entry_checksum=entry_sha,
        identity_source=identity_source,
        deadline=protocol["deadline"],
        phases=phases,
        errors=errors,
        saved_accepted_visual_phases=sum(
            p["accepted"] and p["target"] is not None for p in complete
        ),
        saved_accepted_review_phases=sum(p["accepted"] and p["target"] is None for p in complete),
        phase_api_requests_lower_bound=sum(
            p["requests"]["api_requests_observed"] for p in complete
        ),
        completion=completion_mirror(directory, entry, complete),
        source_or_unfinished_requests_not_inferred=True,
        clinical_validation=False,
        authorization=False,
        automatic_replay=False,
        diagnostic_api_requests=0,
    )


def input_identity(directory, protocol):
    version = protocol.get("schema_version")
    if version in TEXT_SUCCESSOR_VERSIONS | RETAINED_ACQUISITION_VERSIONS:
        # These original-owner successors have no entry.json. Read their own
        # sealed request; do not assemble a campaign, follow ancestors or dispatch.
        raw, sha = bounded(successor_request_path(directory, protocol))
        request = validate(TaskRunRequest, raw)
        mode = "acquisition" if version in RETAINED_ACQUISITION_VERSIONS else "review"
        if (
            protocol.get("mode") != mode
            or protocol["input"] != request.input_checksum
            or protocol["task"] != request.task.checksum
        ):
            raise ValueError("cohort_progress_renewal_binding")
        return (
            dict(
                input_checksum=request.input_checksum, intent=request.task.model_dump(mode="json")
            ),
            sha,
            "renewal_sealed_acquisition_request_not_ancestor_authorization"
            if mode == "acquisition"
            else "renewal_sealed_review_request_not_ancestor_authorization",
        )
    entry, sha = bounded(directory / "entry.json")
    if protocol["campaign_id"] != directory.name or protocol["input"] != entry["input_checksum"]:
        raise ValueError("cohort_progress_input_binding")
    return entry, sha, "original_entry"


def retained_scope_counts(counts, fields):
    """Recognize the original report's explicit retained-context count split."""
    ids = fields.get("retained_context_not_newly_reviewed_ids")
    names = (
        "acquired",
        "obtained_observations",
        "review_scope_observations",
        "retained_unscored_observations",
    )
    return (
        isinstance(ids, list)
        and bool(ids)
        and all(isinstance(value, str) and value for value in ids)
        and len(set(ids)) == len(ids)
        and fields["all_obtained_reviewed"] is False
        and all(type(counts.get(name)) is int and counts[name] >= 0 for name in names)
        and counts["retained_unscored_observations"] == len(ids)
        and counts["review_scope_observations"] == counts["obtained_observations"]
        and counts["acquired"] == counts["review_scope_observations"] + len(ids)
    )


def completion_mirror(directory, entry, phases):
    """Display saved coverage separately from the legacy result.complete flag.

    This does not authorize recovery or independently validate database facts.
    Old reports without explicit completion fields remain unknown, never passed.
    """
    result_path = directory / "result.json"
    report_path = EXPORT_ROOT / ("Multimodal_" + directory.name + ".json")
    if not result_path.exists() or not report_path.exists():
        return dict(status="final_report_not_saved", authorization=False)
    try:
        result, result_sha = bounded(result_path)
        report, report_sha = bounded(report_path)
        if not isinstance(result, dict) or not isinstance(report, dict):
            raise ValueError("cohort_completion_object")
        if (
            report["result"] != result
            or result["task"] != entry["intent"]
            or report["task"] != entry["intent"]
        ):
            raise ValueError("cohort_completion_report_binding")
        local = {p["run_id"]: p for p in phases}
        for phase in result["phases"]:
            saved = local.get(phase["run_id"])
            if saved is None or any(
                saved[key] != phase[key]
                for key in ("accepted", "acceptance_key", "target", "requests")
            ):
                raise ValueError("cohort_completion_phase_binding")
        fields = report.get("completion")
        if fields is None:
            return dict(
                status="legacy_completion_unknown",
                result_checksum=result_sha,
                report_checksum=report_sha,
                authorization=False,
            )
        if not isinstance(fields, dict):
            raise ValueError("cohort_completion_fields")
        for key in ("report_saved", "domain_accepted", "all_obtained_reviewed"):
            if type(fields.get(key)) is not bool:
                raise ValueError("cohort_completion_boolean")
        selected = fields.get("selected_increment_reviewed")
        if selected is not None and type(selected) is not bool:
            raise ValueError("cohort_completion_increment_boolean")
        pending = fields["pending_review_ids"]
        if (
            not isinstance(pending, list)
            or any(not isinstance(v, str) or not v for v in pending)
            or len(set(pending)) != len(pending)
        ):
            raise ValueError("cohort_completion_pending_ids")
        if pending and (fields["all_obtained_reviewed"] or selected is True):
            raise ValueError("cohort_completion_pending_conflict")
        counts = dict(report["counts"])
        scope_split = report.get(
            "schema_version"
        ) == "dynamic-case-report/1.0.0" and retained_scope_counts(counts, fields)
        if report.get("schema_version") == "dynamic-case-report/1.0.0" and "acquired" in counts:
            # Original report owner uses acquired; accepted-aggregate reports use
            # obtained_observations. Normalize only the local status projection,
            # never the persisted report. A retained-context report explicitly
            # distinguishes total acquired from the current Review scope.
            if (
                "obtained_observations" in counts
                and (
                    type(counts["obtained_observations"]) is not int
                    or counts["obtained_observations"] != counts["acquired"]
                )
                and not scope_split
            ):
                raise ValueError("cohort_completion_count_alias")
            counts["obtained_observations"] = counts["acquired"]
        for key in ("requirements", "obtained_observations", "reviewed"):
            if type(counts[key]) is not int or counts[key] < 0:
                raise ValueError("cohort_completion_count")
        unknown = 0
        for phase in result["phases"]:
            count = (phase.get("transport") or {}).get("remote_response_unknown", 0)
            if type(count) is not int or count < 0:
                raise ValueError("cohort_completion_transport_count")
            unknown += count
        return dict(
            status=(
                "accepted_partial_review"
                if fields["domain_accepted"] and (pending or selected is False or unknown)
                else "accepted_review_saved"
                if fields["domain_accepted"]
                else "saved_without_domain_acceptance"
            ),
            reported_result_complete=result.get("complete"),
            report_saved=fields["report_saved"],
            domain_accepted=fields["domain_accepted"],
            all_obtained_reviewed=fields["all_obtained_reviewed"],
            selected_increment_reviewed=selected,
            pending_review_ids=pending,
            remote_response_unknown_lower_bound=unknown,
            counts={
                **{k: counts[k] for k in ("requirements", "obtained_observations", "reviewed")},
                **(
                    {
                        k: counts[k]
                        for k in ("review_scope_observations", "retained_unscored_observations")
                    }
                    if scope_split
                    else {}
                ),
            },
            retained_context_not_newly_reviewed_ids=(
                fields["retained_context_not_newly_reviewed_ids"] if scope_split else []
            ),
            result_checksum=result_sha,
            report_checksum=report_sha,
            original_database_readback_required=True,
            whole_case_success_not_inferred=True,
            authorization=False,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return dict(
            status="completion_mirror_not_readable",
            error_type=type(exc).__name__,
            authorization=False,
        )
