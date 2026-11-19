"""Read explicit saved ancestry, not current qualification or retry permission."""

import json
from collections import Counter
from pathlib import Path

from db.runtime.repository.case_pipeline import identity
from packages.grounding.models import digest
from packages.object_store import checksum

from .demand_coverage import inspect_saved

SOURCE_KINDS = {
    "confirmed_source_schema_failure",
    "confirmed_source_schema_deferred",
    "terminal_source_unknown_deferred",
    "routed_source_failure_deferred",
    "interrupted_public_material_deferred",
}


def inspect_history(report_path, history_root):
    saved = inspect_saved(report_path)
    raw = Path(report_path).read_bytes()
    if checksum(raw) != saved["report_bytes_checksum"]:
        raise ValueError("demand_history_report_changed")
    coverage = saved["coverage"]
    report = json.loads(raw)
    root = Path(history_root).resolve(strict=True)
    history = report["result"].get("previous_attempts", [])
    if not root.is_dir() or len(history) > 64:
        raise ValueError("demand_history_bound")
    files, phases, seen, total = [], [], set(), 0
    source_failures = []
    case = report["result"]["case_code"]
    target = report["task"]["target_code"]
    for entry in history:
        directory = Path(entry["directory"]).resolve(strict=True)
        if not directory.is_relative_to(root) or directory in seen:
            raise ValueError("demand_history_path_scope")
        seen.add(directory)
        kind = entry.get("kind")
        if kind is not None and kind not in SOURCE_KINDS:
            raise ValueError("demand_history_kind")
        path = directory / ("stage-failure.json" if kind else "result.json")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(directory):
            raise ValueError("demand_history_path_scope")
        total += resolved.stat().st_size
        if total > 64 * 1024 * 1024:
            raise ValueError("demand_history_bound")
        value = json.loads(resolved.read_bytes())
        if digest(value) != entry["checksum"]:
            raise ValueError("demand_history_checksum")
        if kind:
            # Original Source failures have no visual attempt. They are retained
            # as provenance only; a search timeout is not a patient examination.
            if value.get("stage") != "definitions" or (directory / "result.json").exists():
                raise ValueError("demand_history_source_stage")
            interrupted = kind == "interrupted_public_material_deferred"
            if kind == "routed_source_failure_deferred" or interrupted:
                protocol_path = (directory / "protocol.json").resolve(strict=True)
                if not protocol_path.is_relative_to(directory):
                    raise ValueError("demand_history_path_scope")
                total += protocol_path.stat().st_size
                if total > 64 * 1024 * 1024:
                    raise ValueError("demand_history_bound")
                protocol = json.loads(protocol_path.read_bytes())
                run = identity(
                    "RUN",
                    "task-research:mm-live:"
                    + digest((directory.name, "definition-expansion"))[7:55],
                )
                if (
                    value.get("campaign_id") != directory.name
                    or value.get("run_id") != run
                    or value.get("protocol_checksum") != digest(protocol)
                    or value.get("input_checksum") != protocol.get("input")
                    or protocol.get("task") != report["task"].get("checksum")
                    or not protocol.get("input")
                    or not protocol.get("task")
                    or type(value.get("new_requests")) is not int
                    or value["new_requests"] < 0
                    or type(entry.get("requests")) is not int
                    or value["new_requests"] != entry["requests"]
                    or value.get("failure_kind")
                    not in (
                        {"no_provider_reservation"}
                        if interrupted
                        else {"dispatched_unknown", "confirmed_response_or_domain_refusal"}
                    )
                    or (
                        interrupted
                        and (
                            value.get("error_code") != "execution_process_interrupted"
                            or value.get("requests")
                            or protocol.get("schema_version")
                            != "multimodal-campaign-renewal/1.56.0"
                        )
                    )
                ):
                    raise ValueError("demand_history_source_binding")
            source_failures.append(
                dict(
                    path=str(path),
                    checksum=entry["checksum"],
                    kind=kind,
                    failure_kind=value.get("failure_kind"),
                    current_execution_authorization=False,
                )
            )
        else:
            if value.get("case_code") != case or value.get("task", {}).get("target_code") != target:
                raise ValueError("demand_history_case_target")
            phases.extend((str(path), p) for p in value.get("phases", []))
        files.append(dict(path=str(path), checksum=entry["checksum"], kind=kind))
    phases.extend((str(Path(report_path)), p) for p in report["result"].get("phases", []))
    if len(phases) > 512:
        raise ValueError("demand_history_bound")
    indexed = {}
    unique = {}
    for origin, phase in phases:
        identifier, run = phase.get("target"), phase.get("run_id")
        if identifier is None:
            continue
        if not isinstance(identifier, str) or not isinstance(run, str) or not run:
            raise ValueError("demand_history_phase_identity")
        requests, transport = phase.get("requests") or {}, phase.get("transport") or {}
        if (
            requests.get("run_id", run) != run
            or (phase.get("planner") or {}).get("run_id", run) != run
        ):
            raise ValueError("demand_history_phase_identity")
        key = identifier, run
        if key in unique:
            if unique[key] != phase:
                raise ValueError("demand_history_phase_conflict")
            continue
        unique[key] = phase
        unknown = bool(
            transport.get("remote_response_unknown", 0)
            or transport.get("dispatch_unknown", 0)
            or requests.get("api_requests_unknown", 0)
            or requests.get("pending_unknown", 0)
        )
        indexed.setdefault(identifier, []).append(
            dict(
                path=origin,
                phase_checksum=digest(phase),
                run_id=run,
                accepted=phase.get("accepted"),
                acceptance_key=phase.get("acceptance_key"),
                error=phase.get("error"),
                requests=requests,
                transport=transport,
                remote_or_dispatch_unknown=unknown,
            )
        )
    rows = []
    for row in coverage["rows"]:
        attempts = indexed.get(row["id"], [])
        unknown = any(p["remote_or_dispatch_unknown"] for p in attempts)
        state = row["state"]
        if not row["obtained"] and row["state"] != "input_missing" and attempts:
            state = (
                "historical_remote_or_dispatch_unknown"
                if unknown
                else "historical_attempt_without_current_observation"
            )
        rows.append(
            dict(
                id=row["id"],
                reported_state=row["state"],
                reconciled_state=state,
                method=row["planned_method"],
                component=row["component"],
                source_refs=row["source_refs"],
                execution_binding=row["execution_binding"],
                observation_reference=row["observation_reference"],
                obtained=row["obtained"],
                reviewed=row["reviewed"],
                e=row["e"],
                missing_reason=row["missing_reason"],
                historical_attempts=attempts,
                historical_unknown_present=unknown,
                next_step="preserve_unknown_no_replay_check_alternative_owner"
                if unknown
                else row["next_step"],
            )
        )
    return dict(
        schema_version="saved-demand-history-audit/1.0.0",
        report_checksum=digest(report),
        case_code=case,
        target=target,
        scope="explicit_checksum_bound_saved_ancestry_not_full_database_history",
        original_coverage=coverage["schema_version"],
        history_files=files,
        source_failures=source_failures,
        state_counts=dict(Counter(row["reconciled_state"] for row in rows)),
        rows=rows,
        observation_or_score_changed=False,
        current_execution_authorization=False,
        historical_unknown_replay_authorized=False,
        new_api_requests=0,
        database_writes=0,
    )
