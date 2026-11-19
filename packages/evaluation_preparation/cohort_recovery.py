"""Readonly recovery audit through original case/acceptance/report owners.

This is not a runner, a new scoring policy or an expansion permission. It keeps
historical UNKNOWN and historical unassessable judgments visible and distinct
from unattempted work. No persisted medical value or original report is changed.
"""

import json
import re
from pathlib import Path

from agents.expert_agent.reporting import read_research_report
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.acceptance_models import DomainState
from packages.grounding.models import digest, validate
from packages.multimodal_research.campaign_renewal import completed_parent
from packages.multimodal_research.report_storage import unpack
from packages.research_runtime.agent_records import RuntimeAgentJournal


def read_file(path, *, root=None):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("recovery_audit_symlink")
    path = path.resolve(strict=True)
    if root is not None and not path.is_relative_to(Path(root).resolve(strict=True)):
        raise ValueError("recovery_audit_history_scope")
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("recovery_audit_file_bound")
    return json.loads(path.read_bytes())


def gap_inventory(report_path, evidence_path, case):
    """Project preserved evidence, not infer medical directions or authorize replay.

    An old audit binds this report's bytes. A future sender still must recheck
    current original owners, response availability, attempt identity and budget.
    """
    report, evidence = read_file(report_path), read_file(evidence_path)
    audit = evidence["cases"][case]["audit"]
    if (
        audit.get("original_owner_verified") is not True
        or audit.get("original_score_json_html_equal") is not True
        or audit["report_checksum"] != digest(report)
    ):
        raise ValueError("gap_inventory_report_audit_binding")
    reviews = {r["id"]: r for r in report["scores"]["reviews"]}
    requirements = report["requirements"]
    if len(reviews) != len(report["scores"]["reviews"]) or len(
        {r["id"] for r in requirements}
    ) != len(requirements):
        raise ValueError("gap_inventory_duplicate_identity")
    unknown = audit.get("unconfirmed_phase_requests", [])
    retained = audit.get("retained_historical_reviews", {})
    rows = []
    for index, requirement in enumerate(requirements):
        key = requirement["id"]
        sections = requirement.get("presentation_sections") or {}
        acquisition = sections.get("acquisition") or {}
        observation, pointer = {}, None
        for suffix, candidate in (
            ("observation", requirement.get("observation")),
            ("unscored_observation", requirement.get("unscored_observation")),
            ("presentation_sections/observation", sections.get("observation")),
        ):
            if candidate:
                observation, pointer = candidate, f"/requirements/{index}/{suffix}"
                break
        review = reviews.get(key)
        requests = [r for r in unknown if r.get("target") == key]
        missing = acquisition.get("missing_reason")
        if review is not None:
            value = review["judgment"]["value"]
            status = (
                "reviewed_null"
                if value is None
                else "reviewed_neutral"
                if value == 0
                else "reviewed_nonzero"
            )
            action = (
                "inspect_specific_relation_or_method_gap"
                if value is None
                else "retain_accepted_review"
            )
        elif key in retained:
            status, action = "retained_historical_review", "inspect_original_reason_before_new_work"
        elif observation.get("reference") and observation.get("status") != "obtained":
            # A saved observation's unknown clinical status is not a transport
            # UNKNOWN. Keep both the historical requests and exact material;
            # this projection never proves that a new request may be sent.
            status, action = (
                "saved_observation_requires_interpretation",
                "inspect_saved_observation_method_scope_and_target_relation",
            )
        elif requests:
            status, action = "unknown_attempt", "check_saved_response_then_controlled_new_attempt"
        elif isinstance(missing, str) and (
            missing.startswith("patient_input_missing:")
            or missing == "patient_modality_not_provided"
        ):
            status, action = "patient_input_missing", "retain_gap_do_not_invent_patient_observation"
        elif observation.get("status") == "obtained":
            status, action = "obtained_without_current_review", "verify_original_review_readiness"
        else:
            status, action = (
                "requires_owner_inspection",
                "inspect_input_method_and_execution_history",
            )
        rows.append(
            dict(
                id=key,
                examination=requirement.get("examination"),
                status=status,
                saved_status=requirement.get("status"),
                input_modalities=acquisition.get("input_modalities"),
                method=acquisition.get("method") or observation.get("method"),
                missing_reason=missing,
                judgment=review.get("judgment") if review else None,
                observation_reference=observation.get("reference"),
                observation_status=observation.get("status"),
                observation_pointer=pointer,
                observation_checksum=digest(observation) if observation else None,
                unknown_request_ids=[r["request_id"] for r in requests],
                next_action=action,
                dispatch_authorized=False,
            )
        )
    return dict(
        schema_version="terminal-case-gap-inventory/1.1.0",
        case=case,
        report_checksum=digest(report),
        audit_checksum=digest(audit),
        scope="saved_evidence_projection_not_current_authorization",
        counts={
            s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})
        },
        rows=rows,
        cumulative_requests=audit["cumulative_requests"],
        unknown_requests=unknown,
        source_failures=audit.get("source_failures", []),
        new_api_requests=0,
        new_scores=0,
        database_writes=0,
    )


def ancestral_result(entry, root):
    """Only the two original result layouts, never a directory scan or guess."""
    directory = Path(entry["directory"]).absolute()
    if entry.get("kind"):
        candidates = [directory / "stage-failure.json"]
    else:
        candidates = [directory / "result.json", directory / "evidence/result.json"]
    found = [p for p in candidates if p.is_file()]
    if len(found) != 1:
        raise ValueError("recovery_audit_ancestry_layout")
    value = read_file(found[0], root=root)
    if (
        digest(value) != entry["checksum"]
        or type(entry["requests"]) is not int
        or entry["requests"] < 0
    ):
        raise ValueError("recovery_audit_ancestry_binding")
    return found[0], value


def phase_owner(campaign, phase):
    docs = campaign.documents(phase["run_id"])
    control = campaign.sessions.read(docs.run_id)
    if (
        docs.read("dynamic-phase-result") != (phase,)
        or control is None
        or control.phase != "stopped"
        or control.pending is not None
    ):
        raise ValueError("recovery_audit_phase_terminal")
    events = RuntimeAgentJournal(docs).read()
    reservations = [e["request_id"] for e in events if e.get("phase") == "reserved"]
    records = [e["record"] for e in events if e.get("phase") == "completed"]
    confirmations = {
        e["request_id"]: e["confirmed"] for e in events if e.get("phase") == "response_confirmation"
    }
    if (
        len(reservations) != len(set(reservations))
        or len(records) != len(reservations)
        or {r["request_id"] for r in records} != set(reservations)
        or any(r["simulated"] for r in records)
    ):
        raise ValueError("recovery_audit_request_accounting")
    accepted = None
    if phase["accepted"]:
        key = phase["acceptance_key"]
        values = docs.read("accepted:" + key)
        if not values or len(values) != 4 or values[3]["status"] != "accepted":
            raise ValueError("recovery_audit_acceptance_missing")
        if phase["target"] is None:
            state = validate(DomainState, docs.read("dynamic-phase-input")[0])
            accepted = read_research_report(
                ResearchAcceptance(owner=None, documents=docs, requests=None), key=key, parent=state
            )
    elif phase.get("acceptance_key") is not None:
        raise ValueError("recovery_audit_failed_acceptance")
    requests = [
        dict(
            run_id=docs.run_id,
            request_id=r["request_id"],
            role=r["role"],
            target=phase["target"],
            response_confirmed=confirmations.get(r["request_id"]),
            error=r["error"],
            api_requests=r["api_requests"],
            replay_authorized=False,
        )
        for r in records
    ]
    return docs, accepted, requests


def retained_reviews(current, histories):
    """Require exact observations and original accepted null reasons, not a label.

    These remain historical judgments outside the current review scope. They do
    not add votes or become current medical-use permissions.
    """
    current_rows = {r["id"]: r for r in current["requirements"]}
    wanted = set(current.get("completion", {}).get("retained_context_not_newly_reviewed_ids", []))
    found = {}
    for report, proof in histories:
        old_rows = {r["id"]: r for r in report["requirements"]}
        for review in report["scores"]["reviews"]:
            key = review["id"]
            if key not in wanted or review["judgment"]["value"] is not None:
                continue
            old = old_rows[key].get("observation") or old_rows[key].get("unscored_observation")
            new = current_rows[key].get("observation") or current_rows[key].get(
                "unscored_observation"
            )
            if old is None or old != new:
                continue
            direction = report["scores"].get("direction_assessments", {}).get(key)
            reason = (direction or {}).get("unassessable_reason") or review.get("reason")
            if not reason:
                continue
            found[key] = dict(
                origin=proof,
                assessment_checksum=digest(review),
                observation_checksum=digest(old),
                original_reason=reason,
                direction_assessment=direction,
                interpretation="historical_unassessable_not_new_current_review",
                source_definition_unchanged=all(
                    old_rows[key].get(k) == current_rows[key].get(k)
                    for k in ("sources", "source_requirement")
                ),
                contributes_new_vote=False,
            )
    return found, sorted(wanted - set(found))


def check_tree(report, page):
    from packages.evaluation_preparation.evidence_tree_v3 import bundle, render_packet, unbundle

    match = re.search(
        r'<script id="evidence-view" type="application/json">(.*?)</script>', page, re.S
    )
    if not match:
        return False
    packet = unbundle(bundle(json.loads(match.group(1))))
    # The renderer represents an unscored report with an empty display mapping.
    # This is not a medical zero; retain the exact source report checksum.
    expected_scores = {} if report["scores"] is None else report["scores"]
    if packet["scores"] != expected_scores or packet["report_content_checksum"] != digest(report):
        raise ValueError("recovery_audit_tree_binding")
    if render_packet(packet).split("</body>")[0] not in page:
        raise ValueError("recovery_audit_tree_render")
    return True


def recorded_closure(campaign, original, directory, report, report_path, history_root):
    """Bind a separate local export to the original saved acceptance, never replay.

    Original phases remain immutable (including their failed outer flags). The
    recorded-finalization owner proves the newer acceptance and current bindings.
    No scan for a convenient successful result and no caller-supplied authority.
    """
    from packages.evidence_details.recorded_finalization import REPORT_KEY, RESULT_KEY
    from packages.multimodal_research.campaign_renewal import recorded_acquisition_origin

    directory = Path(directory).absolute()
    result = read_file(directory / "result.json", root=history_root)
    old = read_file(directory / "original-failure.json", root=history_root)
    if not original.get("phases") or old != original:
        raise ValueError("recovery_audit_local_original_binding")
    run_id = original["phases"][-1]["run_id"]
    proof = recorded_acquisition_origin(campaign, run_id)
    docs = campaign.documents(run_id)
    if (
        docs.read(RESULT_KEY) != (result,)
        or proof["closure_checksum"] != digest(result)
        or result.get("original_result_checksum") != digest(original)
        or result.get("previous_attempts", []) != original.get("previous_attempts", [])
        or result.get("new_requests") != 0
        or result.get("original_requests") != original["new_requests"]
    ):
        raise ValueError("recovery_audit_local_result_binding")
    saved, envelope = unpack(docs.read(REPORT_KEY), key=REPORT_KEY)
    key = proof["confirmation"]["acceptance_key"]
    state = validate(DomainState, docs.read("dynamic-phase-input")[0])
    accepted = read_research_report(
        ResearchAcceptance(owner=None, documents=docs, requests=None), key=key, parent=state
    )
    if (
        saved != report
        or report["result"] != result
        or report["scores"] != accepted["scores"]
        or report.get("original_review") != accepted
        or read_file(directory / "review/review-report.json", root=history_root) != accepted
        or Path(report_path).with_suffix(".html").read_text() != envelope["html"]
    ):
        raise ValueError("recovery_audit_local_report_binding")
    return result, envelope, proof


def audit_case(directory, report_path, history_root, *, local_closure=None):
    """Original-owner terminal read. Deliberately does not sign a four-case gate."""
    directory = Path(directory).absolute()
    report = read_file(report_path)
    campaign, original = completed_parent(directory)
    current_report, local_proof = None, None
    if local_closure is None:
        result = original
    else:
        result, current_report, local_proof = recorded_closure(
            campaign, original, local_closure, report, report_path, history_root
        )
    if (
        report["result"] != result
        or not result["complete"]
        or result["hidden_labels_used"] is not False
    ):
        raise ValueError("recovery_audit_result_binding")
    attempts = result.get("previous_attempts", [])
    if len(attempts) > 64:
        raise ValueError("recovery_audit_history_bound")
    histories, phase_records, source_failures = [], {}, []
    seen, requests = set(), {}
    for entry in attempts:
        path, old = ancestral_result(entry, history_root)
        if path in seen:
            raise ValueError("recovery_audit_duplicate_ancestry")
        seen.add(path)
        if entry.get("kind"):
            docs = campaign.documents(old["run_id"])
            if docs.read("multimodal-stage-failure") != (old,):
                raise ValueError("recovery_audit_source_failure_binding")
            source_failures.append(dict(path=str(path), checksum=digest(old), kind=entry["kind"]))
            continue
        if old["case_code"] != result["case_code"]:
            raise ValueError("recovery_audit_case_binding")
        for phase in old["phases"]:
            rid = phase["run_id"]
            if rid in phase_records and phase_records[rid] != phase:
                raise ValueError("recovery_audit_phase_conflict")
            phase_records[rid] = phase
    # Audit immutable original phase facts. A local closure does not rewrite a
    # previously failed phase to look accepted; its separate proof is above.
    for phase in original["phases"]:
        rid = phase["run_id"]
        if rid in phase_records and phase_records[rid] != phase:
            raise ValueError("recovery_audit_phase_conflict")
        phase_records[rid] = phase
    for phase in phase_records.values():
        docs, accepted, phase_requests = phase_owner(campaign, phase)
        for request in phase_requests:
            key = request["request_id"]
            if key in requests and requests[key] != request:
                raise ValueError("recovery_audit_request_reused")
            requests[key] = request
        if accepted is None:
            continue
        key = (
            "accepted-aggregation-report" if phase.get("execution_kind") else "dynamic-case-report"
        )
        values = docs.read(key)
        # An old accepted Review may have failed its later report export. Its
        # acceptance is not lost, but no unbound local file supplies a proof.
        if not values:
            continue
        old_report, envelope = unpack(values, key=key)
        if old_report["scores"] != accepted["scores"]:
            raise ValueError("recovery_audit_original_score_binding")
        histories.append(
            (
                old_report,
                dict(
                    run_id=docs.run_id,
                    acceptance_key=phase["acceptance_key"],
                    report_checksum=digest(old_report),
                ),
            )
        )
        if local_closure is None and docs.run_id == result["phases"][-1]["run_id"]:
            if (
                old_report != report
                or Path(report_path).with_suffix(".html").read_text() != envelope["html"]
            ):
                raise ValueError("recovery_audit_report_export_binding")
            current_report = envelope
    if current_report is None:
        raise ValueError("recovery_audit_current_report_missing")
    retained, missing = retained_reviews(report, histories)
    reviews = report["scores"]["reviews"]
    numeric = [r for r in reviews if r["judgment"]["value"] is not None]
    cumulative = sum(a["requests"] for a in attempts) + original["new_requests"]
    if not 0 <= cumulative <= 40 or result.get("cumulative_requests", cumulative) != cumulative:
        raise ValueError("recovery_audit_budget")
    unknown = [r for r in requests.values() if r["response_confirmed"] is not True]
    value = dict(
        schema_version="terminal-case-recovery-audit/1.1.0"
        if local_proof
        else "terminal-case-recovery-audit/1.0.0",
        input=str(directory),
        case_code=result["case_code"],
        sample=result["sample"],
        original_owner_verified=True,
        current_medical_consumption_authorized=False,
        original_score_json_html_equal=True,
        tree_packet_verified=check_tree(report, current_report["html"]),
        report_checksum=digest(report),
        report_json=str(report_path),
        model=campaign.config.transport.model,
        provider_checksum=digest(campaign.config),
        counts=dict(
            requirements=len(report["requirements"]),
            current_reviews=len(reviews),
            numeric=len(numeric),
            nonzero=sum(r["judgment"]["value"] != 0 for r in numeric),
            neutral=sum(r["judgment"]["value"] == 0 for r in numeric),
            current_null=len(reviews) - len(numeric),
        ),
        completion=report["completion"],
        retained_historical_reviews=retained,
        retained_without_review_proof=missing,
        cumulative_requests=cumulative,
        phase_request_count=len(requests),
        requests=list(requests.values()),
        unconfirmed_phase_requests=unknown,
        source_failures=source_failures,
        historical_reports=[p for _, p in histories],
        all_registered_phases_stopped=True,
        new_api_requests=0,
        database_writes=0,
        four_case_gate_passed=False,
        expansion_authorized=False,
    )
    if local_proof is not None:
        value.update(
            local_closure=str(Path(local_closure).absolute()),
            local_confirmation=local_proof,
            original_attempt_requests=original["new_requests"],
            original_failure_checksum=digest(original),
            original_failure_unchanged=True,
        )
    return value
