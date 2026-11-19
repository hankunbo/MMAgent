"""Bounded gap decisions over the original definition executor and Runtime journal.

No second campaign, request budget, medical classifier or tool installer. A
decision authorizes neither a patient observation nor a disease direction.
"""

import json

from packages.grounding.models import canonical, digest
from packages.llm_provider.provider import ProviderError

LIMITS = {
    "definition_close_reserve",
    "definition_review_reserve",
    "definition_review_time_reserve",
    "definition_operation_budget",
    "definition_public_request_budget",
}
FINAL = "definition:gap:final-candidate"


def classify(candidate):
    """Describe owner outcomes, never infer a medical relation from wording."""
    reasons = {r.get("reason") for r in candidate["backlog"]}
    issues = []
    if reasons & {"public_search_incomplete", "source_fetch_technical_failure"}:
        issues.append("retrieval_technical_error_not_empty_result")
    if candidate["added"] or candidate["replaced"]:
        status = "adopted_increment"
    elif "same_material_no_new_context" in reasons:
        status = "unchanged_material"
    elif reasons & {
        "definition_joint_input_method_not_installed",
        "definition_input_method_mismatch",
        "interpretation_method_or_unit_mismatch",
    }:
        status = "method_gap"
    elif candidate["merged"]:
        status = "equivalent_requirement_retained"
    elif candidate["found"]:
        status = "no_usable_definition"
    else:
        status = "no_usable_material"
    for row in candidate["rows"]:
        if row["id"] not in candidate["added"]:
            continue
        if row.get("missing_reason"):
            issues.append("patient_input_or_method_missing")
        if row.get("review_allowed") is False:
            issues.append("interpretation_dependency_unresolved")
    return dict(
        outcome=status,
        issues=sorted(set(issues)),
        reasons=sorted(r for r in reasons if r),
        added=candidate["added"],
        replaced=[r["id"] for r in candidate["replaced"]],
        merged=[r["existing"] for r in candidate["merged"]],
        # A method backlog is not an automatically qualified new Tool.
        method_extension="not_dispatched_requires_original_tool_owner"
        if status == "method_gap"
        else None,
    )


def empty(expansion):
    return dict(
        rows=list(expansion.original_rows),
        found={},
        accepted={},
        added=[],
        merged=[],
        replaced=[],
        backlog=[],
        material_fingerprint=digest([]),
        evaluation_fingerprint=digest([]),
    )


def finish(expansion, candidate, summary, until):
    expansion.gap_expansion_summary = summary
    expansion.sources.restore(list(candidate["found"].values()))
    if until in {"sources", "definitions"}:
        return expansion.checkpoint(
            until,
            dict(
                accepted=candidate["accepted"],
                rejected=candidate["backlog"],
                materials=list(candidate["found"].values()),
                gap_expansion=summary,
            ),
        )
    return expansion.commit_revision(
        candidate["rows"],
        candidate["found"],
        candidate["accepted"],
        candidate["added"],
        candidate["merged"],
        candidate["backlog"],
        replaced=candidate["replaced"],
        stop=summary["stop_reason"],
    )


def advance(expansion, *, until=None):
    expansion.control()
    saved = expansion.documents.read(FINAL)
    if saved:
        value = saved[0]
        if value["binding"] != expansion.binding:
            raise ValueError("definition_gap_candidate_binding")
        verify_summary(expansion, value["summary"])
        # Reuse immutable computation, NOT authorization. commit_revision/current
        # rebuild accepted definitions and their current input/source projections.
        return finish(expansion, value["candidate"], value["summary"], until)
    plan = [p for p in expansion.query_plan() if p["selected"]]
    candidate, attempts = empty(expansion), []
    stop = "current_definitions_reused" if not plan else "configured_directions_exhausted"
    expansion.gap_seen_materials = set()
    try:
        for index, direction in enumerate(plan[: expansion.policy.max_rounds]):
            expansion.gap_attempt = index
            key = f"definition:gap:{index}:decision"
            previous = expansion.documents.read(key)
            try:
                if previous and until != "sources":
                    record = previous[0]
                    if record["binding"] != expansion.binding or record["direction"] != direction:
                        raise ValueError("definition_gap_attempt_binding")
                    candidate = record["candidate"]
                    expansion.sources.restore(list(candidate["found"].values()))
                else:
                    candidate = expansion._advance(
                        until="sources" if until == "sources" else None,
                        planned_queries=[direction],
                        candidate_only=True,
                    )
            except ProviderError as exc:
                if exc.code not in LIMITS:
                    raise
                stop = exc.code
                break
            if until == "sources":
                return candidate
            decision = classify(candidate)
            adopted = bool(candidate["added"] or candidate["replaced"])
            technical = "retrieval_technical_error_not_empty_result" in decision["issues"]
            has_next = not technical and index + 1 < min(len(plan), expansion.policy.max_rounds)
            action = (
                "schedule_adopted_requirements"
                if adopted
                else "try_distinct_public_direction"
                if has_next
                else "continue_existing_qualified_requirements"
            )
            record = json.loads(
                canonical(
                    dict(
                        schema_version="definition-gap-attempt/1.0.0",
                        binding=expansion.binding,
                        attempt=index,
                        direction=direction,
                        candidate=candidate,
                        decision=dict(decision, action=action),
                    )
                )
            )
            with expansion.documents.session():
                expansion.control()
                expansion.documents.write(key, (record,))
            attempts.append(dict(key=key, checksum=digest(record), **record["decision"]))
            expansion.gap_seen_materials.add(candidate["evaluation_fingerprint"])
            if adopted:
                stop = "adopted_increment_ready"
                break
            if technical:
                stop = "retrieval_technical_backlog"
                break
        summary = dict(
            schema_version="definition-gap-loop/1.0.0",
            binding=expansion.binding,
            attempts=attempts,
            stop_reason=stop,
            next_action="schedule_adopted_requirements"
            if candidate["added"] or candidate["replaced"]
            else "continue_existing_qualified_requirements",
            shared_budget=True,
            reset_deadline=False,
            clinical_success_claimed=False,
            mandatory_increment=False,
        )
        value = dict(binding=expansion.binding, candidate=candidate, summary=summary)
        with expansion.documents.session():
            expansion.control()
            expansion.documents.write(FINAL, (value,))
        return finish(expansion, candidate, summary, until)
    finally:
        expansion.gap_attempt = None
        expansion.gap_seen_materials = set()


def verify_summary(expansion, summary):
    if summary["binding"] != expansion.binding:
        raise ValueError("definition_gap_summary_binding")
    if summary.get("terminal_source_deferral"):
        continuation = getattr(expansion.campaign, "definition_continuation", None)
        if (
            continuation is None
            or not continuation.defer
            or continuation.reference != summary["terminal_source_deferral"]
            or summary["stop_reason"] != "terminal_source_technical_backlog"
            or summary["attempts"]
            or summary.get("resolved_nonadopted_definitions")
            != getattr(continuation, "proof", {}).get("resolved_merges")
        ):
            raise ValueError("definition_gap_deferral_binding")
        continuation.check_attempts()
    for attempt in summary["attempts"]:
        saved = expansion.documents.read(attempt["key"])
        if not saved or digest(saved[0]) != attempt["checksum"]:
            raise ValueError("definition_gap_attempt_binding")
        if saved[0]["binding"] != expansion.binding:
            raise ValueError("definition_gap_attempt_binding")
