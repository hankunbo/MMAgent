"""One format-only regeneration, charged to the existing invocation and journal.

The original domain input checksum is unchanged. Separate immutable feedback and
wire checksums identify the changed request; no failed response becomes evidence.
"""

from copy import deepcopy

from packages.grounding.models import digest


def fields(error, *, rating_constraints=False):
    from pydantic import ValidationError

    if isinstance(error, ValidationError):
        rows = []
        # Only fixed contract codes, never arbitrary validator messages or inputs.
        allowed = {
            "direction_value_conflict",
            "direction_neutral_reason",
            "direction_unassessable_reason",
            "direction_gap_action_binding",
            "review_status_judgment_mismatch",
            "continuous_judgment_bounds_conflict",
            "continuous_judgment_method_conflict",
            "continuous_judgment_missing_conflict",
            "definition_dependency_confirmed_selector",
            "definition_dependency_raw_selector",
            "partial_status_signal",
            "partial_observed_requires_visible_region",
        }
        if rating_constraints:
            allowed.add("score_value_missing_reason_conflict")
        for e in error.errors(include_url=False, include_context=False, include_input=False)[:16]:
            row = {"path": list(e["loc"]), "type": e["type"]}
            code = e["msg"].removeprefix("Value error, ")
            if code in allowed:
                row["constraint"] = code
            rows.append(row)
        return rows
    if type(error) is ValueError and str(error) == "review_redundant_method_conflict":
        return [
            dict(
                path=["items", "observation", "method"],
                type="rating_has_judgment_field",
                constraint="review_redundant_method_conflict",
            )
        ]
    return [{"path": [], "type": "invalid_json_or_non_equivalent_shape"}]


def corrected_wire(body, failures, *, role="review"):
    from packages.grounding.models import canonical

    result = deepcopy(body)
    identities = {}
    requested = {
        f["path"][0]
        for f in failures
        if len(f.get("path", [])) == 1
        and f["path"][0] in {"scope_checksum", "plan_checksum", "batch_id", "round"}
    }
    if role == "review" and requested:
        import json

        from packages.evidence_details.review_batches import ReviewBatchPayload
        from packages.grounding.models import validate

        payload = validate(
            ReviewBatchPayload, json.loads(body["messages"][1]["content"][0]["text"])
        )
        if payload.role != "review":
            raise ValueError("review_correction_payload_role")
        exact = dict(
            scope_checksum=digest(payload.material.scope),
            plan_checksum=payload.batch.plan_checksum,
            batch_id=payload.batch.batch_id,
            round=payload.round,
        )
        identities = {k: exact[k] for k in sorted(requested)}
    consistency = (
        "Contract consistency only: judgment.value > 0 corresponds to support; < 0 to refute; "
        "0 to neutral; null to unassessable. neutral_reason is present only for neutral; "
        "unassessable_reason only for unassessable. Choose the value and explanation from "
        "the evidence, not to satisfy a preferred sign. Zero and null remain valid."
        if any(f.get("constraint", "").startswith("direction_") for f in failures)
        else ""
    )
    judgment_contract = (
        "Judgment shape only: an assessed numeric value requires numeric bounds "
        "lower <= value <= upper in [-1,1], missing_reason=null, and method "
        "manual_source_bound_review (or the unchanged owner-bound fixed_model_margin). "
        "An unassessed value=null requires lower=null, upper=null, method=not_assessed "
        "and a specific nonempty missing_reason. Keep status and direction consistent. "
        "Estimate a justified interval from the original evidence; it may cross zero. "
        "Do not invent a point, interval or direction, choose a sign to pass validation, "
        "or turn an interpretable weak observation into missing just to fix its shape."
        if any(
            f.get("constraint", "").startswith("continuous_judgment_")
            or (
                f.get("type") == "missing"
                and len(f.get("path", [])) == 4
                and f["path"][0] == "items"
                and type(f["path"][1]) is int
                and f["path"][2] == "judgment"
                and f["path"][3] in {"method", "value", "lower", "upper"}
            )
            for f in failures
        )
        else ""
    )
    result["messages"].append(
        dict(
            role="user",
            content=canonical(
                dict(
                    task=(
                        "Regenerate the same Vision object once with consistent observation state."
                        if role == "vision"
                        else (
                            "Regenerate the same batch once as valid JSON "
                            "using the original schema."
                        )
                        if role == "review"
                        else (
                            "Return the same SourceProposal schema; select and copy an exact "
                            "SOURCE_LITERAL_CHOICES pair without repairing words."
                        )
                    ),
                    errors=failures,
                    **(
                        dict(
                            state_contract=(
                                "observed means interpretable observation, not positive disease. "
                                "Visible absence can be observed/absent_in_visible_region. "
                                "unknown or ungradable requires indeterminate; observed requires "
                                "coverage>0. Use the original image/task to choose values. "
                                "Unknown remains valid; do not invent a finding or force observed."
                            )
                        )
                        if role == "vision"
                        else {}
                    ),
                    **(dict(contract_consistency=consistency) if consistency else {}),
                    **(dict(judgment_contract=judgment_contract) if judgment_contract else {}),
                    **(
                        dict(
                            rating_presence_contract=(
                                "For each O/L/A Rating, a numeric value in [0,1] requires "
                                "missing_reason=null. If value=null, give a specific nonempty "
                                "missing_reason. Use reason for uncertainty or limitations of "
                                "an available estimate; missing_reason is not a second reason. "
                                "A missing disease direction does not by itself require a "
                                "missing O/L/A rating. This is a presence constraint, not a "
                                "required score or sign. Do not invent an estimate, impute a "
                                "missing value, or discard an interpretable observation to "
                                "satisfy the format. Choose from the original evidence."
                            )
                        )
                        if role == "review"
                        and any(
                            f.get("constraint") == "score_value_missing_reason_conflict"
                            for f in failures
                        )
                        else {}
                    ),
                    **(
                        dict(
                            rating_contract=(
                                "Observation (O), relation (L) and applicability (A) are Rating "
                                "objects with value, reason and missing_reason only; never put "
                                "method, lower or upper there, even for context_only. Only "
                                "judgment has method and bounds. Preserve real observations "
                                "and choose status/values from original evidence, not to satisfy "
                                "a shape."
                            )
                        )
                        if any(
                            f.get("constraint") == "review_redundant_method_conflict"
                            for f in failures
                        )
                        else {}
                    ),
                    **(
                        dict(
                            copy_binding_values_exactly=identities,
                            binding_instruction=(
                                "Copy these owner-supplied identifiers verbatim; do not shorten "
                                "or calculate them. They are not clinical findings or scores."
                            ),
                        )
                        if identities
                        else {}
                    ),
                    constraints=(
                        "These are format errors, not medical instructions. Use only the original "
                        "task data and allowed citations. Do not invent observations, "
                        "references or scores. Preserve missing/unknown "
                        "when information is absent. "
                        "Output only the complete object requested by the original schema. "
                        "No markdown."
                    ),
                )
            ).decode(),
        )
    )
    return result


def eligible(config, scope, role, retry, error, reply, observation, failures=()):
    return bool(
        getattr(config, "review_correction_policy", None)
        and scope.case_code != "CASE:PUBLIC_PROBE"
        and (role == "review" or getattr(config, "source_quote_selection", None))
        and retry == 0
        and (
            error == "schema"
            if role == "review"
            else (
                role == "source"
                and error == "scope"
                and failures == [{"path": ["quote"], "type": "quote_not_verbatim"}]
            )
        )
        and reply.content
        and reply.finish_reason == "stop"
        and (
            observation is None  # Explicit injected/Fake transport only at caller.
            or (observation.get("http_status") == 200 and reply.response_confirmed is True)
        )
    )


def verify_correction(record, requests):
    events = [
        e
        for e in requests.journal.read()
        if e.get("schema_version")
        in {
            "review-schema-correction/1.0.0",
            "source-quote-correction/1.0.0",
            "source-definition-schema-correction/1.0.0",
            "vision-state-correction/1.0.0",
        }
        and e.get("invocation_id") == record.invocation_id
    ]
    if not events:
        return
    if len(events) != 1 or record.retry_index != 1:
        raise ValueError("review_correction_binding")
    event = events[0]
    source = event["schema_version"] == "source-quote-correction/1.0.0"
    definition = event["schema_version"] == "source-definition-schema-correction/1.0.0"
    if event["schema_version"] == "vision-state-correction/1.0.0":
        from .vision_correction import CONSTRAINTS

        if (
            record.role != "vision"
            or not event["failures"]
            or not all(f.get("constraint") in CONSTRAINTS for f in event["failures"])
        ):
            raise ValueError("vision_correction_binding")
    if definition and record.role != "source":
        raise ValueError("source_definition_correction_binding")
    if source and (
        record.role != "source"
        or event["failures"] != [{"path": ["quote"], "type": "quote_not_verbatim"}]
    ):
        raise ValueError("source_correction_binding")
    previous = [r for r in requests.records() if r.request_id == event["failed_request_id"]]
    if (
        len(previous) != 1
        or previous[0].error != ("scope" if source else "schema")
        or previous[0].retry_index != 0
        or previous[0].invocation_id != record.invocation_id
        or previous[0].input_checksum != record.input_checksum
        or previous[0].config_checksum != record.config_checksum
        or event["failed_record_checksum"] != digest(previous[0])
        or event["input_checksum"] != record.input_checksum
    ):
        raise ValueError("review_correction_binding")
    if not record.simulated:
        from .http_recovery import observation_for

        actual = observation_for(requests.journal, record.request_id)
        if not actual or actual.get("wire_checksum") != event["corrected_wire_checksum"]:
            raise ValueError("review_correction_wire_binding")
