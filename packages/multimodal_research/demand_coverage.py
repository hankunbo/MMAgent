"""Saved demand/observation/Review coverage, never a current-use authorization.

One outcome per requirement; absence of an observation is not evidence that no
request was sent. Prior campaign request ledgers still own retry permission.
"""

import html
import json
import math
from pathlib import Path

from packages.grounding.models import digest
from packages.object_store import checksum

VERSION = "demand-execution-coverage/1.1.0"
LEGACY_VERSION = "demand-execution-coverage/1.0.0"


def indexed(values, key):
    result = {}
    for value in values:
        identifier = value[key]
        if not isinstance(identifier, str) or not identifier or identifier in result:
            raise ValueError("demand_coverage_duplicate_identity")
        result[identifier] = value
    return result


def legacy_project(report):
    if (
        report.get("schema_version") != "dynamic-case-report/1.0.0"
        or not isinstance(report.get("requirements"), list)
        or not isinstance(report.get("result"), dict)
    ):
        raise ValueError("demand_coverage_report_contract")
    requirements = indexed(report["requirements"], "id")
    reviews = indexed((report.get("scores") or {}).get("reviews", []), "id")
    if not set(reviews) <= set(requirements):
        raise ValueError("demand_coverage_review_scope")
    acquired = report.get("acquisition_observations", {})
    rows = []
    for identifier, requirement in requirements.items():
        candidates = [
            value
            for value in (
                requirement.get("observation"),
                requirement.get("unscored_observation"),
                acquired.get(identifier),
            )
            if value is not None
        ]
        if any(value != candidates[0] for value in candidates):
            raise ValueError("demand_coverage_observation_conflict")
        observation = candidates[0] if candidates else None
        source = requirement.get("source_requirement") or {}
        review = reviews.get(identifier)
        judgment = (review or {}).get("judgment") or {}
        value = judgment.get("value")
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError("demand_coverage_invalid_score")
        attempts = [
            phase
            for phase in report["result"].get("phases", [])
            if phase.get("target") == identifier
        ]
        obtained = observation is not None and observation.get("status") == "obtained"
        if review:
            if value is not None and not obtained:
                raise ValueError("demand_coverage_score_without_observation")
            state = (
                "reviewed_unassessable"
                if value is None
                else "reviewed_neutral"
                if value == 0
                else "reviewed_directional"
            )
            next_step = (
                "inspect_review_gap_before_alternative"
                if value is None
                else "retain_current_bound_review"
            )
        elif obtained:
            state, next_step = "obtained_pending_review", "review_current_bound_observation"
        elif observation is not None:
            state, next_step = (
                "observation_not_interpretable",
                "inspect_partial_features_and_method_before_alternative",
            )
        elif source.get("state") == "input_gap":
            state, next_step = "input_missing", "record_input_gap_no_fabrication"
        elif attempts:
            state, next_step = (
                "attempt_recorded_without_observation",
                "verify_terminal_attempt_before_alternative",
            )
        elif source.get("state") == "scheduled":
            state, next_step = (
                "scheduled_without_observation",
                "verify_current_eligibility_and_history_before_dispatch",
            )
        else:
            state, next_step = "unresolved_requirement", "resolve_definition_input_and_method"
        rows.append(
            dict(
                id=identifier,
                examination=requirement.get("examination"),
                state=state,
                input_modalities=requirement.get("modality", []),
                component=source.get("component"),
                planned_method=source.get("method"),
                source_refs=requirement.get("sources", []),
                execution_binding=requirement.get("execution_binding"),
                definition=source,
                observation_reference=(observation or {}).get("reference"),
                observation_status=(observation or {}).get("status"),
                obtained=obtained,
                reviewed=review is not None,
                e=value,
                judgment=judgment if review else None,
                direction_assessment=requirement.get("direction_assessment"),
                missing_reason=source.get("reason"),
                attempt_status="phase_recorded" if attempts else "not_recorded_in_report",
                attempts=attempts,
                stop_context=report.get("acquisition_budget_stop"),
                next_step=next_step,
            )
        )
    basis = {k: v for k, v in report.items() if k != "demand_coverage"}
    return dict(
        schema_version=LEGACY_VERSION,
        basis_checksum=digest(basis),
        case_code=report["result"].get("case_code"),
        target=report.get("task", {}).get("target_code"),
        current_execution_authorization=False,
        interpretation="saved_report_projection_not_current_owner_or_complete_request_history",
        previous_attempts=report["result"].get("previous_attempts", []),
        counts=dict(
            requirements=len(rows),
            obtained=sum(row["obtained"] for row in rows),
            reviewed=sum(row["reviewed"] for row in rows),
            scored=sum(row["e"] is not None for row in rows),
            nonzero=sum(row["e"] is not None and row["e"] != 0 for row in rows),
            input_missing=sum(row["state"] == "input_missing" for row in rows),
            scheduled_without_observation=sum(
                row["state"] == "scheduled_without_observation" for row in rows
            ),
        ),
        state_counts={
            state: sum(r["state"] == state for r in rows)
            for state in sorted({r["state"] for r in rows})
        },
        fully_obtained=bool(rows) and all(row["obtained"] for row in rows),
        rows=rows,
    )


def project(report):
    """Separate current requirement coverage from historical record aliases.

    This is a projection of saved facts, not a new medical judgment or permission.
    Legacy persisted coverage is verified by legacy_project in inspect_saved.
    """
    result = legacy_project(report)
    result["schema_version"] = VERSION
    plan = report.get("active_acquisition", {}).get("plan", {})
    planned = indexed(plan.get("rows", []), "id")
    for row in result["rows"]:
        source = row["definition"]
        reason = source.get("missing_reason") or source.get("reason")
        row["missing_reason"] = reason
        if row["state"] == "obtained_pending_review" and (
            source.get("review_allowed") is False
            or source.get("definition", {}).get("interpretation_status") == "missing"
        ):
            row["state"] = "obtained_definition_pending"
            row["next_step"] = "resolve_current_definition_relation_before_review"
        elif row["state"] == "unresolved_requirement" and (
            planned.get(row["id"], {}).get("state") == "input_gap"
            or source.get("component") is None
            and isinstance(reason, str)
            and (
                reason.startswith("patient_input_missing:")
                or reason == "patient_modality_not_provided"
            )
        ):
            row["state"] = "input_missing"
            row["next_step"] = "record_input_gap_no_fabrication"
    rows = result["rows"]
    result["counts"]["input_missing"] = sum(r["state"] == "input_missing" for r in rows)
    result["state_counts"] = {
        state: sum(r["state"] == state for r in rows)
        for state in sorted({r["state"] for r in rows})
    }
    current = {r["id"] for r in rows}
    obtained = {r["id"] for r in rows if r["obtained"]}
    reviewed = {r["id"] for r in rows if r["reviewed"]}
    acquired = report.get("acquisition_observations", {})
    raw_obtained = {k for k, value in acquired.items() if value.get("status") == "obtained"}
    outside = raw_obtained - current
    declared = []
    for item in plan.get("measurement_fulfillment", {}).get("outcomes", []):
        if (
            item.get("state") == "shared"
            and item.get("requirement") in outside
            and item.get("fulfilled_by") in current
        ):
            declared.append(
                dict(
                    archived_requirement_id=item["requirement"],
                    current_requirement_id=item["fulfilled_by"],
                    current_requirement_reviewed=item["fulfilled_by"] in reviewed,
                )
            )
    result["observation_accounting"] = dict(
        schema_version="saved-observation-count-scope/1.0.0",
        raw_obtained_records=len(raw_obtained),
        current_requirement_obtained=len(obtained),
        reviewed_requirements=len(reviewed),
        current_obtained_without_review=sorted(obtained - reviewed),
        archival_obtained_ids=sorted(outside),
        declared_shared_aliases=declared,
        all_current_obtained_have_review=bool(obtained) and obtained <= reviewed,
        saved_completion_claim=report.get("completion", {}).get("all_obtained_reviewed"),
        independent_evidence_count=None,
        scope="saved_requirement_identities_not_independent_votes_or_current_authorization",
    )
    return result


def render_summary(coverage):
    labels = {
        "reviewed_directional": "已评阅：非零方向",
        "reviewed_neutral": "已评阅：无净方向",
        "reviewed_unassessable": "已评阅：尚不能判读",
        "obtained_pending_review": "已取得：待评阅",
        "obtained_definition_pending": "已取得：定义或目标关系待补",
        "observation_not_interpretable": "已有观察记录：尚不能解释目标",
        "input_missing": "缺少对应输入",
        "attempt_recorded_without_observation": "已有执行阶段：未取得观察",
        "scheduled_without_observation": "已列入计划：无观察记录",
        "unresolved_requirement": "需求或方法待核对",
    }
    rows = "".join(
        "<tr>"
        + "".join(
            "<td>" + html.escape(str(cell)) + "</td>"
            for cell in (
                row["examination"],
                labels[row["state"]],
                row["e"],
                row["missing_reason"] or "—",
            )
        )
        + "</tr>"
        for row in coverage["rows"]
    )
    return (
        '<section id="demand-coverage"><h2>逐项取证覆盖</h2>'
        "<p>本表按当前需求分别统计已取得、已评阅与缺输入；历史别名不增加疾病票数。无观察记录不证明从未派发；"
        "再次执行前仍须核对原请求账本、当前来源及工具资格。本表不修改评分。</p>"
        '<table border="1"><thead><tr><th>检查</th><th>实际状态</th><th>e</th>'
        "<th>输入缺口</th></tr></thead><tbody>" + rows + "</tbody></table></section>"
    )


def inspect_saved(path):
    path = Path(path)
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("demand_coverage_report_bound")
    raw = path.read_bytes()
    report = json.loads(raw)
    result = project(report)
    saved = report.get("demand_coverage")
    if saved is not None:
        version = saved.get("schema_version")
        if version not in (LEGACY_VERSION, VERSION):
            raise ValueError("demand_coverage_saved_version")
        expected = legacy_project(report) if version == LEGACY_VERSION else result
        if saved != expected:
            raise ValueError("demand_coverage_saved_projection_changed")
    return dict(
        report_path=str(path),
        report_bytes_checksum=checksum(raw),
        coverage=result,
        saved_coverage_version=saved.get("schema_version") if saved else None,
        original_projection_verified=saved is not None,
        api_requests=0,
        database_writes=0,
    )
