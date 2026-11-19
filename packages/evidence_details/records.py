"""Normalize persisted v4 reviews and typed vascular facts; never assign new scores."""

import copy

from packages.evidence_details.models import digest, resolve
from packages.evidence_details.records_models import EvidenceRecords

STATUSES = {
    "direction_reviewed": "assessed",
    "method_mismatch": "method_mismatch",
    "reference_missing": "reference_missing",
    "context_only": "context_only",
    "ungradable": "observation_ungradable",
    "input_missing": "input_missing",
    "duplicate_or_intermediate": "intermediate",
    "not_reviewed": "not_reviewed",
}


def ref(base, pointer):
    return {"pointer": pointer, "checksum": digest(resolve(base, pointer))}


def _missing(reason):
    return {"value": None, "reason": reason, "missing_reason": reason}


def _values(assessment, review, obtained):
    if not obtained or not assessment:
        return []
    values = assessment["measurements"]
    metric = assessment["comparison"].get("metric")
    if metric:
        values = [v for v in values if v["name"] == metric]
    if review["id"] == "criterion:AGE":
        values = [v for v in values if v["name"] == "age"]
    # Neighboring pixel measurements cannot become a visual notch or vessel observation.
    if review["kind"] == "visual" or assessment["comparison"].get("phenomenon") and not metric:
        values = [v for v in values if isinstance(v.get("value"), str)]
    return [
        {
            "name": v["name"],
            "value": v.get("value"),
            "unit": v.get("unit") or "",
            "missing_reason": (v.get("missing_reason") or "原记录未取得该值")
            if v.get("value") is None
            else None,
        }
        for v in values
    ]


def _vascular_values(record, node):
    fields = {
        "VASCULAR_SKELETON_DENSITY": ("whole_image", "skeleton_density", "px^-1"),
        "VASCULAR_FRACTAL": ("skeleton_fractal", "value", "1"),
        "AV_TORTUOSITY": ("tortuosity", "mean", "1"),
        "ARTERIOLAR_CALIBER": ("annulus", "width_proxy_disc_ratio", "disc_diameter_ratio"),
    }
    section, field, unit = fields[node]
    return [
        {
            "name": k + ":" + field,
            "value": m[section][field],
            "unit": unit,
            "missing_reason": "没有符合测量规则的有效结果" if m[section][field] is None else None,
        }
        for k, m in sorted(record["classes"].items())
    ]


def build(base):
    """Pure projection. Public storage accepts only a validated persisted parent."""
    scores = base["research_scoring"]
    if scores["schema_version"] != "case-research-scoring/4.0.0":
        raise ValueError("record_supported_source_version")
    if scores["case_code"] != base["case_evidence"]["case_code"] or (
        scores["input_checksum"] != base["case_evidence"]["input"]["checksum"]
    ):
        raise ValueError("record_source_scope")
    if "evidence_records" in base:
        raise ValueError("record_recursive_source")
    assessments = {
        a["id"]: (i, a) for i, a in enumerate(base["judgment_details"]["ledger"]["assessments"])
    }
    decisions = {a["assessment_id"]: (i, a) for i, a in enumerate(scores["adjudications"])}
    evaluations = {a["assessment_id"]: (i, a) for i, a in enumerate(scores["evaluations"])}
    updates = {
        a["node"]: (i, a)
        for i, a in enumerate(base.get("vascular_followup", {}).get("comparisons", []))
    }
    observations, interpretations = [], []
    for index, r in enumerate(scores["reviews"]):
        di, decision = decisions[r["id"]]
        ei, evaluation = evaluations[r["id"]]
        rp = f"/research_scoring/reviews/{index}"
        dp = f"/research_scoring/adjudications/{di}"
        ep = f"/research_scoring/evaluations/{ei}"
        obs_id = "observation:" + r["id"]
        status = STATUSES[decision["state"]]
        obtained = r["observation_available"] and status != "input_missing"
        ap, a = (None, None)
        if r["id"] in assessments:
            ai, a = assessments[r["id"]]
            ap = f"/judgment_details/ledger/assessments/{ai}"
        kind = (
            "unavailable"
            if not obtained
            else "model_output"
            if r["role"] == "model"
            else "provided_information"
            if r["node"] == "RISK_FIELDS"
            else "visual_review"
            if r["kind"] == "visual" or r["role"] == "visual" or status == "observation_ungradable"
            else "measurement"
            if a and any(isinstance(m.get("value"), (int, float)) for m in a["measurements"])
            else "legacy_observation"
        )
        o = {
            "id": obs_id,
            "assessment_id": r["id"],
            "node": r["node"],
            "title": r["title"],
            "acquisition": "obtained" if obtained else "not_obtained",
            "result_kind": kind,
            "summary": evaluation["observed"],
            "values": _values(a, r, obtained),
            "missing_reason": None if obtained else decision["unresolved"],
            "input_bindings": [ref(base, "/case_evidence/input")],
            "result_bindings": [ref(base, ep)],
            "method_bindings": [ref(base, rp)],
            "context_bindings": [ref(base, ap)] if ap else [],
            "origin": "persisted_fact_projection_no_new_observation",
        }
        if ap:
            o["method_bindings"] = [
                ref(base, ap + "/facets/" + k)
                for k in ["tool", "execution", "preprocessing", "input_match", "output_qc"]
            ]
            o["result_bindings"].append(ref(base, ap + "/measurements"))
        j = {
            "id": "interpretation:" + r["id"],
            "assessment_id": r["id"],
            "node": r["node"],
            "observation_id": obs_id,
            "target": r["target"],
            "status": status,
            "comparison": decision["comparison"],
            "reference_interpretation": decision["reference_interpretation"],
            "unresolved": decision["unresolved"],
            "next_action": decision["next_action"],
            "reference_bindings": [
                ref(base, ap + "/facets/" + k) for k in ["source", "source_quality", "reference"]
            ]
            if ap
            else [],
            "provenance_bindings": [ref(base, dp), ref(base, rp)],
            "observation_reliability": copy.deepcopy(r["observation"]),
            "relation_strength": copy.deepcopy(r["relation"]),
            "applicability": copy.deepcopy(r["applicability"]),
            "judgment": copy.deepcopy(r["judgment"]),
            "correlation_group": r["correlation_group"],
            "dependency_paths": copy.deepcopy(r["dependency_paths"]),
            "dependency_semantics": "current_binding",
            "score_action": "preserve_bound_legacy_review",
            "legacy_score_binding": ref(base, rp),
        }
        for i, reading in enumerate(scores["readings"]):
            if reading["id"] in decision["source_ids"]:
                j["reference_bindings"].append(ref(base, f"/research_scoring/readings/{i}"))
        for i, image in enumerate(scores["image_reviews"]):
            if image["assessment_id"] == r["id"]:
                o["summary"] = image["observed"]
                o["result_bindings"] = [ref(base, f"/research_scoring/image_reviews/{i}")]
        if r["id"] == "visual:rim_continuity":
            o["summary"] = scores["visual_review"]["observed"]
            o["result_bindings"] = [ref(base, "/research_scoring/visual_review")]
            j["reference_bindings"].append(ref(base, "/research_scoring/visual_review"))
        if r["node"] in updates and r["id"].startswith("association:"):
            ui, update = updates[r["node"]]
            u = base["vascular_followup"]
            values = _vascular_values(u["record"], r["node"])
            available = any(x["value"] is not None for x in values)
            o.update(
                acquisition="obtained" if available else "not_obtained",
                result_kind=(
                    "proxy_measurement" if update["state"] == "proxy_measured" else "measurement"
                )
                if available
                else "unavailable",
                summary=update["observed"],
                values=values if available else [],
                missing_reason=None if available else "没有符合测量规则的有效结果",
                input_bindings=[
                    ref(base, "/case_evidence/input"),
                    ref(base, "/vascular_followup/record/input_bindings"),
                ],
                result_bindings=[
                    ref(base, "/vascular_followup/record/classes"),
                    ref(base, "/vascular_followup/artifact"),
                ],
                method_bindings=[
                    ref(base, "/vascular_followup/execution"),
                    ref(base, "/vascular_followup/record/policy"),
                ],
            )
            reason = "新观察尚未绑定新的疾病方向评阅；不复用旧测量的评分"
            j.update(
                status=(
                    "method_mismatch"
                    if update["state"] == "proxy_measured"
                    else "reference_missing"
                )
                if available
                else "input_missing",
                comparison=update["interpretation"],
                reference_interpretation=update["reference"],
                unresolved=update["remaining"],
                next_action=update["remaining"],
                reference_bindings=[
                    ref(base, "/vascular_followup/sources/" + key) for key in update["source_ids"]
                ],
                provenance_bindings=[
                    ref(base, f"/vascular_followup/comparisons/{ui}"),
                    ref(base, "/vascular_followup/policy"),
                ],
                observation_reliability=_missing(reason),
                relation_strength=_missing(reason),
                applicability=_missing(reason),
                judgment={
                    **_missing(reason),
                    "lower": None,
                    "upper": None,
                    "method": "not_assessed",
                },
                dependency_semantics="legacy_not_rebased",
                score_action="new_observation_not_rescored",
            )
        observations.append(o)
        interpretations.append(j)
    observations.sort(key=lambda x: x["id"])
    interpretations.sort(key=lambda x: x["id"])
    obs = {o["id"]: o for o in observations}
    value = {
        "schema_version": "case-observation-interpretation/1.0.0",
        "projection_version": "observation-interpretation-normalization/1.0.0",
        "case_code": scores["case_code"],
        "input_checksum": scores["input_checksum"],
        "base_checksum": base["checksum"],
        "task_scope": "slo_image_research",
        "observations": observations,
        "interpretations": interpretations,
        "node_index": {
            n["key"]: [j["id"] for j in interpretations if j["node"] == n["key"]]
            for n in base["nodes"]
        },
        "inventory": {
            "total": len(obs),
            "observation_obtained": sum(o["acquisition"] == "obtained" for o in observations),
            "observation_not_obtained": sum(
                o["acquisition"] == "not_obtained" for o in observations
            ),
            "direction_assessed": sum(j["status"] == "assessed" for j in interpretations),
            "measurement_without_direction": sum(
                obs[j["observation_id"]]["result_kind"] in {"measurement", "proxy_measurement"}
                and j["status"] != "assessed"
                for j in interpretations
            ),
            "meaning": "record_counts_not_independent_evidence_or_clinical_coverage",
        },
        "legacy_scoring_binding": ref(base, "/research_scoring"),
        "scoring_status": "historical_v4_unchanged_not_recomputed_for_new_measurements",
        "clinical_acceptance": False,
    }
    value["checksum"] = digest(value)
    return EvidenceRecords.model_validate(value).model_dump(mode="json")


def validate(base, value):
    parsed = EvidenceRecords.model_validate(value)
    # Reconstruction rechecks all source pointers and semantics, not just a caller's digest.
    if parsed.model_dump(mode="json") != build(base):
        raise ValueError("record_source_reconstruction_mismatch")
    return parsed
