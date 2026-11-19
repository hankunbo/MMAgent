"""Continuous case judgments, dependence-aware aggregation and exact replay."""

import copy
import math

from packages.evidence_details.models import digest
from packages.evidence_scoring import continuous_policy as policy
from packages.evidence_scoring.continuous_models import ContinuousReview, ContinuousScores, Judgment
from packages.evidence_scoring.engine import (
    _node_ids,
    coverage,
    path_available,
    rating,
)
from packages.evidence_scoring.engine import (
    build as build_legacy,
)

ROLES = {"diagnostic", "risk", "model", "association", "visual"}


def _unassessed(reason):
    return Judgment(
        value=None,
        lower=None,
        upper=None,
        reason=reason,
        missing_reason=reason,
        method="not_assessed",
    )


def _review(raw, assessment, bound):
    row = copy.deepcopy(raw)
    key = row["id"]
    name = key.split(":", 1)[1]
    reason = "当前病例没有绑定的连续判读评阅记录"
    effect = _unassessed(reason)
    if bound:
        reasons = {
            "association": "缺少与本例测量定义一致的个体参照，尚不能确定疾病方向",
            "measurement": "此项是测量或计算背景，本身没有独立疾病判读",
            "visual": "已有视觉记录尚不足以确定本项疾病方向",
            "diagnostic": "缺少所需检查或可明确定位的现象",
            "risk": "缺少本项风险背景输入",
            "model": "缺少匹配的固定模型判读配置",
        }
        effect = _unassessed(reasons[row["role"]])
        chosen = policy.REVIEW_BUNDLE["judgments"].get(key)
        if chosen is not None:
            effect = Judgment(**chosen, method="manual_source_bound_review")
        model = policy.REVIEW_BUNDLE["model"]
        if key == model["id"]:
            measures = {m.get("name"): m.get("value") for m in assessment["measurements"]}
            negative, positive = (measures.get("非青光眼模型分数"), measures.get("青光眼模型分数"))
            if (
                not all(
                    type(v) in {float, int} and math.isfinite(v) and 0 <= v <= 1
                    for v in (negative, positive)
                )
                or negative + positive == 0
            ):
                raise ValueError("continuous_model_class_scores_invalid")
            value = (positive - negative) / (positive + negative)
            effect = Judgment(
                value=value,
                lower=max(-1.0, min(0.0, value * 2)),
                upper=min(1.0, max(0.0, value * 2)),
                reason=model["reason"],
                method="fixed_model_margin",
            )
            for field, number in zip(
                ("observation", "relation", "applicability"), model["ratings"], strict=True
            ):
                row[field] = rating(number, model["reason"]).model_dump(mode="json")
    # A matching review cannot create an absent artifact or a missing source.
    values = [row[f]["value"] for f in ("observation", "relation", "applicability")]
    weight = values[0] * values[1] * values[2] if all(v is not None for v in values) else None
    if effect.value is not None and (weight is None or not row["observation_available"]):
        raise ValueError("continuous_bound_review_missing_evidence")
    family = (
        "cup-rim"
        if name in {"VCDR_GE_07", "RIM_RADIAL_LT_01", "RIM_NOTCH", "rim"}
        else "rnfl"
        if name in {"RNFL_DEFECT", "rnfl"}
        else "disc-hemorrhage"
        if name in {"DISC_HEMORRHAGE", "hemorrhage"}
        else "classifier"
        if row["role"] == "model"
        else name
    )
    block = (
        "demographic:" + name
        if name in {"AGE", "FAMILY", "DEMOGRAPHICS"}
        else "image:" + assessment["same_image_group"]
    )
    e = effect.value
    row.update(
        target="glaucoma_research_hypothesis",
        judgment=effect.model_dump(mode="json"),
        direction="unavailable"
        if e is None
        else "support"
        if e > 0
        else "refute"
        if e < 0
        else "neutral",
        correlation_group=block + ":" + family,
        dependence_block=block,
        weight=weight,
        support=weight * max(e, 0.0) if e is not None else 0.0,
        refutation=weight * max(-e, 0.0) if e is not None else 0.0,
        reason=effect.reason,
    )
    return ContinuousReview.model_validate(row)


def _unique(rows):
    result = {}
    for value in rows:
        # Recheck model_copy/model_construct at the numerical boundary as well.
        from packages.evidence_scoring.continuous_models import TaskContinuousReview

        model = (
            TaskContinuousReview if isinstance(value, TaskContinuousReview) else ContinuousReview
        )
        row = model.model_validate(value.model_dump(mode="json"))
        if result and row.target != next(iter(result.values())).target:
            raise ValueError("continuous_mixed_targets")
        if row.id in result and result[row.id] != row:
            raise ValueError("continuous_conflicting_duplicate")
        result[row.id] = row
    return [result[key] for key in sorted(result)]


def _pool(values):
    ordered = sorted(values, reverse=True)
    if not ordered:
        return 0.0
    top = ordered[0]
    return top + policy.POLICY["shared_input_rho"] * (1 - top) * (
        1 - math.prod(1 - v for v in ordered[1:])
    )


def channel(rows, roles=ROLES, bound="point"):
    """No independence claim: exact duplicates -> families -> shared-input blocks."""
    families = {}
    for row in _unique(rows):
        if row.role not in roles or row.judgment.value is None:
            continue
        values = [row.observation.value, row.relation.value, row.applicability.value]
        lo = math.prod(max(0.0, v - 0.1) for v in values)
        hi = math.prod(min(1.0, v + 0.1) for v in values)
        if bound == "lower":
            support = lo * max(row.judgment.lower, 0.0)
            refute = hi * max(-row.judgment.lower, 0.0)
        elif bound == "upper":
            support = hi * max(row.judgment.upper, 0.0)
            refute = lo * max(-row.judgment.upper, 0.0)
        elif bound == "point":
            support, refute = row.support, row.refutation
        else:
            raise ValueError("continuous_unknown_bound")
        if not support and not refute:
            continue
        group = families.setdefault(row.correlation_group, [row.dependence_block, 0.0, 0.0])
        if group[0] != row.dependence_block:
            raise ValueError("continuous_family_block_conflict")
        group[1], group[2] = max(group[1], support), max(group[2], refute)
    blocks = {}
    for _, (block, support, refute) in sorted(families.items()):
        item = blocks.setdefault(block, [[], []])
        item[0].append(support)
        item[1].append(refute)
    support = 1.0 - math.prod(1 - _pool(g[0]) for _, g in sorted(blocks.items()))
    refute = 1.0 - math.prod(1 - _pool(g[1]) for _, g in sorted(blocks.items()))
    return {
        "support": support,
        "refutation": refute,
        "conflict": min(support, refute),
        "net": support - refute,
        "active_groups": len(families),
    }


def _confidence(rows, combined):
    basis = sorted(
        {r.role for r in rows if r.judgment.value is not None and r.weight > 0 and r.role in ROLES}
    )
    if not basis:
        return {
            "value": None,
            "lower": None,
            "upper": None,
            "status": "unavailable",
            "direction": "unavailable",
            "stability": "unavailable",
            "basis": [],
            "kind": "uncalibrated_research_evidence_score",
            "missing_reason": "没有可用的本例连续判读；不以50分代替缺项",
        }
    value = 0.5 + 0.5 * combined["net"]
    lower = 0.5 + 0.5 * channel(rows, bound="lower")["net"]
    upper = 0.5 + 0.5 * channel(rows, bound="upper")["net"]
    return {
        "value": value,
        "lower": lower,
        "upper": upper,
        "status": "available",
        "direction": "support_side"
        if value > 0.5
        else "refutation_side"
        if value < 0.5
        else "balanced",
        "stability": "support_side"
        if lower > 0.5
        else "refutation_side"
        if upper < 0.5
        else "direction_sensitive",
        "basis": basis,
        "kind": "uncalibrated_research_evidence_score",
        "missing_reason": None,
    }


def summarize(rows, removed_resources=frozenset(), removed_ids=frozenset()):
    rows = _unique(rows)
    active = [
        r
        for r in rows
        if r.id not in removed_ids and path_available(r.dependency_paths, removed_resources)
    ]
    combined = channel(active)
    return {
        "assessment_ids": [r.id for r in rows],
        "diagnostic": channel(active, {"diagnostic", "visual", "association"}),
        "risk": channel(active, {"risk"}),
        "model": channel(active, {"model"}),
        "combined": combined,
        "confidence": _confidence(active, combined),
        "criterion_coverage": coverage(
            sum(r.criterion_checked for r in active), sum(r.kind == "criterion" for r in rows)
        ),
        "observation_coverage": coverage(sum(r.observation_available for r in active), len(rows)),
        "judgment_coverage": coverage(
            sum(r.judgment.value is not None for r in active), sum(r.role in ROLES for r in rows)
        ),
        "contributing_ids": sorted(r.id for r in active if r.support or r.refutation),
    }


def build(base):
    legacy = build_legacy(base)
    bundle = policy.REVIEW_BUNDLE
    bound = legacy["case_code"] == bundle["case_code"]
    if bound and any(legacy[key] != bundle[key] for key in ("input_checksum", "trace_checksum")):
        raise ValueError("continuous_review_scope_or_source_changed")
    assessments = {a["id"]: a for a in base["judgment_details"]["ledger"]["assessments"]}
    rows = [_review(r, assessments[r["id"]], bound) for r in legacy["reviews"]]
    root, ids = _node_ids(base, rows)
    subsets = {k: [r for r in rows if r.id in subset] for k, subset in ids.items()}
    nodes = {k: summarize(v) for k, v in subsets.items()}
    removals = []
    for original in legacy["removals"]:
        resources, removed = (
            set(original["unavailable_resources"]),
            set(original["removed_assessments"]),
        )
        after_nodes = {k: summarize(v, resources, removed) for k, v in subsets.items()}
        before, after = nodes[root], after_nodes[root]
        old_score, new_score = before["confidence"]["value"], after["confidence"]["value"]
        row = copy.deepcopy(original)
        row.update(
            after=after,
            changed_nodes=sorted(k for k in nodes if nodes[k] != after_nodes[k]),
            diagnostic_support_delta=after["diagnostic"]["support"]
            - before["diagnostic"]["support"],
            risk_support_delta=after["risk"]["support"] - before["risk"]["support"],
            confidence_delta=new_score - old_score
            if new_score is not None and old_score is not None
            else None,
            explanation="固定其余评阅，移除判读或其必需资源后重算总分；分母不变，非生物学因果干预。",
        )
        removals.append(row)
    raw = {
        k: legacy[k]
        for k in (
            "case_code",
            "input_checksum",
            "base_checksum",
            "trace_checksum",
            "reviewer",
            "reviewed_at",
            "purpose",
            "disease_probability",
            "clinical_acceptance",
            "resources",
        )
    }
    raw.update(
        schema_version="case-research-scoring/2.0.0",
        policy_version=policy.POLICY["version"],
        policy_checksum=digest(policy.POLICY),
        review_bundle_checksum=digest(bundle),
        review_status="bound_case_review" if bound else "not_reviewed",
        reviews=[r.model_dump(mode="json") for r in rows],
        nodes=nodes,
        removals=removals,
        method_notes=policy.NOTES,
    )
    return ContinuousScores.model_validate({**raw, "checksum": digest(raw)}).model_dump(mode="json")


def validate(base, scores):
    parsed = ContinuousScores.model_validate(copy.deepcopy(scores)).model_dump(mode="json")
    if parsed != build(base):
        raise ValueError("continuous_score_reconstruction_mismatch")
    return parsed
