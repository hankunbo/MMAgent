"""All-node participation; observed direction and missing coverage stay separate."""

import copy
import json

from packages.evidence_details.models import digest
from packages.evidence_scoring import continuous
from packages.evidence_scoring import participation_policy as policy
from packages.evidence_scoring.continuous_models import ContinuousReview
from packages.evidence_scoring.engine import _node_ids, coverage, path_available, rating
from packages.evidence_scoring.participation_models import ParticipationScores

SCOPES = ("comprehensive_research", "slo_image_research")


def _rows(base, previous, bound):
    rows = copy.deepcopy(previous["reviews"])
    if not bound:
        return [ContinuousReview.model_validate(r) for r in rows]
    review = policy.VISUAL_REVIEW
    if review["input_checksum"] != previous["input_checksum"] or not any(
        a["artifact_type_code"] == "ARTIFACT:SLO_SEGMAN_OVERLAY"
        and a["checksum"] == review["overlay_checksum"]
        for a in base["case_evidence"].get("artifacts", [])
    ):
        raise ValueError("participation_visual_input_mismatch")
    notch = next(r for r in rows if r["id"] == "criterion:RIM_NOTCH")
    notch.update(
        judgment=continuous._unassessed(
            "细微局灶切迹受血管、条带和边界影响，尚不能确认；粗大连续性另项记录。"
        ).model_dump(mode="json"),
        direction="unavailable",
        support=0.0,
        refutation=0.0,
        reason="已检查不等于已确认不存在细微切迹；本项不以中性0伪装可判读。",
    )
    extra = copy.deepcopy(notch)
    values = review["ratings"]
    weight = values[0] * values[1] * values[2]
    effect = review["effect"]
    extra.update(
        id="visual:rim_continuity",
        title="可见范围的粗大盘沿连续性",
        kind="visual",
        role="visual",
        assessment_checksum=digest(
            {"parent": notch["assessment_checksum"], "visual_review": review}
        ),
        source_binding_checksum=digest(
            {"parent": notch["source_binding_checksum"], "visual_review": review}
        ),
        observation=rating(values[0], review["reason"]).model_dump(mode="json"),
        relation=rating(values[1], review["reason"]).model_dump(mode="json"),
        applicability=rating(values[2], review["reason"]).model_dump(mode="json"),
        judgment={
            "value": effect,
            "lower": review["lower"],
            "upper": review["upper"],
            "reason": review["reason"],
            "missing_reason": None,
            "method": "manual_source_bound_review",
        },
        direction="refute" if effect < 0 else "support" if effect > 0 else "neutral",
        observation_available=True,
        criterion_checked=False,
        reason=review["reason"],
        dependency_paths=[["image", "segman_disc", "segman_cup"]],
        weight=weight,
        support=weight * max(effect, 0.0),
        refutation=weight * max(-effect, 0.0),
    )
    rows.append(extra)
    return sorted((ContinuousReview.model_validate(r) for r in rows), key=lambda r: r.id)


def groups_for(rows, definitions, removed_resources=frozenset(), removed_ids=frozenset()):
    rows = {r.id: r for r in continuous._unique(rows)}
    groups = []
    for d in sorted(definitions, key=lambda g: g["id"]):
        slots = []
        for ids in d["slots"]:
            if not set(ids) <= set(rows):
                raise ValueError("participation_missing_inventory_slot")
            acquired = any(
                i in policy.ACQUIRED
                and i not in removed_ids
                and rows[i].observation_available
                and path_available(rows[i].dependency_paths, removed_resources)
                for i in ids
            )
            slots.append(
                {
                    "assessment_ids": ids,
                    "acquired": acquired,
                    "reason": "已取得此槽要求的同定义观察；不表示正常或临床通过。"
                    if acquired
                    else d["next_action"],
                }
            )
        groups.append(
            {
                **{
                    k: d[k] for k in ("id", "title", "nodes", "importance", "reason", "next_action")
                },
                "slots": slots,
                "coverage": coverage(sum(s["acquired"] for s in slots), len(slots)),
                "definition_checksum": digest(d),
                "source_assessment_checksums": {
                    i: rows[i].assessment_checksum for s in slots for i in s["assessment_ids"]
                },
            }
        )
    return groups


def adequacy(groups, confidence, scope):
    selected = sorted((g for g in groups if g["importance"][scope] > 0), key=lambda g: g["id"])
    if len({g["id"] for g in selected}) != len(selected):
        raise ValueError("participation_duplicate_group")
    denominator = sum(g["importance"][scope] for g in selected)
    numerator = sum(g["importance"][scope] * g["coverage"]["value"] for g in selected)
    q = numerator / denominator if denominator else None

    def adjusted(field):
        c = confidence[field]
        return 0.5 + q * (c - 0.5) if q is not None and c is not None else None

    return {
        "scope": scope,
        "group_ids": [g["id"] for g in selected],
        "weighted_acquired": float(numerator),
        "total_importance": float(denominator),
        "value": q,
        "missing": 1 - q if q is not None else None,
        "adjusted": adjusted("value"),
        "adjusted_lower": adjusted("lower"),
        "adjusted_upper": adjusted("upper"),
        "status": "available" if denominator else "not_applicable",
    }


def _summary(rows, groups, removed_resources=frozenset(), removed_ids=frozenset()):
    result = continuous.summarize(rows, removed_resources, removed_ids)
    result["adequacy"] = {scope: adequacy(groups, result["confidence"], scope) for scope in SCOPES}
    result["missing_impacts"] = []
    for scope, a in result["adequacy"].items():
        if a["value"] is None:
            continue
        for g in groups:
            if g["importance"][scope] == 0 or g["coverage"]["value"] == 1:
                continue
            missing = g["importance"][scope] * (1 - g["coverage"]["value"]) / a["total_importance"]
            q_after = min(1.0, a["value"] + missing)
            c = result["confidence"]["value"]
            adjusted = 0.5 + q_after * (c - 0.5) if c is not None else None
            result["missing_impacts"].append(
                {
                    "group_id": g["id"],
                    "scope": scope,
                    "missing_contribution": missing,
                    "coverage_after_completion": q_after,
                    "adjusted_after_completion": adjusted,
                    "adjusted_delta": adjusted - a["adjusted"] if adjusted is not None else None,
                    "scenario_kind": "coverage_only_holds_direction_fixed",
                    "disease_direction_impact": "not_quantified",
                    "reason": "仅假设该组输入取得，保持既有疾病方向D不变；未预测检查结果。"
                    "缺少匹配效应范围，阳性/阴性对D的影响尚不能量化。" + g["next_action"],
                }
            )
    return result


def _selected_groups(key, subset, groups, roles):
    serves = set(roles.get(key, {}).get("serves", []))
    return [
        g
        for g in groups
        if key == "ROOT"
        or key in g["nodes"]
        or serves.intersection(g["nodes"])
        or any(set(s["assessment_ids"]).intersection(subset) for s in g["slots"])
    ]


def _evaluations(base, rows, groups, removals, root, bound):
    assessments = {a["id"]: a for a in base["judgment_details"]["ledger"]["assessments"]}
    after = {r["id"]: r["after"] for r in removals}
    result = []
    for r in rows:
        selected = [
            g
            for g in groups
            if r.node in g["nodes"] or any(r.id in s["assessment_ids"] for s in g["slots"])
        ]
        a = assessments.get(r.id)
        direct = r.id == "visual:rim_continuity"
        if direct:
            observed = policy.VISUAL_REVIEW["observed"]
            unresolved = policy.VISUAL_REVIEW["unresolved"]
        else:
            observed = (
                a["comparison"].get("assessment")
                or a["comparison"].get("finding")
                or (
                    json.dumps(a["measurements"], ensure_ascii=False)[:1500]
                    if a["measurements"]
                    else "当前没有该项病例观察。"
                )
            )
            unresolved = (
                r.judgment.missing_reason
                or "方向已评阅；工作范围用于参数敏感性，不是统计置信区间。"
            )
        quality = policy.QUALITY.get(r.id) if bound else None
        target = (
            "disease_direction"
            if r.judgment.value is not None
            else (
                "evidence_quality" if quality or r.role == "measurement" else "evidence_sufficiency"
            )
        )
        removed = after["review:" + r.id]
        delta = (
            removed["confidence"]["value"] - root["confidence"]["value"]
            if removed["confidence"]["value"] is not None
            and root["confidence"]["value"] is not None
            else None
        )
        result.append(
            {
                "assessment_id": r.id,
                "target": target,
                "observed": observed,
                "unresolved": unresolved,
                "group_ids": [g["id"] for g in selected],
                "next_action": "；".join(dict.fromkeys(g["next_action"] for g in selected))
                or "作为技术/描述记录保留；需匹配医学参照后才增加疾病方向。",
                "quality_review": rating(*quality).model_dump(mode="json") if quality else None,
                "effective_support": max(
                    0.0, root["combined"]["support"] - removed["combined"]["support"]
                ),
                "effective_refutation": max(
                    0.0, root["combined"]["refutation"] - removed["combined"]["refutation"]
                ),
                "marginal_direction_delta": delta,
                "assessment_checksum": r.assessment_checksum,
                "provenance": "assistant_direct_image_inspection"
                if direct
                else "source_bound_existing_record_review",
            }
        )
    return result


def build(base):
    previous = continuous.build(base)
    bound = previous["case_code"] == policy.REVIEW_SCOPE["case_code"]
    if bound and any(
        previous[k] != policy.REVIEW_SCOPE[k] for k in ("input_checksum", "trace_checksum")
    ):
        raise ValueError("participation_review_scope_or_source_changed")
    rows = _rows(base, previous, bound)
    root, subsets = _node_ids(base, rows)
    roles = base.get("evidence_structure", {}).get("node_roles", {})
    definitions = policy.GROUPS if bound else []
    groups = groups_for(rows, definitions)

    def summaries(removed_resources=frozenset(), removed_ids=frozenset()):
        gs = groups_for(rows, definitions, removed_resources, removed_ids)
        return {
            key: _summary(
                [r for r in rows if r.id in subset],
                _selected_groups(key, subset, gs, roles),
                removed_resources,
                removed_ids,
            )
            for key, subset in subsets.items()
        }

    nodes = summaries()
    scenarios = [
        (r["id"], r["title"], r["unavailable_resources"], r["removed_assessments"])
        for r in previous["removals"]
    ]
    if bound:
        scenarios.append(
            (
                "review:visual:rim_continuity",
                "移除「可见范围的粗大盘沿连续性」",
                [],
                ["visual:rim_continuity"],
            )
        )
    removals = []
    before = nodes[root]
    for key, title, resources, removed in scenarios:
        after_nodes = summaries(set(resources), set(removed))
        after = after_nodes[root]
        old_c, new_c = before["confidence"]["value"], after["confidence"]["value"]

        def delta(scope, field, after=after):
            a, b = before["adequacy"][scope][field], after["adequacy"][scope][field]
            return b - a if a is not None and b is not None else None

        removals.append(
            {
                "id": key,
                "title": title,
                "unavailable_resources": resources,
                "removed_assessments": removed,
                "affected_assessments": [
                    r.id
                    for r in rows
                    if r.id in removed
                    or any(set(p).intersection(resources) for p in r.dependency_paths)
                    and not path_available(r.dependency_paths, set(resources))
                ],
                "after": after,
                "changed_nodes": sorted(k for k in nodes if nodes[k] != after_nodes[k]),
                "diagnostic_support_delta": after["diagnostic"]["support"]
                - before["diagnostic"]["support"],
                "risk_support_delta": after["risk"]["support"] - before["risk"]["support"],
                "confidence_delta": new_c - old_c
                if new_c is not None and old_c is not None
                else None,
                "criterion_coverage_delta": after["criterion_coverage"]["value"]
                - before["criterion_coverage"]["value"]
                if before["criterion_coverage"]["value"] is not None
                else None,
                "adequacy_delta": {s: delta(s, "value") for s in SCOPES},
                "adjusted_delta": {s: delta(s, "adjusted") for s in SCOPES},
                "explanation": "移除判读或必需资源；独立证据保留、分母不变。"
                "D、Q和调整分分别重算；不是患者生物学反事实。",
            }
        )
    evaluations = _evaluations(base, rows, groups, removals, before, bound)
    participation = {}
    for node in base["nodes"]:
        key = node["key"]
        own = [
            e
            for e in evaluations
            if any(r.id == e["assessment_id"] and r.node == key for r in rows)
        ]
        resource_ids = policy.TECH_RESOURCES.get(key, [])
        available = sum(bool(previous["resources"].get(r)) for r in resource_ids)
        role = (
            "organization"
            if not own
            else "mixed"
            if len({e["target"] for e in own}) > 1
            else {
                "disease_direction": "disease_evidence",
                "evidence_quality": "evidence_quality",
                "evidence_sufficiency": "evidence_sufficiency",
            }[own[0]["target"]]
        )
        if resource_ids:
            role = "evidence_quality"
        participation[key] = {
            "role": role,
            "own_assessment_ids": [e["assessment_id"] for e in own],
            "group_ids": [g["id"] for g in _selected_groups(key, subsets[key], groups, roles)],
            "serves": roles.get(key, {}).get("serves", []),
            "required_resources": resource_ids,
            "technical_coverage": coverage(available, len(resource_ids)),
            "quality_reviews": [e["quality_review"] for e in own if e["quality_review"]],
            "reason": "目录仅汇总唯一后代；技术资源到位率不等于模型准确率或医学支持。"
            "相关观察的质量已进入O/A，不再重复乘权重。",
        }
    raw = {
        k: previous[k]
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
        schema_version="case-research-scoring/3.0.0",
        policy_version=policy.POLICY["version"],
        policy_checksum=digest(
            {
                "policy": policy.POLICY,
                "groups": definitions,
                "acquired": sorted(policy.ACQUIRED),
                "quality": policy.QUALITY,
                "technical": policy.TECH_RESOURCES,
            }
        ),
        direction_policy_checksum=previous["policy_checksum"],
        review_bundle_checksum=digest(
            {
                "scope": policy.REVIEW_SCOPE,
                "visual": policy.VISUAL_REVIEW,
                "previous": previous["review_bundle_checksum"],
            }
        ),
        review_status="bound_case_review" if bound else "not_reviewed",
        default_scope="comprehensive_research",
        visual_review=policy.VISUAL_REVIEW if bound else None,
        reviews=[r.model_dump(mode="json") for r in rows],
        groups=groups,
        evaluations=evaluations,
        node_participation=participation,
        nodes=nodes,
        removals=removals,
        method_notes=policy.NOTES,
    )
    return ParticipationScores.model_validate({**raw, "checksum": digest(raw)}).model_dump(
        mode="json"
    )


def validate(base, scores):
    parsed = ParticipationScores.model_validate(copy.deepcopy(scores)).model_dump(mode="json")
    if parsed != build(base):
        raise ValueError("participation_reconstruction_mismatch")
    return parsed
