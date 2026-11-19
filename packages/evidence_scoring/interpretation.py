"""Case-bound adjudication; reuse v2 direction and v3 adequacy mathematics."""

import copy
import json
from pathlib import Path

from packages.evidence_details.models import digest
from packages.evidence_scoring import continuous, participation
from packages.evidence_scoring import interpretation_policy as policy
from packages.evidence_scoring import participation_policy as previous_policy
from packages.evidence_scoring.continuous_models import ContinuousReview
from packages.evidence_scoring.engine import _node_ids, path_available, rating
from packages.evidence_scoring.interpretation_models import InterpretationScores

READINGS = json.loads(Path(__file__).with_name("interpretation_sources.json").read_text())
SOURCE_KEYS = {
    "association:ARTERIOLAR_CALIBER": ["caliber", "null_geometry"],
    "association:AV_TORTUOSITY": ["geometry", "null_geometry"],
    "association:VENULAR_BRANCH_ANGLE": ["geometry"],
    "association:VASCULAR_SKELETON_DENSITY": ["density", "density_methods"],
    "association:VASCULAR_FRACTAL": ["density", "density_methods", "null_geometry"],
    "association:DISC_SHAPE_CONTEXT": ["tilt_lee", "tilt_sawada"],
    "association:DISC_VESSEL_COURSE": ["aao"],
    "association:PERIPAPILLARY_APPEARANCE": ["aao"],
    "association:INTEREYE_COMPARISON": ["aao"],
    "association:STRUCTURE_FUNCTION_MATCH": ["aao"],
    "criterion:RNFL_DEFECT": ["aao"],
    "criterion:RIM_NOTCH": ["aao"],
    "criterion:DISC_HEMORRHAGE": ["aao"],
}


def _rows(previous, bound):
    rows = copy.deepcopy(previous["reviews"])
    images = []
    if bound:
        missing = {r["id"] for r in rows if r["judgment"]["value"] is None}
        if missing != set(policy.DECISIONS):
            raise ValueError("interpretation_unreviewed_inventory_change")
        for r in rows:
            if r["id"] in policy.DECISIONS:
                _, _, reference, gap, _ = policy.DECISIONS[r["id"]]
                r["judgment"] = continuous._unassessed(gap).model_dump(mode="json")
                r["reason"] = reference + " " + gap
        for v in policy.VISUAL:
            parent = next(r for r in rows if r["id"] == v["parent"])
            row = copy.deepcopy(parent)
            provenance = {
                "assessment_id": v["id"],
                "parent_assessment_id": v["parent"],
                "observer": "assistant_direct_image_inspection_not_deployed_vlm",
                "input_checksum": previous["input_checksum"],
                "region_xyxy": v["region"],
                "image_size_wh": [512, 664],
                "observed": v["observed"],
                "unresolved": v["unresolved"],
                "source_ids": ["aao"],
            }
            reason = (
                v["reference"]
                + " "
                + v["unresolved"]
                + " 数值为助手研究工作赋值，范围允许相反方向，不是测得敏感度。"
            )
            weight = v["ratings"][0] * v["ratings"][1] * v["ratings"][2]
            row.update(
                id=v["id"],
                title=v["title"],
                kind="visual",
                role="visual",
                assessment_checksum=digest(
                    {"parent": parent["assessment_checksum"], "visual": provenance}
                ),
                source_binding_checksum=digest(
                    {"parent": parent["source_binding_checksum"], "readings": READINGS, "review": v}
                ),
                judgment={
                    "value": v["effect"],
                    "lower": v["lower"],
                    "upper": v["upper"],
                    "reason": reason,
                    "missing_reason": None,
                    "method": "manual_source_bound_review",
                },
                observation_available=True,
                criterion_checked=False,
                reason=reason,
                direction="refute",
                dependency_paths=[["image"]],
                correlation_group=parent["dependence_block"] + ":" + v["family"],
                weight=weight,
                support=0.0,
                refutation=weight * -v["effect"],
            )
            for field, value in zip(
                ("observation", "relation", "applicability"), v["ratings"], strict=True
            ):
                row[field] = rating(value, reason).model_dump(mode="json")
            rows.append(row)
            images.append(provenance)
    return continuous._unique([ContinuousReview.model_validate(r) for r in rows]), images


def _decisions(rows, previous, bound):
    old = {a["assessment_id"]: a for a in previous["evaluations"]}
    visual = {v["id"]: v for v in policy.VISUAL}
    result = []
    for r in rows:
        if not bound:
            state, comparison, reference, gap, action = (
                "not_reviewed",
                "本病例尚无绑定的逐项评阅。",
                "不借用另一病例的工作判断。",
                "需本病例的来源与观察。",
                "为本病例建立独立评阅。",
            )
        elif r.id in policy.DECISIONS:
            state, comparison, reference, gap, action = policy.DECISIONS[r.id]
        elif r.id in visual:
            v = visual[r.id]
            state, comparison, reference, gap, action = (
                "direction_reviewed",
                v["observed"],
                v["reference"],
                v["unresolved"],
                "已计入弱反向研究评阅；不替代父项完整检查，也不增加原检查槽覆盖率。",
            )
        else:
            e = old[r.id]
            state, comparison, reference, gap, action = (
                "direction_reviewed",
                e["observed"],
                r.reason,
                e["unresolved"],
                e["next_action"],
            )
        result.append(
            {
                "assessment_id": r.id,
                "state": state,
                "comparison": comparison,
                "reference_interpretation": reference,
                "unresolved": gap,
                "next_action": action,
                "source_ids": (["aao"] if r.id in visual else SOURCE_KEYS.get(r.id, []))
                if bound
                else [],
                "assessment_checksum": r.assessment_checksum,
                "source_binding_checksum": r.source_binding_checksum,
                "review_completed": bound,
            }
        )
    return result


def build(base):
    previous = participation.build(base)
    bound = previous["review_status"] == "bound_case_review"
    rows, images = _rows(previous, bound)
    root, subsets = _node_ids(base, rows)
    roles = base.get("evidence_structure", {}).get("node_roles", {})
    definitions = previous_policy.GROUPS if bound else []
    groups = participation.groups_for(rows, definitions)

    def summaries(resources=frozenset(), ids=frozenset()):
        gs = participation.groups_for(rows, definitions, resources, ids)
        return {
            key: participation._summary(
                [r for r in rows if r.id in subset],
                participation._selected_groups(key, subset, gs, roles),
                resources,
                ids,
            )
            for key, subset in subsets.items()
        }

    nodes = summaries()
    before = nodes[root]
    scenarios = [
        (r["id"], r["title"], r["unavailable_resources"], r["removed_assessments"])
        for r in previous["removals"]
    ]
    scenarios += (
        [("review:" + v["id"], "移除「" + v["title"] + "」", [], [v["id"]]) for v in policy.VISUAL]
        if bound
        else []
    )
    removals = []
    for key, title, resources, ids in scenarios:
        after_nodes = summaries(set(resources), set(ids))
        after = after_nodes[root]

        def delta(section, field, after=after):
            a, b = before[section][field], after[section][field]
            return b - a if a is not None and b is not None else None

        def adequacy_delta(field, after=after):
            return {
                s: after["adequacy"][s][field] - before["adequacy"][s][field]
                if after["adequacy"][s][field] is not None
                and before["adequacy"][s][field] is not None
                else None
                for s in participation.SCOPES
            }

        removals.append(
            {
                "id": key,
                "title": title,
                "unavailable_resources": resources,
                "removed_assessments": ids,
                "affected_assessments": [
                    r.id
                    for r in rows
                    if r.id in ids
                    or any(set(p).intersection(resources) for p in r.dependency_paths)
                    and not path_available(r.dependency_paths, set(resources))
                ],
                "after": after,
                "changed_nodes": sorted(k for k in nodes if nodes[k] != after_nodes[k]),
                "diagnostic_support_delta": delta("diagnostic", "support"),
                "risk_support_delta": delta("risk", "support"),
                "confidence_delta": delta("confidence", "value"),
                "criterion_coverage_delta": delta("criterion_coverage", "value"),
                "adequacy_delta": adequacy_delta("value"),
                "adjusted_delta": adequacy_delta("adjusted"),
                "explanation": "沿用v2方向与v3充分性公式重算；固定分母和同源去重；"
                "不是患者生物学反事实。",
            }
        )
    decisions = _decisions(rows, previous, bound)
    decisions_by_id = {a["assessment_id"]: a for a in decisions}
    removals_by_id = {a["id"]: a for a in removals}
    old_e = {e["assessment_id"]: e for e in previous["evaluations"]}
    evaluations = []
    for r in rows:
        a = decisions_by_id[r.id]
        removed = removals_by_id["review:" + r.id]
        selected = [
            g["id"]
            for g in groups
            if r.node in g["nodes"] or any(r.id in s["assessment_ids"] for s in g["slots"])
        ]
        evaluations.append(
            {
                "assessment_id": r.id,
                "target": old_e[r.id]["target"] if r.id in old_e else "disease_direction",
                "observed": a["comparison"],
                "unresolved": a["unresolved"],
                "next_action": a["next_action"],
                "group_ids": selected,
                "quality_review": old_e.get(r.id, {}).get("quality_review"),
                "effective_support": max(
                    0.0, before["combined"]["support"] - removed["after"]["combined"]["support"]
                ),
                "effective_refutation": max(
                    0.0,
                    before["combined"]["refutation"] - removed["after"]["combined"]["refutation"],
                ),
                "marginal_direction_delta": removed["confidence_delta"],
                "assessment_checksum": r.assessment_checksum,
                "provenance": "assistant_direct_image_inspection"
                if r.id in {v["assessment_id"] for v in images}
                else old_e[r.id]["provenance"],
            }
        )
    node_participation = copy.deepcopy(previous["node_participation"])
    for key, entry in node_participation.items():
        own = [
            e
            for e in evaluations
            if next(r for r in rows if r.id == e["assessment_id"]).node == key
        ]
        entry["own_assessment_ids"] = [e["assessment_id"] for e in own]
        if len({e["target"] for e in own}) > 1 and not entry["required_resources"]:
            entry["role"] = "mixed"
    raw = copy.deepcopy(previous)
    raw.update(
        schema_version="case-research-scoring/4.0.0",
        policy_version=policy.VERSION,
        policy_checksum=digest(
            {
                "previous": previous["policy_checksum"],
                "version": policy.VERSION,
                "decisions": policy.DECISIONS,
                "visual": policy.VISUAL,
                "sources": READINGS,
            }
        ),
        review_bundle_checksum=digest(
            {
                "previous": previous["review_bundle_checksum"],
                "adjudications": decisions,
                "images": images,
            }
        ),
        reviews=[r.model_dump(mode="json") for r in rows],
        nodes=nodes,
        groups=groups,
        evaluations=evaluations,
        removals=removals,
        node_participation=node_participation,
        readings=READINGS if bound else [],
        adjudications=decisions,
        image_reviews=images,
        method_notes=previous["method_notes"]
        + [
            "v4逐项核对观察、来源终点与方法。核对完成不代表缺失检查已经取得。",
            "新增两项仅是可辨局部的粗大视觉观察；O/L/A和e为助手模拟LLM的研究工作值，非经验验证的准确率。",
            "较低迂曲度或较窄分叉角的群体关联没有自动给出本例的参考分布；方法不匹配不强行赋疾病方向。",
            "新增局部观察不替代细微/全范围RNFL及盘内血管检查，v3充分性槽和分母保持不变。",
        ],
    )
    raw.pop("checksum")
    return InterpretationScores.model_validate({**raw, "checksum": digest(raw)}).model_dump(
        mode="json"
    )


def validate(base, scores):
    parsed = InterpretationScores.model_validate(copy.deepcopy(scores)).model_dump(mode="json")
    if parsed != build(base):
        raise ValueError("interpretation_reconstruction_mismatch")
    return parsed
