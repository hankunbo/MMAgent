"""Deterministic QBAF-inspired two-channel propagation, without a disease prior."""

import copy
import math

from packages.evidence_details.models import digest
from packages.evidence_scoring.models import Rating, Review, Scores
from packages.evidence_scoring.policy import NOTES, POLICY


def rating(value, reason, missing="所需观察或可比判读依据未提供"):
    return Rating(value=value, reason=reason, missing_reason=missing if value is None else None)


def path_available(paths, removed):
    return any(path and not set(path).intersection(removed) for path in paths)


def _dependencies(a):
    name = a["id"].split(":", 1)[1]
    if name in {"AGE", "FAMILY", "DEMOGRAPHICS"}:
        return [["demographics"]]
    if name in {"VCDR_GE_07", "RIM_RADIAL_LT_01", "CUP_AREA_CONTEXT", "CUP_PIXELS"}:
        return [["image", "segman_disc", "segman_cup"]]
    if name == "DISC_SHAPE_CONTEXT":
        return [["image", "segman_disc"]]
    if name == "VENULAR_BRANCH_ANGLE":
        return [["image", "segman_disc", "slo_vein", "branch_geometry"]]
    if name == "CLASSIFICATION":
        return [["image", "fairclip"]]
    if name.startswith("VESSEL_") or name in {
        "ARTERIOLAR_CALIBER",
        "AV_TORTUOSITY",
        "VASCULAR_SKELETON_DENSITY",
        "VASCULAR_FRACTAL",
    }:
        return [["image", "slo_vessel"]]
    if name == "AV_AREAS":
        return [["image", "slo_artery", "slo_vein"]]
    if name in {"FOVEA", "fovea"}:
        return [["image", "fovea"]] if a["kind"] == "descriptive" else [["image"]]
    if a["comparison"].get("comparison") == "unavailable":
        return []
    return [["image"]]


def _available_resources(base):
    types = {a["artifact_type_code"] for a in base["case_evidence"].get("artifacts", [])}
    available = {"image", "demographics"}
    for name, kind in {
        "segman_disc": "ARTIFACT:SLO_SEGMAN_DISC",
        "segman_cup": "ARTIFACT:SLO_SEGMAN_CUP",
        "slo_vessel": "ARTIFACT:SLO_VESSEL_MASK",
        "slo_vein": "ARTIFACT:SLO_VEIN_MASK",
        "slo_artery": "ARTIFACT:SLO_ARTERY_MASK",
        "fovea": "ARTIFACT:SLO_FOVEA_MAP",
        "branch_geometry": "ARTIFACT:SLO_VENULAR_BRANCH_GEOMETRY",
    }.items():
        if kind in types:
            available.add(name)
    if base["case_evidence"].get("fairclip", {}).get("outputs"):
        available.add("fairclip")
    return available


def _resource_bindings(base):
    """Precise input records for the human-readable resource dependency names."""
    from packages.evidence_details.models import resolve

    pointers = {
        "image": ["/case_evidence/input"],
        "demographics": [
            "/judgment_details/ledger/assessments/" + str(i)
            for i, a in enumerate(base["judgment_details"]["ledger"]["assessments"])
            if a["id"] in {"criterion:AGE", "criterion:FAMILY", "descriptive:DEMOGRAPHICS"}
        ],
        "fairclip": ["/case_evidence/fairclip/outputs"],
    }
    kinds = {
        "segman_disc": "ARTIFACT:SLO_SEGMAN_DISC",
        "segman_cup": "ARTIFACT:SLO_SEGMAN_CUP",
        "slo_vessel": "ARTIFACT:SLO_VESSEL_MASK",
        "slo_vein": "ARTIFACT:SLO_VEIN_MASK",
        "slo_artery": "ARTIFACT:SLO_ARTERY_MASK",
        "fovea": "ARTIFACT:SLO_FOVEA_MAP",
        "branch_geometry": "ARTIFACT:SLO_VENULAR_BRANCH_GEOMETRY",
    }
    for name, kind in kinds.items():
        pointers[name] = [
            "/case_evidence/artifacts/" + str(i)
            for i, a in enumerate(base["case_evidence"].get("artifacts", []))
            if a["artifact_type_code"] == kind
        ]
    return {
        key: [
            {
                "pointer": p,
                "checksum": digest(resolve(base, p)),
                "purpose": "判读的必需输入绑定；不是新增医学支持边",
            }
            for p in paths
        ]
        for key, paths in pointers.items()
    }


def _review(a, resources):
    kind, name, comparison = a["kind"], a["id"].split(":", 1)[1], a["comparison"]
    state = comparison.get("comparison")
    source = a["facets"]["source"]
    source_known = bool(source["facts"]) and source["status"] in {"bound", "partial"}
    reason = "已保存技术测量，工作评分评价其可追溯性；没有匹配的单例医学阈值，不生成疾病方向。"
    role, relation, applicability = "measurement", None, None
    observed = state in {"met", "not_met", "observed", "visually_reviewed"}
    observation = 0.70 if observed else None
    checked, direction = False, "neutral" if observed else "unavailable"
    if kind == "criterion":
        if name not in POLICY["criterion_ratings"]:
            raise ValueError("research_score_unreviewed_criterion")
        observation, relation, applicability = POLICY["criterion_ratings"][name]
        reason = POLICY["criterion_reasons"][name]
        role = "risk" if name in {"AGE", "FAMILY", "IOP"} else "diagnostic"
        observed = state in {"met", "not_met", "not_observed"}
        if not observed:
            observation, applicability = None, None
        checked = state in {"met", "not_met"}
        direction = "support" if state == "met" else "neutral" if observed else "unavailable"
    elif kind == "association":
        role, relation = "association", 0.25 if name == "DISC_SHAPE_CONTEXT" else 0.35
        observed = any(m.get("value") is not None for m in a["measurements"] if isinstance(m, dict))
        observation = 0.65 if observed else None
        direction = "neutral" if observed else "unavailable"
        reason = (
            "已有文献关联，关系强度暂取探索级工作值；"
            "群体相关性不提供当前测量方法的单例效应或阈值，适用性留空。"
        )
    elif kind == "visual":
        role = "visual"
        observed = comparison.get("finding") in {"appearance_observed", "visible_artifacts"}
        observation = 0.55 if observed else None
        direction = "neutral" if observed else "unavailable"
        reason = (
            "读取历史助手看图记录，本次未运行VLM；"
            "可见伪影或外观可记录，未定位到的现象不按阴性计分。"
        )
    elif name == "CLASSIFICATION":
        role = "model"
        observation = 0.75 if observed else None
        reason = (
            "固定FairCLIP执行和原始分数已保存；本分只评输出来源，不是模型准确率。"
            "原softmax单列，不换算为医学支持或患病概率。"
        )
    elif name == "FOVEA":
        observation = None
        reason = "模型输出已保存，但点靠近边界且视觉复核未确认定位；该点不用于方向判定。"
        observed, direction = False, "unavailable"
    if not source_known:
        relation = None
    paths = _dependencies(a)
    missing_resources = {r for path in paths for r in path} - resources
    if not path_available(paths, missing_resources):
        observed, checked, direction = False, False, "unavailable"
        observation, applicability = None, None
    weight = (
        observation * relation * applicability
        if all(v is not None for v in (observation, relation, applicability))
        else None
    )
    if direction == "support" and weight is None:
        direction = "unavailable"
    group = (
        "demographic-risk" if name in {"AGE", "FAMILY"} else "same-image:" + a["same_image_group"]
    )
    return Review(
        id=a["id"],
        node=a["node"],
        title=a["title"],
        kind=kind,
        assessment_checksum=digest(a),
        source_binding_checksum=digest(source),
        observation=rating(observation, reason),
        relation=rating(relation, reason),
        applicability=rating(applicability, reason),
        role=role,
        direction=direction,
        observation_available=observed,
        criterion_checked=checked,
        reason=reason,
        correlation_group=group,
        dependency_paths=paths,
        weight=weight,
        support=weight if direction == "support" else 0.0,
        refutation=0.0,
    )


def coverage(numerator, denominator):
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
        "status": "available" if denominator else "not_applicable",
    }


def channel(rows, role, shift=0.0):
    groups = {}
    for row in rows:
        if row.role != role or row.direction not in {"support", "refute"} or row.weight is None:
            continue
        values = [row.observation.value, row.relation.value, row.applicability.value]
        weight = math.prod(max(0.0, min(1.0, v + shift)) for v in values)
        group = groups.setdefault(row.correlation_group, {"support": 0.0, "refute": 0.0})
        group[row.direction] = max(group[row.direction], weight)
    support = 1.0 - math.prod(1 - g["support"] for g in groups.values())
    refute = 1.0 - math.prod(1 - g["refute"] for g in groups.values())
    return {
        "support": support,
        "refutation": refute,
        "conflict": min(support, refute),
        "net": support - refute,
        "active_groups": len(groups),
    }


def summarize(rows, removed_resources=frozenset(), removed_ids=frozenset()):
    # Denominators are frozen before removal. A missing evidence item cannot
    # improve coverage by silently disappearing from the question inventory.
    active = [
        r
        for r in rows
        if r.id not in removed_ids and path_available(r.dependency_paths, removed_resources)
    ]
    return {
        "assessment_ids": sorted(r.id for r in rows),
        "diagnostic": channel(active, "diagnostic"),
        "risk": channel(active, "risk"),
        "criterion_coverage": coverage(
            sum(r.criterion_checked for r in active), sum(r.kind == "criterion" for r in rows)
        ),
        "observation_coverage": coverage(sum(r.observation_available for r in active), len(rows)),
        "contributing_ids": sorted(r.id for r in active if r.support or r.refutation),
    }


def _node_ids(base, rows):
    own = {n["key"]: {r.id for r in rows if r.node == n["key"]} for n in base["nodes"]}
    edges = base.get("evidence_structure", {}).get("main_edges", [])
    children = {key: [] for key in own}
    for edge in edges:
        children[edge["from"]].append(edge["to"])

    def visit(key, seen):
        if key in seen:
            raise ValueError("research_score_graph_cycle")
        return own[key] | set().union(*(visit(c, seen | {key}) for c in children[key]))

    result = {k: visit(k, set()) for k in own}
    # Root is the complete case inventory, including unavailable auxiliary tests.
    root = next(n["key"] for n in base["nodes"] if n["parent"] is None)
    result[root] = {r.id for r in rows}
    return root, result


def build(base):
    from apps.evidence_graph_details import validate_view

    if "research_scoring" in base or "judgment_details" not in base:
        raise ValueError("research_score_requires_unscored_trace_snapshot")
    validate_view(base)
    ledger = base["judgment_details"]["ledger"]
    rows = [
        _review(a, _available_resources(base))
        for a in ledger["assessments"]
        if a["kind"] != "node_context"
    ]
    rows.sort(key=lambda r: r.id)
    root, ids = _node_ids(base, rows)
    subsets = {k: [r for r in rows if r.id in subset] for k, subset in ids.items()}
    nodes = {k: summarize(v) for k, v in subsets.items()}
    scenarios = [("review:" + r.id, "移除「" + r.title + "」", [], [r.id]) for r in rows]
    scenarios += [
        ("resource:" + resource, "移除" + title, [resource], [])
        for resource, title in [
            ("segman_disc", "SegMAN视盘掩膜"),
            ("segman_cup", "SegMAN视杯掩膜"),
            ("slo_vein", "SLOctolyzer静脉掩膜"),
            ("slo_vessel", "SLOctolyzer血管掩膜"),
            ("fairclip", "FairCLIP输出"),
            ("demographics", "人口学输入"),
            ("image", "本例图像"),
        ]
    ]
    removals = []
    for key, title, resources, removed in scenarios:
        after_nodes = {k: summarize(v, set(resources), set(removed)) for k, v in subsets.items()}
        after, before = after_nodes[root], nodes[root]
        affected = [
            r.id
            for r in rows
            if r.id in removed
            or any(set(path).intersection(resources) for path in r.dependency_paths)
            and not path_available(r.dependency_paths, set(resources))
        ]
        removals.append(
            {
                "id": key,
                "title": title,
                "unavailable_resources": resources,
                "removed_assessments": removed,
                "affected_assessments": affected,
                "after": after,
                "changed_nodes": sorted(k for k in nodes if nodes[k] != after_nodes[k]),
                "diagnostic_support_delta": after["diagnostic"]["support"]
                - before["diagnostic"]["support"],
                "risk_support_delta": after["risk"]["support"] - before["risk"]["support"],
                "criterion_coverage_delta": (
                    after["criterion_coverage"]["value"] - before["criterion_coverage"]["value"]
                    if before["criterion_coverage"]["value"] is not None
                    else None
                ),
                "explanation": (
                    "按必需输入重算；保留独立分支。疾病分数不变也可能损失观察或条件覆盖，"
                    "不能据此删除原始事实。"
                ),
            }
        )
    sensitivity = {
        role + "_support": [channel(rows, role, shift)["support"] for shift in (-0.1, 0.1)]
        for role in ("diagnostic", "risk")
    }
    raw = {
        "schema_version": "case-research-scoring/1.0.0",
        "policy_version": POLICY["version"],
        "policy_checksum": digest(POLICY),
        "case_code": ledger["case_code"],
        "input_checksum": ledger["input_checksum"],
        "base_checksum": base["checksum"],
        "trace_checksum": ledger["checksum"],
        "reviewer": "assistant_manual_simulation_of_llm",
        "reviewed_at": "2026-09-20",
        "purpose": "research_explanation_not_clinical_authorization",
        "disease_probability": None,
        "clinical_acceptance": False,
        "reviews": [r.model_dump(mode="json") for r in rows],
        "resources": _resource_bindings(base),
        "nodes": nodes,
        "removals": removals,
        "sensitivity": sensitivity,
        "method_notes": NOTES,
    }
    return Scores.model_validate({**raw, "checksum": digest(raw)}).model_dump(mode="json")


def validate(base, scores):
    parsed = Scores.model_validate(copy.deepcopy(scores)).model_dump(mode="json")
    if parsed != build(base):
        raise ValueError("research_score_reconstruction_mismatch")
    return parsed
