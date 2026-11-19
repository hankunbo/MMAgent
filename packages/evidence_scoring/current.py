"""Read exact normalized records; reuse the original bipolar propagation engine.

No new patient observations or medical effect estimates are made by this module.
"""

import copy
import itertools

from packages.evidence_details.models import digest, resolve
from packages.evidence_scoring import continuous, continuous_policy, participation_policy
from packages.evidence_scoring.continuous_models import ContinuousReview
from packages.evidence_scoring.current_models import CurrentScores
from packages.evidence_scoring.engine import _node_ids, path_available

POLICY = {
    "version": "record-connected-scoring/1.0.0",
    "direction": continuous_policy.POLICY,
    "groups": participation_policy.GROUPS,
    "coverage": "definition_matched_acquisition_and_separate_interpretation_fixed_denominators",
    "missingness": "no_disease_direction_without_observation_and_review",
    "confidence": "existing_bipolar_direction_no_coverage_multiplier",
    "key_set_tolerance": 0.01,
    "key_set_retention": 0.8,
    "key_set_max_candidates": 12,
    "key_set_domains": "retain_each_contributing_role",
    "vascular_bundle": "atomic_artery_vein_disc_and_saved_measurement",
}


def rows_from_records(base):
    from apps.evidence_graph_records import validate_view

    validate_view(base)
    old = {r["id"]: r for r in base["research_scoring"]["reviews"]}
    obs = {o["id"]: o for o in base["evidence_records"]["observations"]}
    rows, obtained, interpreted = [], set(), set()
    resources = copy.deepcopy(base["research_scoring"]["resources"])
    for name, bindings in resources.items():
        for b in bindings:
            if digest(resolve(base, b["pointer"])) != b["checksum"]:
                raise ValueError("current_resource_drift:" + name)
    for j in base["evidence_records"]["interpretations"]:
        o = obs[j["observation_id"]]
        row = copy.deepcopy(old[j["assessment_id"]])
        row.update(
            assessment_checksum=digest({"observation": o, "interpretation": j}),
            source_binding_checksum=digest(j["reference_bindings"]),
            observation=j["observation_reliability"],
            relation=j["relation_strength"],
            applicability=j["applicability"],
            judgment=j["judgment"],
            observation_available=o["acquisition"] == "obtained",
            criterion_checked=j["status"] == "assessed" and row["kind"] == "criterion",
            dependency_paths=j["dependency_paths"],
            correlation_group=j["correlation_group"],
            reason=j["judgment"]["reason"],
        )
        if j["score_action"] == "new_observation_not_rescored":
            pointer = "/vascular_followup/artifact"
            resources["av_measurement"] = [
                {
                    "pointer": pointer,
                    "checksum": digest(resolve(base, pointer)),
                    "purpose": "保存的联合动静脉/视盘测量输出；按实际执行包整体失效",
                }
            ]
            row["dependency_paths"] = [
                ["image", "slo_artery", "slo_vein", "segman_disc", "av_measurement"]
            ]
        values = [row[f]["value"] for f in ("observation", "relation", "applicability")]
        w = values[0] * values[1] * values[2] if all(v is not None for v in values) else None
        e = j["judgment"]["value"]
        row.update(
            weight=w,
            support=w * max(e, 0) if e is not None else 0.0,
            refutation=w * max(-e, 0) if e is not None else 0.0,
            direction="unavailable"
            if e is None
            else "support"
            if e > 0
            else "refute"
            if e < 0
            else "neutral",
        )
        rows.append(ContinuousReview.model_validate(row))
        # An ungradable image and a proxy for a different definition do not fill a slot.
        if o["acquisition"] == "obtained" and j["status"] not in {
            "observation_ungradable",
            "input_missing",
            "method_mismatch",
            "intermediate",
        }:
            obtained.add(row["id"])
        if j["status"] == "assessed":
            interpreted.add(row["id"])
    if obtained - {r.id for r in rows} or interpreted - obtained:
        raise ValueError("current_coverage_binding")
    return rows, resources, obtained, interpreted


def groups(obtained, interpreted):
    return [
        {
            "id": g["id"],
            "title": g["title"],
            "importance": g["importance"],
            "slots": [
                {
                    "assessment_ids": ids,
                    "acquired": bool(set(ids) & obtained),
                    "interpreted": bool(set(ids) & interpreted),
                }
                for ids in g["slots"]
            ],
        }
        for g in POLICY["groups"]
    ]


def completeness(gs, *, scopes=("slo_image_research", "comprehensive_research")):
    result = {}
    for scope in scopes:
        denominator = sum(g["importance"][scope] for g in gs)
        row = {"denominator": denominator}
        for name, flag in (("acquisition", "acquired"), ("interpretation", "interpreted")):
            numerator = sum(
                g["importance"][scope] * sum(s[flag] for s in g["slots"]) / len(g["slots"])
                for g in gs
            )
            row[name] = numerator / denominator if denominator else None
        result[scope] = row
    return result


def state(rows, obtained, interpreted, removed_ids=frozenset(), removed_resources=frozenset()):
    active = {
        r.id
        for r in rows
        if r.id not in removed_ids and path_available(r.dependency_paths, removed_resources)
    }
    return {
        "summary": continuous.summarize(rows, removed_resources, removed_ids),
        "coverage": completeness(groups(obtained & active, interpreted & active)),
    }


def _delta(before, after):
    a, b = before["confidence"]["value"], after["confidence"]["value"]
    return b - a if a is not None and b is not None else None


def conclusion(rows):
    all_rows = continuous.summarize(rows)
    age = continuous.summarize(rows, removed_ids={"criterion:AGE"})
    no_background = continuous.summarize([r for r in rows if r.role != "risk"])
    image = continuous.summarize([r for r in rows if r.dependence_block.startswith("image:")])
    background = continuous.summarize([r for r in rows if r.role == "risk"])
    direction = all_rows["confidence"]["direction"]

    def flip(s):
        other = s["confidence"]["direction"]
        return {direction, other} == {"support_side", "refutation_side"}

    sensitive = (
        flip(age)
        or flip(no_background)
        or all_rows["confidence"]["stability"] == "direction_sensitive"
    )
    text = (
        "当前证据没有可用方向评分。"
        if all_rows["confidence"]["value"] is None
        else "本例研究综合分处于支持侧，但方向受背景或参数选择影响；不能据此确定患病。"
        if direction == "support_side" and sensitive
        else "本例研究综合分处于反驳侧，但方向受背景或参数选择影响；不能据此排除患病。"
        if direction == "refutation_side" and sensitive
        else "本例研究证据方向：" + direction + "；这是证据汇总，不是校准患病概率。"
    )
    return {
        "all_evidence": all_rows,
        "without_age": age,
        "without_background": no_background,
        "image_evidence": image,
        "background_only": background,
        "age_changes_direction": flip(age),
        "background_changes_direction": flip(no_background),
        "age_delta": _delta(age, all_rows),
        "conclusion": text,
        "disease_probability": None,
        "decision_threshold": None,
    }


def key_analysis(rows, obtained, interpreted):
    active = [r for r in rows if r.support or r.refutation]
    tol, retention = POLICY["key_set_tolerance"], POLICY["key_set_retention"]
    roles = sorted({r.role for r in active})
    result = {
        "status": "bounded_not_run",
        "candidates": len(active),
        "evaluated_subsets": 0,
        "score_tolerance": tol,
        "minimum_interpretation_retention": retention,
        "required_roles": roles,
        "smallest_fidelity_sets": [],
        "smallest_direction_flip_removals": [],
        "meaning": "score_fidelity_not_sufficient_for_diagnosis_or_biological_causality",
    }
    if len(active) > POLICY["key_set_max_candidates"]:
        return result
    result["status"] = "exhaustive"
    root = state(rows, obtained, interpreted)
    initial = root["summary"]["confidence"]
    q = root["coverage"]["slo_image_research"]["interpretation"]
    ids = {r.id for r in active}
    flip_size = fidelity_size = None
    for n in range(len(active) + 1):
        for subset in itertools.combinations(active, n):
            chosen = {r.id for r in subset}
            result["evaluated_subsets"] += 1
            s = state(rows, obtained, interpreted, ids - chosen)
            c = s["summary"]["confidence"]
            retained = s["coverage"]["slo_image_research"]["interpretation"] / q if q else 0.0
            if (
                c["value"] is not None
                and initial["value"] is not None
                and abs(c["value"] - initial["value"]) <= tol
                and retained >= retention
                and set(roles) == {r.role for r in subset}
                and (fidelity_size is None or n == fidelity_size)
            ):
                fidelity_size = n
                result["smallest_fidelity_sets"].append(
                    {
                        "assessment_ids": sorted(chosen),
                        "value": c["value"],
                        "interpretation_retention": min(1.0, retained),
                    }
                )
            removed = state(rows, obtained, interpreted, chosen)["summary"]["confidence"]
            if {removed["direction"], initial["direction"]} == {
                "support_side",
                "refutation_side",
            } and (flip_size is None or n == flip_size):
                flip_size = n
                result["smallest_direction_flip_removals"].append(sorted(chosen))
    return result


def missing_scenarios(gs):
    result = []
    for g in gs:
        missing = sorted(
            {k for s in g["slots"] if not s["interpreted"] for k in s["assessment_ids"]}
        )
        if not missing:
            continue
        completed = copy.deepcopy(gs)
        target = next(x for x in completed if x["id"] == g["id"])
        for s in target["slots"]:
            s.update(acquired=True, interpreted=True)
        cov = completeness(completed)
        result.append(
            {
                "group_id": g["id"],
                "unresolved_assessments": missing,
                "kind": "coverage_only_no_assumed_patient_result",
                "acquisition_if_completed": {k: v["acquisition"] for k, v in cov.items()},
                "interpretation_if_reviewed": {k: v["interpretation"] for k, v in cov.items()},
                "disease_effect": None,
                "reason": "假设完成该组全部定义匹配的取证和判读，仅计算覆盖上限；"
                "未给缺失患者结果赋方向。",
            }
        )
    return result


def build(base):
    from packages.evidence_scoring.replacement import build as replacement

    rows, resources, obtained, interpreted = rows_from_records(base)
    root_key, node_ids = _node_ids(base, rows)
    gs = groups(obtained, interpreted)
    baseline = state(rows, obtained, interpreted)
    swap = replacement(base)
    invalidated = {r.id for r in rows if not path_available(r.dependency_paths, {"segman_disc"})}
    swap.update(
        invalidated_assessments=sorted(invalidated),
        remaining_evidence_summary=continuous.summarize(rows, removed_ids=invalidated),
    )
    removals = []
    for key, ids, lost in [
        *[("review:" + r.id, {r.id}, set()) for r in rows],
        *[("resource:" + r, set(), {r}) for r in sorted(resources)],
    ]:
        after = state(rows, obtained, interpreted, ids, lost)
        removals.append(
            {
                "id": key,
                "removed_assessments": sorted(ids),
                "removed_resources": sorted(lost),
                "affected_assessments": sorted(
                    r.id
                    for r in rows
                    if r.id in ids or not path_available(r.dependency_paths, lost)
                ),
                "after": after,
                "confidence_delta": _delta(baseline["summary"], after["summary"]),
            }
        )
    raw = {
        "schema_version": "case-research-scoring/5.0.0",
        "policy_version": POLICY["version"],
        "case_code": base["evidence_records"]["case_code"],
        "input_checksum": base["evidence_records"]["input_checksum"],
        "base_checksum": base["checksum"],
        "records_checksum": base["evidence_records"]["checksum"],
        "policy_checksum": digest(POLICY),
        "reviews": [r.model_dump(mode="json") for r in rows],
        "resources": resources,
        "groups": gs,
        "root": baseline,
        "nodes": {
            k: continuous.summarize([r for r in rows if r.id in ids]) for k, ids in node_ids.items()
        },
        "conclusion": conclusion(rows),
        "removals": removals,
        "key_analysis": key_analysis(rows, obtained, interpreted),
        "missing_scenarios": missing_scenarios(gs),
        "tool_replacement": swap,
        "clinical_acceptance": False,
        "method_notes": [
            "复用原连续双极聚合；可靠性×关系×适用性，同源族去重、共享图像折减。",
            "测量覆盖M与可判读覆盖J分开。二者均不乘入疾病方向分；缺失不填成阴性。",
            "50分为代数中点，不是诊断阈值；上下限是参数敏感性，不是统计置信区间。",
            "新增血管记录按实际联合测量包绑定依赖；宽度代理不替代CRAE/CRVE。",
            "关键集是当前评分的保真压缩，保留各已有论据角色；不等于最小诊断检查集。",
            "移除/替换为模型层面敏感性分析，不声称患者层面的生物因果反事实。",
        ],
    }
    if raw["nodes"][root_key] != baseline["summary"]:
        raise ValueError("current_root_aggregation")
    raw["checksum"] = digest(raw)
    return CurrentScores.model_validate(raw).model_dump(mode="json")


def validate(base, value):
    raw = value.model_dump(mode="json") if isinstance(value, CurrentScores) else value
    parsed = CurrentScores.model_validate(raw).model_dump(mode="json")
    if parsed != build(base):
        raise ValueError("current_source_reconstruction")
    return parsed
