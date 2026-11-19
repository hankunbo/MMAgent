"""Only source-defined disease features, risk factors and referral triggers in §5.

Descriptive tool output stays in §4. This reporting policy does not create a
clinical graph acceptance event, probability or an independently weighted vote.
"""

import copy
from collections import Counter

from apps import glaucoma_criteria_v1_1 as v11
from apps import glaucoma_criteria_v1_2 as previous
from apps import glaucoma_research_criteria as legacy
from apps.glaucoma_evidence_plan import checksum
from apps.glaucoma_rim_radial import from_snapshot

VERSION = "glaucoma-structural-criteria/1.3.0"
AAO = {
    "id": "AAO-POAG-CLINICAL-FEATURES",
    "title": "AAO：原发性开角型青光眼的临床特征",
    "url": "https://eguideline.guidelinecentral.com/i/655134-primary-open-angle-glaucoma/5",
    "version": "read-2026-09-17",
    "fact": (
        "可重复的鼻侧阶梯、弓形或旁中心成簇视野缺损，与神经纤维损伤相应；"
        "随访可见盘沿变薄及杯凹扩大。"
    ),
}
NICE = {
    "id": "NICE-NG81-REFERRAL",
    "title": "NICE NG81：眼压检查与转诊",
    "url": "https://www.nice.org.uk/guidance/ng81/chapter/Recommendations",
    "version": "NG81; 1.1.2/1.1.4/1.1.5",
    "fact": (
        "Goldmann型压平眼压达到24 mmHg属于转诊条件之一；应考虑重复测量，"
        "不能只凭非接触眼压决定转诊。"
    ),
}
RETAIN = {
    "VCDR_GE_07",
    "RIM_NOTCH",
    "RNFL_DEFECT",
    "DISC_HEMORRHAGE",
    "SAME_EYE_CHANGE",
    "FIELD",
    "IOP",
    "AGE",
    "FAMILY",
}
EXCLUDED = {
    "CUP_PIXELS": "像素面积随图像尺度变化，没有当前可用疾病阈值。",
    "CUP_AREA_CONTEXT": "没有当前可用的面积比医学阈值；0.49几何换算不作疾病判据。",
    "DISC_SIZE_CONTEXT": "像素盘大小用于解释测量，当前没有物理标尺和判读阈值。",
    "OCT_REFERENCE": "缺少匹配设备/人群的厚度参考库及本例OCT输入。",
}


def policy():
    p = previous.policy()
    p["version"] = VERSION
    p["selection_rule"] = "source_backed_relation_and_explicit_threshold_or_phenomenon"
    p["criteria"] = [r for r in p["criteria"] if r["id"] in RETAIN]
    rules = {r["id"]: r for r in p["criteria"]}
    definitions = {
        "VCDR_GE_07": (
            "numeric",
            "垂直杯盘比 ≥ 0.7",
            "structural_feature",
            [legacy.PAPER, legacy.EGS["onh"]],
        ),
        "RIM_NOTCH": (
            "phenomenon",
            "盘沿出现可定位的局灶切迹或局部显著变薄，尤其检查上下极",
            "structural_feature",
            [legacy.PAPER, legacy.EGS["onh"]],
        ),
        "RNFL_DEFECT": (
            "phenomenon",
            "盘缘向外延伸的局灶楔形/扇区性纤维缺损，与邻近结构位置对应",
            "structural_feature",
            [v11.RNFL, legacy.PAPER],
        ),
        "DISC_HEMORRHAGE": (
            "phenomenon",
            "盘缘/盘周有火焰状或裂片样出血，核对邻近RNFL缺损或盘沿切迹",
            "structural_feature",
            [legacy.EGS["bleed"]],
        ),
        "SAME_EYE_CHANGE": (
            "phenomenon",
            "同眼可比随访显示盘沿继续变薄、杯凹扩大，或缺损周向范围扩大",
            "progression_feature",
            [AAO, v11.RIM],
        ),
        "FIELD": (
            "phenomenon",
            "可重复的鼻侧阶梯、弓形或旁中心成簇缺损，与RNFL损伤位置对应",
            "functional_feature",
            [AAO],
        ),
        "IOP": (
            "numeric",
            "Goldmann型压平眼压 ≥ 24 mmHg，并核对重复测量",
            "referral_trigger",
            [NICE],
        ),
        "AGE": ("numeric", "年龄 > 60岁", "risk_context", [previous.NEI]),
        "FAMILY": ("phenomenon", "已记录青光眼家族史", "risk_context", [previous.NEI]),
    }
    for code, (kind, condition, relation, sources) in definitions.items():
        rules[code].update(kind=kind, condition=condition, relation_role=relation, sources=sources)
    rules["IOP"].update(
        operator="ge", threshold=24, unit="mmHg", required_method="Goldmann-type applanation"
    )
    rules["AGE"].update(operator="gt", threshold=60, unit="year")
    p["criteria"].insert(
        1,
        {
            "id": "RIM_RADIAL_LT_01",
            "leaf": "CUP_RIM_MEASURE",
            "title": "最薄径向盘沿比例",
            "kind": "numeric",
            "operator": "lt",
            "threshold": 0.1,
            "unit": "1",
            "condition": "最薄径向盘沿宽度 / 同轴视盘直径 < 0.1",
            "relation_role": "structural_feature",
            "sources": [v11.RIM],
            "operationalization": (
                "EGS给出颞下区示例并要求结合盘大小。"
                "本例按图像坐标全圆采样，眼别未知，不指定鼻/颞方位。"
            ),
        },
    )
    return p


def _summary(nodes, results):
    return {
        key: {
            "criterion_ids": [
                r["criterion_id"]
                for r in results
                if key in r["path"] or previous.ALIASES.get(key) in r["path"]
            ],
            "meaning": "同一条件沿路径引用，不重复计分",
        }
        for key in nodes
    }


def build(value):
    out = previous.build(value)
    p = policy()
    old_rules = out["policy"]["criteria"]
    all_results = out["results"]
    nodes = {n["key"]: n for n in value["nodes"]}
    radial = from_snapshot(value)
    results = [r for r in all_results if r["criterion_id"] in RETAIN]
    by_id = {r["criterion_id"]: r for r in results}
    vcdr = by_id["VCDR_GE_07"]
    vcdr["assessment"] = {
        "met": "满足杯盘比偏大条件，可提供这一项青光眼相关结构线索。",
        "not_met": "不满足≥0.7条件；此项指标不能说明本例有青光眼。",
        "unavailable": "缺少测量，未检查这一条件。",
    }[vcdr["comparison"]]
    observations = {
        "RIM_NOTCH": "本图未见明确局灶切迹；未观察到所定义的现象，本项不提供青光眼支持。",
        "RNFL_DEFECT": "本图未找到可稳定定位的楔形缺损；本次观察未提供该项青光眼支持。",
        "DISC_HEMORRHAGE": (
            "本图未识别到明确的火焰状/裂片样出血候选；本次观察未提供该项青光眼支持。"
        ),
    }
    for code, assessment in observations.items():
        by_id[code].update(comparison="not_observed", assessment=assessment)
        out["visual_review"]["observations"][code]["assessment"] = assessment
    for code in ("SAME_EYE_CHANGE", "FIELD", "IOP", "FAMILY"):
        if by_id[code]["comparison"] == "unavailable":
            by_id[code]["assessment"] = "当前输入没有对应检查资料，本项未检查。"
    age = by_id["AGE"]
    if age["value"] is not None:
        age.update(
            comparison="met" if age["value"] > 60 else "not_met",
            assessment="满足年龄风险条件，作为青光眼风险背景。"
            if age["value"] > 60
            else "不满足>60岁条件，不提供这一年龄风险线索。",
        )
    minimum = radial["minimum"]
    low = minimum["ratio"] < 0.1
    r = {
        "criterion_id": "RIM_RADIAL_LT_01",
        "path": legacy._path(nodes, "CUP_RIM_MEASURE"),
        "value": minimum["ratio"],
        "comparison": "met" if low else "not_met",
        "reason": "",
        "display_result": (
            f"约{minimum['rim_width_px']:g} px / {minimum['disc_diameter_px']:g} px"
            f" = {minimum['ratio']:.3f}"
        ),
        "assessment": "满足<0.1的数值条件，提示局部盘沿偏薄；需定位其解剖方位。"
        if low
        else "全圆采样未见<0.1；此项比例未提供盘沿显著变薄的青光眼支持。",
        "applicability": "mask_based_image_axis_estimate",
        "applicability_note": p["criteria"][1]["operationalization"],
        "medical_effect": "not_assigned",
        "evidence_refs": radial["evidence_refs"],
        "correlation_group": out["input_checksum"],
        "measurement_ref": radial["schema"],
    }
    results.insert(1, r)
    for r in results:
        definition = next(d for d in p["criteria"] if d["id"] == r["criterion_id"])
        r["relation_role"] = definition["relation_role"]
        r["result_status"] = "未检查" if r["comparison"] == "unavailable" else "已有结果"
    descriptive = [r for r in all_results if r["criterion_id"] not in RETAIN]
    for r in descriptive:
        r["placement_reason"] = EXCLUDED.get(
            r["criterion_id"], "这是测量、质量或背景结果，当前没有匹配的独立疾病判据。"
        )
        if r["criterion_id"] == "CUP_AREA_CONTEXT":
            for k in ("distance_to_reference", "interpretation_rule", "equivalent_diameter_ratio"):
                r.pop(k, None)
            r.update(
                comparison="observed",
                applicability="descriptive",
                assessment="记录杯占盘面积比例。",
                applicability_note="面积比保留为测量值，不使用0.49几何换算作疾病阈值。",
            )
    descriptive_rules = [r for r in old_rules if r["id"] not in RETAIN]
    for r in descriptive_rules:
        if r["id"] == "CUP_AREA_CONTEXT":
            for k in ("threshold", "derivation", "reference_kind"):
                r.pop(k, None)
            r.update(kind="measurement", operationalization="同图杯掩膜面积 / 视盘掩膜面积。")
    out.update(
        schema="case-research-criteria/1.3.0",
        policy=p,
        policy_checksum=checksum(p),
        results=results,
        descriptive_criteria=descriptive_rules,
        descriptive_results=descriptive,
        node_summaries=_summary(nodes, results),
        descriptive_node_summaries=_summary(nodes, descriptive),
        radial_rim_measurement=radial,
    )
    for source in (AAO, NICE):
        out["source_bindings"][source["id"]] = {
            "url": source["url"],
            "knowledge_source_ref": None,
            "reading_version": source["version"],
            "status": "external_reading_not_published_claim",
        }
    out["visual_review"]["schema"] = "case-assistant-visual-review/1.1.0"
    out["visual_review"]["completed"] = True
    for r in results + descriptive:
        if r.get("review_ref"):
            r["review_ref"] = out["visual_review"]["schema"]
    return out


def apply(value):
    result = copy.deepcopy(value)
    result.pop("display_explanations", None)
    result.pop("case_graph_artifact", None)
    criteria = build(value)
    result["research_criteria"] = criteria
    for n in result["nodes"]:
        p = n.get("patient_evidence", {})
        for obj, field in [
            (p, "status"),
            (p.get("acquisition", {}), "label"),
            *[(c, "status") for c in p.get("children", [])],
        ]:
            if obj.get(field) in {"已有观察", "已有观察·判读不确定"}:
                obj[field] = "已有结果"
        if n["key"] == "CUP_RIM_MEASURE":
            p["results"] = [r for r in p["results"] if r["name"] != "最薄径向盘沿比例"]
            p["results"].append(
                {
                    "name": "最薄径向盘沿比例",
                    "value": criteria["radial_rim_measurement"]["minimum"]["ratio"],
                    "unit": "1",
                    "method": "SegMAN掩膜径向采样估计（本轮报告计算）",
                }
            )
    result["observation_summary"] = dict(
        Counter(n["patient_evidence"]["status"] for n in result["nodes"])
    )
    result["checksum"] = checksum({k: v for k, v in result.items() if k != "checksum"})
    return result
