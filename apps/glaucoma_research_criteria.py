"""Source-bound research comparisons, not the accepted clinical rule engine.

The fixed v1 policy is deliberately small. A comparison is one fact referenced
at multiple ancestors, never multiple diagnostic votes. No caller-set threshold.
"""

import copy
import math
from collections import Counter

from apps.glaucoma_evidence_plan import checksum

VERSION = "glaucoma-structural-criteria/1.0.0"
PAPER = {
    "id": "PHENE-2019-V2",
    "title": "Phene等：眼底照片中视盘特征与青光眼转诊判断",
    "url": "https://arxiv.org/abs/1812.08911v2",
    "version": "arXiv:1812.08911v2",
    "location": "Abstract Results; Methods: Datasets / Feature-Level Grading",
    "fact": "研究将较大的垂直杯盘比、盘沿切迹及RNFL缺损等与需转诊的视神经外观联系起来。",
    "population_and_endpoint": "彩色眼底照片；需转诊的青光眼性视神经病变，而非单项确诊",
    "ingestion": "本次直接阅读的原始论文；尚未登记为Knowledge Published Claim",
}
EGS = {
    "onh": {
        "id": "EGS-ONH-2019",
        "title": "EGS：视神经头检查",
        "url": "https://www.eugs.org/news/clinical-examination-of-the-optic-nerve-head-2019-09-30-pearl",
        "fact": "杯盘比受视盘大小影响，应结合盘沿形态解释，不能仅用杯盘比分类。",
    },
    "oct": {
        "id": "EGS-OCT-2020",
        "title": "EGS：OCT与青光眼诊断",
        "url": "https://eugs.org/news/optical-coherence-tomography-for-glaucoma-diagnosis-2020-12-30-pearl",
        "fact": "OCT结果需结合参考数据库、图像质量和临床背景解释。",
    },
    "bleed": {
        "id": "EGS-HEMORRHAGE-2024",
        "title": "EGS：并非所有盘周出血均为青光眼",
        "url": "https://www.eugs.org/news/tip-not-every-peripapillary-hemorrhage-is-a-sign-for-glaucoma-2024-06-16-tip-of-the-month",
        "fact": "盘周出血存在其他解释，需要核对形态、位置及相关背景。",
    },
    "change": {
        "id": "EGS-RNFL-2017",
        "title": "EGS：RNFL测量与早期进展",
        "url": "https://eugs.org/news/oct-measurement-of-the-retinal-nerve-fibre-layer-is-useful-for-detecting-progression-in-early-glaucoma-2017-12-31-tip-of-the-month",
        "fact": "重复OCT测量可用于评估早期RNFL变化，晚期存在测量下限。",
    },
}


def policy():
    """Fresh policy value; source findings and our operational choices are distinct."""
    items = [
        (
            "VCDR_GE_07",
            "CUP_RIM_MEASURE",
            "垂直杯盘比偏大",
            "numeric",
            "vcdr ≥ 0.7",
            [PAPER, EGS["onh"]],
            "以同图杯/盘垂直直径之比作探索性对照；不是杯面积比。",
        ),
        (
            "CUP_AREA_CONTEXT",
            "CUP_RIM_MEASURE",
            "杯面积与面积杯盘比",
            "reference",
            "结合视盘大小与盘沿，不设置通用像素面积阈值",
            [EGS["onh"]],
            "像素面积受分辨率和视野影响；面积比不得代入垂直直径比的阈值。",
        ),
        (
            "RIM_NOTCH",
            "CUP_RIM_MEASURE",
            "局灶盘沿切迹/变薄",
            "qualitative",
            "辨认局灶盘沿缺损及其位置",
            [PAPER, EGS["onh"]],
            "当前包围盒间距不能代表象限盘沿厚度，也不自动判定ISNT或局灶切迹。",
        ),
        (
            "RNFL_DEFECT",
            "RNFL_REFLECTANCE",
            "局灶神经纤维层外观缺损",
            "qualitative",
            "辨认可定位的RNFL外观缺损",
            [PAPER],
            "外观缺损与OCT厚度分开；不确定观察不视为正常。",
        ),
        (
            "DISC_HEMORRHAGE",
            "DISC_BLEED",
            "盘缘/盘周出血线索",
            "qualitative",
            "辨认出血形态和位置，并保留其他解释",
            [EGS["bleed"]],
            "灰度影像未能可靠区分时，不把血管或阴影计为出血。",
        ),
        (
            "OCT_REFERENCE",
            "RNFL_THICKNESS",
            "RNFL/GCC厚度参照",
            "required_input",
            "取得分区厚度、设备参考范围及扫描质量后比较",
            [EGS["oct"]],
            "不跨设备套统一微米阈值；SLO外观不能代替OCT分层厚度。",
        ),
        (
            "DISC_SIZE_CONTEXT",
            "DISC_GEOMETRY",
            "视盘大小背景",
            "required_input",
            "结合标尺和成像方法解释物理视盘大小",
            [EGS["onh"]],
            "不同模型的视盘掩膜分别记录，不混用其面积作杯盘比的分母。",
        ),
        (
            "SAME_EYE_CHANGE",
            "TEMPORAL_CHANGE",
            "同眼纵向变化",
            "required_input",
            "同眼可比的重复检查及质量/时间信息",
            [EGS["change"]],
            "OCT进展依据不自动转成SLO像素变化阈值；单时点不计算进展。",
        ),
    ]
    return {
        "version": VERSION,
        "authorship": "project_source_informed_research_operationalization",
        "clinical_authorization": False,
        "aggregation": "unique_criterion_references_no_disease_score",
        "criteria": [
            {
                "id": code,
                "leaf": leaf,
                "title": title,
                "kind": kind,
                "condition": condition,
                "sources": sources,
                "operationalization": note,
                **(
                    {"metric": "vcdr", "operator": "ge", "threshold": 0.7, "unit": "1"}
                    if code == "VCDR_GE_07"
                    else {}
                ),
            }
            for code, leaf, title, kind, condition, sources, note in items
        ],
    }


def _path(nodes, leaf):
    path, cursor = [], leaf
    while cursor is not None:
        if cursor not in nodes or cursor in path:
            raise ValueError("criterion_path_invalid")
        path.append(cursor)
        cursor = nodes[cursor]["parent"]
    if path[-1] != "ROOT":
        raise ValueError("criterion_root_invalid")
    return list(reversed(path))


def _bound_evidence(value, node):
    case = value["case_evidence"]
    p = node.get("patient_evidence", {})
    known = {a["asset_code"]: a for a in [case["input"], *case["artifacts"]]}
    refs = []
    for code in p.get("artifact_codes", []):
        a = known.get(code)
        if a is None or a["scope"]["case_code"] != case["case_code"]:
            raise ValueError("criterion_evidence_scope")
        refs.append({"code": code, "checksum": a["checksum"]})
    return sorted(refs, key=lambda r: r["code"])


def _numeric(value, p, refs):
    rows = [r for r in p.get("results", []) if r["name"] == "vcdr"]
    if not rows:
        return "unavailable", None, "缺少垂直杯盘比测量"
    if len(rows) != 1:
        raise ValueError("criterion_duplicate_metric")
    row = rows[0]
    x = row.get("value")
    if x is None:
        return "unavailable", None, "本次工具未返回垂直杯盘比"
    if (
        type(x) not in (int, float)
        or not math.isfinite(x)
        or not 0 <= x <= 1
        or row.get("unit") != "1"
        or row.get("missing_reason")
    ):
        raise ValueError("criterion_metric_invalid")
    s = value["case_evidence"].get("supplement", {})
    case = value["case_evidence"]
    if (
        not refs
        or s.get("case_code") != case["case_code"]
        or s.get("input_checksum") != case["input"]["checksum"]
        or s.get("registered_tool") != p.get("tool")
        or sorted(s.get("artifact_codes", [])) != sorted(r["code"] for r in refs)
        or s.get("mask_check", {}).get("metrics", {}).get("vcdr", {}).get("value") != x
    ):
        raise ValueError("criterion_metric_binding")
    return ("met" if x >= 0.7 else "not_met"), x, ""


def build(value):
    """Called after verified case projection. Produces explanatory comparisons only."""
    nodes = {n["key"]: n for n in value["nodes"]}
    case = value["case_evidence"]
    definitions = policy()
    pages = {p["id"]: p for p in value.get("egs_pages", [])}
    results = []
    for rule in definitions["criteria"]:
        node = nodes[rule["leaf"]]
        p = node.get("patient_evidence", {})
        refs = _bound_evidence(value, node)
        comparison, measured, reason = "unavailable", None, ""
        applicability = "not_evaluable"
        if rule["id"] == "VCDR_GE_07":
            comparison, measured, reason = _numeric(value, p, refs)
            applicability = "exploratory_only" if measured is not None else "not_evaluable"
            note = "文献为彩色眼底转诊特征；当前SLO掩膜几何对照尚未验证模态/测量方法等效性。"
        elif rule["kind"] == "reference":
            comparison, applicability = "reference_only", "reference_only"
            note = rule["operationalization"]
        elif rule["kind"] == "qualitative":
            review = p.get("observation_source", {})
            # V1 has no approved qualitative-to-disease classifier; preserve the
            # observed description, never convert free text into a positive vote.
            reason = "已有观察但判读不确定" if review else "尚无符合该条件定义的形态判读"
            note = rule["operationalization"]
            if review and review.get("image_checksum") != case["input"]["checksum"]:
                raise ValueError("criterion_visual_scope")
        else:
            reason = {
                "OCT_REFERENCE": "当前输入无OCT分区厚度与设备参考范围",
                "DISC_SIZE_CONTEXT": "已有像素面积；当前输入无物理标尺",
                "SAME_EYE_CHANGE": "当前仅有单时点影像",
            }[rule["id"]]
            note = rule["operationalization"]
        results.append(
            {
                "criterion_id": rule["id"],
                "path": _path(nodes, rule["leaf"]),
                "comparison": comparison,
                "value": measured,
                "reason": reason,
                "applicability": applicability,
                "applicability_note": note,
                "medical_effect": "not_assigned",
                "evidence_refs": refs,
                "observation": p.get("summary", ""),
                "correlation_group": case["input"]["checksum"],
            }
        )
    return {
        "schema": "case-research-criteria/1.0.0",
        "policy": definitions,
        "policy_checksum": checksum(definitions),
        "source_bindings": {
            source["id"]: {
                "url": source["url"],
                "knowledge_source_ref": pages.get(source["id"], {}).get("source_ref"),
                "reading_version": source.get("version"),
                "status": "archived_knowledge_source"
                if pages.get(source["id"], {}).get("source_ref")
                else "external_reading_not_published_claim",
            }
            for rule in definitions["criteria"]
            for source in rule["sources"]
        },
        "case_code": case["case_code"],
        "input_checksum": case["input"]["checksum"],
        "results": results,
        "node_summaries": {
            key: {
                "criterion_ids": [r["criterion_id"] for r in results if key in r["path"]],
                "comparison_counts": dict(
                    Counter(r["comparison"] for r in results if key in r["path"])
                ),
                "meaning": "条件对照数量，不是疾病支持率或概率",
            }
            for key in nodes
        },
        "clinical_result": "not_computed",
    }


def apply(value):
    result = copy.deepcopy(value)
    result.pop("display_explanations", None)
    result.pop("case_graph_artifact", None)
    result["research_criteria"] = build(value)
    result["checksum"] = checksum({k: v for k, v in result.items() if k != "checksum"})
    return result


def validate(value):
    if "research_criteria" not in value:
        return
    version = value["research_criteria"].get("policy", {}).get("version")
    builder = build
    if version == "glaucoma-structural-criteria/1.1.0":
        from apps.glaucoma_criteria_v1_1 import build as builder
    elif version == "glaucoma-structural-criteria/1.2.0":
        from apps.glaucoma_criteria_v1_2 import build as builder
    elif version == "glaucoma-structural-criteria/1.3.0":
        from apps.glaucoma_criteria_v1_3 import build as builder
    elif version != VERSION:
        raise ValueError("criterion_version_unknown")
    if value["research_criteria"] != builder(value):
        raise ValueError("criterion_result_mismatch")
