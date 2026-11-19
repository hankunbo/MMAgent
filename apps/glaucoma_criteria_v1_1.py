"""Operational descriptions and geometric references; keep v1 replay intact."""

import copy
import math
from collections import Counter

from apps import glaucoma_research_criteria as legacy
from apps.glaucoma_evidence_plan import checksum

VERSION = "glaucoma-structural-criteria/1.1.0"
RIM = {
    "id": "EGS-RIM-2011",
    "title": "EGS：最薄盘沿/同轴盘径比",
    "url": "https://eugs.org/news/rim-disc-ratio-is-a-more-sensitive-parameter-than-cup-disc-ratio-2011-06-30-tip-of-the-month",
    "version": "2011-06-30",
    "location": "The Science behind the Tip",
    "fact": "EGS给出下颞侧最薄盘沿/同轴盘径小于0.1的可疑示例，并要求考虑视盘大小和对应视野。",
}
RNFL = {
    "id": "RNFL-SLO-1999",
    "title": "局灶楔形RNFL缺损与盘出血研究",
    "url": "https://pubmed.ncbi.nlm.nih.gov/10485548/",
    "version": "1999",
    "location": "Abstract: Methods, Results, Conclusion",
    "fact": "该研究用氩蓝光SLO识别局灶楔形RNFL缺损，并发现其与盘出血的局部关联。",
}


def policy():
    p = copy.deepcopy(legacy.policy())
    p["version"] = VERSION
    rules = {r["id"]: r for r in p["criteria"]}
    rules["VCDR_GE_07"].update(
        condition="垂直杯盘比越大，越需关注盘沿损失；以≥0.7作为一项偏大特征参照",
        expected_pattern="杯在垂直方向相对视盘扩大，常需同时寻找上/下盘沿局灶变薄或切迹。",
        interpretation_rule="达到0.7增加对偏大视杯的关注；低于0.7仅减弱这一项线索，不把其他线索清零。",
        operationalization="返回实际数值及与0.7的差值；不把差值当患病概率或线性风险权重。",
    )
    rules["CUP_AREA_CONTEXT"].update(
        kind="geometric_reference",
        metric="acdr",
        unit="1",
        threshold=0.49,
        reference_kind="geometric_proxy_not_literature_threshold",
        condition="杯面积/盘面积作尺度归一化；0.49是由0.7²得到的粗略几何参照，不是文献诊断切点",
        expected_pattern="杯占盘面积增加、盘沿占比减少，是需要关注的方向；结合杯的形状和局灶盘沿缺损。",
        interpretation_rule="先看直接测得的垂直杯盘比。面积参照只补充同一杯扩大的描述，不作为第二份独立证据加权。",
        operationalization="等面积直径比=sqrt(杯面积/盘面积)；只有杯与盘近似相似形状时，才可与垂直比作粗略换算。",
        sources=[legacy.PAPER, legacy.EGS["onh"]],
        derivation={
            "owner": "project_geometric_explanation",
            "formula": "0.7 ** 2 = 0.49",
            "published_area_cutoff": False,
            "independent_evidence": False,
        },
    )
    rules["RIM_NOTCH"].update(
        condition="寻找局灶盘沿变薄/切迹；参考最薄径向盘沿宽度÷同轴盘径<0.1的可疑形态",
        expected_pattern="盘沿某一局部明显变窄或缺口，尤其需检查上下极及与RNFL缺损的空间对应。",
        interpretation_rule="缺口/明显变薄是相关线索，不要求整圈盘沿都变薄。<0.1来自EGS下颞侧示例，不能无条件扩展到所有象限。",
        operationalization="当前4个包围盒距离不是径向最薄盘沿测量；眼别未知时也不把图像左右称为鼻/颞侧。",
        sources=[RIM, legacy.PAPER, legacy.EGS["onh"]],
    )
    rules["RNFL_DEFECT"].update(
        condition="查找从盘缘向外延伸、可定位的楔形/扇区性纤维纹理减弱或缺损",
        expected_pattern="与邻近区域相比出现局灶楔形纤维缺损，并核对是否邻接盘沿切迹或出血。",
        interpretation_rule="形态、位置均吻合时作为结构异常线索；孤立暗带先排除血管阴影、亮度不均和扫描条纹。",
        operationalization="来源研究使用氩蓝光SLO；本例未核实同波长等效性。灰度纹理仅作外观检查，不推算RNFL厚度。",
        sources=[RNFL, legacy.PAPER],
    )
    rules["DISC_HEMORRHAGE"].update(
        condition="在盘缘/盘周寻找火焰状或细长裂片样出血候选，核对其与RNFL缺损、盘沿切迹的相邻关系",
        expected_pattern="EGS描述常见火焰形、邻近RNFL缺损，常位于下方并伴切迹；形状宽大或位置不对应时考虑其他原因。",
        interpretation_rule="形态和结构位置共同吻合才增强相关线索；出血本身不是青光眼专属表现。",
        operationalization="灰度SLO中颜色不能作为判据：检查盘缘局部明暗异常、延伸方向及是否与血管连续；这只是搜索候选的办法，不等于识别出了血液。",
    )
    return p


def _area(value, p):
    rows = [r for r in p.get("results", []) if r["name"] == "acdr"]
    if not rows:
        return None
    if len(rows) != 1:
        raise ValueError("criterion_duplicate_metric")
    row = rows[0]
    x = row["value"]
    if x is None:
        return None
    if (
        type(x) not in (int, float)
        or not math.isfinite(x)
        or not 0 <= x <= 1
        or row.get("unit") != "1"
        or row.get("missing_reason")
    ):
        raise ValueError("criterion_metric_invalid")
    s = value["case_evidence"]["supplement"]
    counts = s["mask_check"]["pixel_counts"]
    # Geometry from the same verified mask family, never the alternative disc.
    if (
        s["registered_tool"] != p["tool"]
        or s["mask_check"]["metrics"]["acdr"]["value"] != x
        or not (0 < counts["disc"] and 0 <= counts["cup"] <= counts["disc"])
        or not math.isclose(x, counts["cup"] / counts["disc"], rel_tol=1e-12)
    ):
        raise ValueError("criterion_area_binding")
    if legacy._numeric(value, p, legacy._bound_evidence(value, {"patient_evidence": p}))[1] is None:
        raise ValueError("criterion_area_requires_bound_mask_family")
    return x


def build(value):
    result = legacy.build(value)
    p = policy()
    result.update(schema="case-research-criteria/1.1.0", policy=p, policy_checksum=checksum(p))
    for source in (RIM, RNFL):
        result["source_bindings"][source["id"]] = {
            "url": source["url"],
            "knowledge_source_ref": None,
            "reading_version": source["version"],
            "status": "external_reading_not_published_claim",
        }
    nodes = {n["key"]: n for n in value["nodes"]}
    rules = {r["id"]: r for r in p["criteria"]}
    for r in result["results"]:
        code = r["criterion_id"]
        rule = rules[code]
        if code == "VCDR_GE_07":
            r["distance_to_reference"] = None if r["value"] is None else r["value"] - 0.7
            r["evidence_direction"] = "仅比较杯扩大线索；未达分界不产生反驳结论"
        elif code == "CUP_AREA_CONTEXT":
            x = _area(value, nodes["CUP_RIM_MEASURE"]["patient_evidence"])
            r.update(
                value=x,
                comparison="unavailable"
                if x is None
                else ("above_geometric_reference" if x >= 0.49 else "below_geometric_reference"),
                reason="缺少同图面积比" if x is None else "",
                applicability="geometric_explanation_only",
                equivalent_diameter_ratio=None if x is None else math.sqrt(x),
                distance_to_reference=None if x is None else x - 0.49,
                counted_independently=False,
                applicability_note=rule["operationalization"]
                + " 0.49为项目几何推导，不是原文面积阈值。",
            )
        elif rule["kind"] == "qualitative":
            r["applicability_note"] = rule["operationalization"]
        r["interpretation_rule"] = rule.get("interpretation_rule", rule["operationalization"])
    for key, summary in result["node_summaries"].items():
        summary["comparison_counts"] = dict(
            Counter(r["comparison"] for r in result["results"] if key in r["path"])
        )
    return result


def apply(value):
    result = copy.deepcopy(value)
    result.pop("display_explanations", None)
    result.pop("case_graph_artifact", None)
    result["research_criteria"] = build(value)
    result["checksum"] = checksum({k: v for k, v in result.items() if k != "checksum"})
    return result
