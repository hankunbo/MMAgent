"""Inspection-first case interpretation; measurements are not diagnostic votes.

The visual notes are a recorded review of one exact image/mask family, not an
automated VLM implementation. Other inputs never inherit these observations.
"""

import copy
import math
from collections import Counter

from apps import glaucoma_criteria_v1_1 as previous
from apps import glaucoma_model_selection as selection
from apps import glaucoma_research_criteria as legacy
from apps.glaucoma_evidence_plan import checksum

VERSION = "glaucoma-structural-criteria/1.2.0"
NEI = {
    "id": "NEI-GLAUCOMA-20260917",
    "title": "NEI：青光眼检查与风险背景",
    "url": "https://www.nei.nih.gov/eye-health-information/eye-conditions-and-diseases/glaucoma",
    "version": "read-2026-09-17",
    "fact": "年龄超过60岁及家族史属于风险背景；眼压与青光眼不是一一对应。",
}
VISUAL = {
    "RIM_NOTCH": (
        "盘沿形态",
        "原图与SegMAN叠加图可见盘沿轮廓；分割盘沿环连续，未见明确整段缺口。",
        "本图未见明确局灶切迹；细小局灶变薄受血管遮挡、亮度不均影响，未作定量判断。",
    ),
    "RNFL_DEFECT": (
        "神经纤维层外观",
        "盘周可见放射纹理、血管暗线和横向条带，未找到边界清晰、可稳定定位的楔形缺损。",
        "已完成外观检查，局灶RNFL缺损仍不确定；暗带与成像条纹不能可靠分开。",
    ),
    "DISC_HEMORRHAGE": (
        "盘缘及盘周出血",
        "检查了盘缘及邻近血管，未见可与血管阴影明确分开的裂片状或火焰状候选。",
        "本图未识别到明确出血候选；灰度图对出血性质的辨别有限。",
    ),
    "IMAGE_QUALITY": (
        "图像质量",
        "视盘及主要血管可见；横向条带、亮度不均和下方局部暗影可见。",
        "可查看盘杯轮廓和主要血管；细微盘沿、RNFL和出血观察受伪影影响。",
    ),
}
# Each entry is one result referenced by ancestors and technical consumers.
EXTRA = [
    ("CUP_PIXELS", "CUP_RIM_MEASURE", "杯面积", "统计本次杯掩膜的前景像素", []),
    ("FIELD", "FIELD_VALUES", "视野缺损模式", "检查点位缺损与神经纤维走向的对应", []),
    ("FIELD_RELIABILITY", "FIELD_QC", "视野可靠性与重复性", "核对可靠性记录及重复视野", []),
    ("IOP", "IOP_VALUES", "眼压", "结合测量方法、时间和重复测量解释", [NEI]),
    ("CORNEA", "CCT_VALUES", "中央角膜厚度", "读取同眼厚度及测量方法，辅助解释眼压", []),
    ("ANGLE", "ANGLE_VALUES", "房角", "查看房角检查记录，区分开放与关闭背景", []),
    ("AGE", "RISK_FIELDS", "年龄", "超过60岁列为年龄风险背景", [NEI]),
    ("FAMILY", "RISK_FIELDS", "家族史", "查看是否存在青光眼家族史", [NEI]),
    ("DEMOGRAPHICS", "RISK_FIELDS", "其他人口学信息", "记录已提供的人口学字段", []),
    ("IMAGE_QUALITY", "IMAGE_QC", "图像质量", "检查视盘可见性、照明、条带与阴影", []),
    ("VESSEL_DENSITY", "DENSITY", "血管面积密度", "血管前景像素数 / 整图像素数", []),
    ("VESSEL_WIDTH", "WIDTH", "血管平均宽度", "血管面积 / 骨架长度（像素尺度）", []),
    ("VESSEL_TORTUOSITY", "TORTUOSITY", "血管曲折度", "路径长度 / 两端直线距离；直线参照为1", []),
    ("VESSEL_FRACTAL", "FRACTAL", "血管分形维数", "按既有盒计数算法描述空间复杂度", []),
    ("AV_AREAS", "AV_FEATURES", "动静脉分割面积", "分别统计动脉与静脉掩膜面积", []),
    ("FOVEA", "FOVEA_LOCATION", "中央凹位置", "核对定位点是否位于有效图像内及靠近边界", []),
    ("CLASSIFICATION", "MODEL_ASSESSMENT", "模型分类分数", "比较固定两类的原始输出及分差", []),
]
ALIASES = {
    "DISC_MASK": "DISC_GEOMETRY",
    "SLO_IMAGE": "IMAGE_QC",
    "OUTPUT_QC": "IMAGE_QC",
    "VESSEL_MASK": "VESSEL_FEATURES",
    "TASK_VESSELS": "VESSEL_FEATURES",
    "TASK_DENSITY": "DENSITY",
    "TASK_WIDTH": "WIDTH",
    "TASK_TORTUOSITY": "TORTUOSITY",
    "TASK_FRACTAL": "FRACTAL",
    "ARTERY_MASK": "AV_FEATURES",
    "VEIN_MASK": "AV_FEATURES",
    "TASK_AVOD": "AV_FEATURES",
    "FOVEA_MAP": "FOVEA_LOCATION",
    "TASK_FOVEA": "FOVEA_LOCATION",
}


def policy():
    p = previous.policy()
    p["version"] = VERSION
    for r in p["criteria"]:
        if r["id"] == "VCDR_GE_07":
            r["title"] = "垂直杯盘比"
        elif r["id"] == "CUP_AREA_CONTEXT":
            r["title"] = "面积杯盘比"
            r["condition"] = "杯面积 / 视盘面积；结合垂直杯盘比和盘沿形态解释"
        elif r["id"] in VISUAL:
            r["title"] = VISUAL[r["id"]][0]
    p["criteria"].extend(
        {
            "id": code,
            "leaf": leaf,
            "title": title,
            "kind": "case_observation",
            "condition": condition,
            "sources": sources,
            "operationalization": "从已保存的同例输入、工具结果或本次视觉观察生成。",
        }
        for code, leaf, title, condition, sources in EXTRA
    )
    return p


def _number(x):
    if type(x) not in (int, float) or not math.isfinite(x):
        raise ValueError("assessment_nonfinite_or_non_numeric")
    return x


def _rows(p):
    rows = p.get("results", [])
    if len({r["name"] for r in rows}) != len(rows):
        raise ValueError("assessment_duplicate_metric")
    return {r["name"]: r["value"] for r in rows}


def _visual_review(value):
    # Requires the image and every reviewed mask, not just a self-declared case ID.
    choice = selection.decision(value)
    return {
        "schema": "case-assistant-visual-review/1.0.0",
        "case_code": choice["case_code"],
        "input_checksum": choice["input_checksum"],
        "evidence_refs": choice["evidence_refs"],
        "reviewer": "assistant_in_this_conversation_2026-09-17",
        "method": "direct_image_inspection_not_deployed_vlm",
        "observations": {k: {"observation": v[1], "assessment": v[2]} for k, v in VISUAL.items()},
    }


def build(value):
    out = previous.build(value)
    p = policy()
    nodes = {n["key"]: n for n in value["nodes"]}
    case = value["case_evidence"]
    visual = _visual_review(value)
    out.update(
        schema="case-research-criteria/1.2.0",
        policy=p,
        policy_checksum=checksum(p),
        visual_review=visual,
    )
    out["source_bindings"][NEI["id"]] = {
        "url": NEI["url"],
        "knowledge_source_ref": None,
        "reading_version": NEI["version"],
        "status": "external_reading_not_published_claim",
    }
    for r in out["results"]:
        code = r["criterion_id"]
        if code in VISUAL:
            r.update(
                comparison="visually_reviewed",
                reason="",
                value=None,
                display_result=VISUAL[code][1],
                assessment=VISUAL[code][2],
                applicability="assistant_visual_observation",
                review_ref=visual["schema"],
                evidence_refs=[
                    {"code": case["input"]["asset_code"], "checksum": case["input"]["checksum"]},
                    *visual["evidence_refs"].values(),
                ],
            )
        elif code == "VCDR_GE_07":
            r["display_result"] = "未测得" if r["value"] is None else f"{r['value']:.4f}（参照0.7）"
            r["assessment"] = (
                "低于本项偏大参照。"
                if r["comparison"] == "not_met"
                else "达到本项偏大参照。"
                if r["comparison"] == "met"
                else r["reason"]
            )
        elif code == "CUP_AREA_CONTEXT":
            r["display_result"] = (
                "未测得"
                if r["value"] is None
                else f"{r['value']:.4f}，即杯占盘面积{r['value'] * 100:.2f}%"
            )
            r["assessment"] = "描述同一杯扩大的面积比例；与垂直比合看。"
        elif code == "DISC_SIZE_CONTEXT":
            counts = case["supplement"]["mask_check"]["pixel_counts"]
            r.update(
                display_result=f"SegMAN视盘面积：{counts['disc']} px",
                assessment="像素面积已取得；物理直径缺少标尺。",
            )
        else:
            r.update(display_result=r["reason"], assessment="当前输入无法取证。")

    for code, leaf, _title, _condition, _sources in EXTRA:
        evidence = nodes[leaf].get("patient_evidence", {})
        rows = _rows(evidence)
        r = {
            "criterion_id": code,
            "path": legacy._path(nodes, leaf),
            "value": None,
            "comparison": "observed",
            "reason": "",
            "applicability": "descriptive",
            "applicability_note": "",
            "medical_effect": "not_assigned",
            "correlation_group": case["input"]["checksum"],
            "evidence_refs": [],
            "display_result": "",
            "assessment": "",
        }
        if code != "CLASSIFICATION":
            r["evidence_refs"] = legacy._bound_evidence(value, nodes[leaf])
        if code == "CUP_PIXELS":
            x = _number(case["supplement"]["mask_check"]["pixel_counts"]["cup"])
            r.update(
                value=x,
                display_result=f"{x} px",
                assessment="本次视杯的像素面积；用于计算面积杯盘比。",
            )
        elif code in {"FIELD", "FIELD_RELIABILITY", "IOP", "CORNEA", "ANGLE"}:
            r.update(
                comparison="unavailable",
                display_result=evidence.get("summary", "未提供检查数据"),
                assessment="当前输入无法取证。",
            )
        elif code in {"AGE", "FAMILY", "DEMOGRAPHICS"}:
            if case["supplement"]["demographics"]["image_checksum"] != case["input"]["checksum"]:
                raise ValueError("assessment_demographics_scope")
            demo = case["supplement"]["demographics"]["fields"]
            if code == "AGE":
                x = demo["age"].get("display_value")
                r.update(
                    value=x,
                    comparison="unavailable" if x is None else "observed",
                    display_result="年龄未提供" if x is None else f"{_number(x)} 岁",
                    assessment="年龄风险背景"
                    if x is not None and x > 60
                    else "未达到该年龄背景参照"
                    if x is not None
                    else "当前输入无法取证。",
                )
            elif code == "FAMILY":
                x = demo["family_history"].get("display_value")
                r.update(
                    display_result="未提供家族史" if x is None else str(x),
                    assessment="当前输入无法取证。" if x is None else "按原记录保留家族史。",
                    comparison="unavailable" if x is None else "observed",
                )
            else:
                names = {
                    "gender": "性别",
                    "race": "族群",
                    "ethnicity": "族裔",
                    "language": "语言",
                    "maritalstatus": "婚姻状况",
                }
                r.update(
                    display_result="；".join(
                        f"{names[k]}：{demo[k].get('display_value', '未提供')}" for k in names
                    ),
                    assessment="已记录描述字段，本轮不自动赋疾病权重。",
                )
        elif code == "IMAGE_QUALITY":
            r.update(
                display_result=VISUAL[code][1],
                assessment=VISUAL[code][2],
                comparison="visually_reviewed",
                review_ref=visual["schema"],
            )
            r["evidence_refs"].append(
                {"code": case["input"]["asset_code"], "checksum": case["input"]["checksum"]}
            )
        elif code.startswith("VESSEL_"):
            metric = {
                "DENSITY": "density",
                "WIDTH": "width",
                "TORTUOSITY": "tortuosity",
                "FRACTAL": "fractal",
            }[leaf]
            x = rows.get(metric)
            raw = case["results"][metric]["observation"]["measurement"]
            if x != raw["value"]:
                raise ValueError("assessment_measurement_binding")
            if x is None:
                r.update(
                    comparison="unavailable",
                    display_result="未取得测量值",
                    assessment="当前没有该项结果。",
                )
            else:
                x = _number(x)
                r.update(value=x, display_result=f"{x:.4f}")
                r["assessment"] = {
                    "density": f"血管覆盖整图的{x * 100:.2f}%。",
                    "width": f"按面积/骨架长度得到平均宽度{x:.2f} px。",
                    "tortuosity": f"路径/弦长比的中位数比直线参照高{(x - 1) * 100:.2f}%。",
                    "fractal": "记录血管空间复杂度，供同方法纵向或群体比较。",
                }[metric] + " 暂无本方法的青光眼分界参照。"
        elif code == "AV_AREAS":
            counts = case["results"]["avod"]["observation"]["model"]["foreground_pixels"]
            if any(rows.get(k + "面积") != counts.get(k) for k in ("artery_mask", "vein_mask")):
                raise ValueError("assessment_measurement_binding")
            r.update(
                display_result=(
                    f"动脉{rows.get('artery_mask面积')} px；静脉{rows.get('vein_mask面积')} px"
                ),
                assessment="已取得两类分割面积；尚未计算分区动静脉口径比。",
            )
            if any(counts.get(k) is None for k in ("artery_mask", "vein_mask")):
                r.update(
                    comparison="unavailable",
                    display_result="动静脉面积记录不完整",
                    assessment="当前不能完成两类面积对照。",
                )
        elif code == "FOVEA":
            xy = rows.get("中央凹(x,y)")
            if xy != case["results"]["fovea"]["observation"]["model"].get("fovea_xy"):
                raise ValueError("assessment_measurement_binding")
            if xy is None:
                r.update(
                    comparison="unavailable",
                    display_result="未取得坐标",
                    assessment="当前没有定位结果。",
                )
            else:
                x, y = map(_number, xy)
                h, w = case["input"]["dimensions"][:2]
                boundary = not (0 < x < w - 1 and 0 < y < h - 1)
                r.update(
                    display_result=f"({x:g}, {y:g})；图像{w}×{h}",
                    assessment="定位落在边界，暂不用于确定解剖方向。"
                    if boundary
                    else "坐标位于图像内部；仍需与解剖位置核对。",
                )
        elif code == "CLASSIFICATION":
            outputs = case["fairclip"].get("outputs", [])
            if len(outputs) != 1:
                raise ValueError("assessment_model_output_ambiguous")
            record = outputs[0]
            output, source = record["output"], record["source"]
            if (
                output["input_checksum"] != case["input"]["checksum"]
                or source["scope"]["case_code"] != case["case_code"]
            ):
                raise ValueError("assessment_model_scope")
            a, b = map(_number, output["normalized_model_scores"])
            if not (0 <= a <= 1 and 0 <= b <= 1):
                raise ValueError("assessment_model_score_range")
            r.update(
                display_result=f"非青光眼{a * 100:.4f}%；青光眼{b * 100:.4f}%",
                assessment=f"两类分差{abs(b - a) * 100:.4f}个百分点；"
                + ("青光眼类较高。" if b > a else "非青光眼类较高。" if a > b else "两类持平。")
                + ("两类输出接近。" if 0 < abs(b - a) < 0.01 else "")
                + "这是未校准模型分数。",
                value=b - a,
            )
            r["evidence_refs"] = [
                {"code": source["artifact_code"], "checksum": source["artifact_checksum"]}
            ]
        out["results"].append(r)
    for key in nodes:
        relevant = [r for r in out["results"] if key in r["path"] or ALIASES.get(key) in r["path"]]
        out["node_summaries"][key] = {
            "criterion_ids": [r["criterion_id"] for r in relevant],
            "comparison_counts": dict(Counter(r["comparison"] for r in relevant)),
            "meaning": "同一观察的路径引用，不重复计分",
        }
    return out


def apply(value):
    result = copy.deepcopy(value)
    result.pop("display_explanations", None)
    result.pop("case_graph_artifact", None)
    result["research_criteria"] = build(value)
    result["checksum"] = checksum({k: v for k, v in result.items() if k != "checksum"})
    return result
