"""Append this case's visual review and verified geometry to the saved graph.

Not a generic VLM and not a clinical acceptance service. Review statements are
bound to the one image actually inspected; geometry is rebuilt from its masks.
"""

import argparse
import base64
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from apps import disc_geometry as geometry
from apps.disc_geometry_tool_registry import FORMULAS, IMPLEMENTATION, SOURCES, TOOLS
from apps.glaucoma_evidence_plan import checksum
from packages.tool_execution.qc import grayscale_png

VERSION = "case-completion-review/1.0.0"
CASE = "CASE:ANON_85A135B787D34A23861AB252ADC26150"
IMAGE = "sha256:2ba00b68ec05340ae68a7b475786bff05c2153344cd9ab67ae88e58c6b42c9cd"
AAO = "https://www.aao.org/education/preferred-practice-pattern/primary-open-angle-glaucoma-ppp"
OBSERVATIONS = [
    {
        "id": "fovea",
        "title": "中央凹定位",
        "nodes": ["FOVEA_LOCATION", "FOVEA_MAP", "TASK_FOVEA"],
        "finding": "not_localizable",
        "coordinate_xy": None,
        "observation": (
            "已检查原图。视盘和主要血管可见；"
            "右边缘的模型点处未见可辨认的中央凹中心，其他区域也未能可靠定位。"
        ),
        "use": "本次视觉观察已完成，但不输出坐标；鼻颞定向、盘—中央凹轴和ISNT暂不计算。",
        "sources": ["https://github.com/jaburke166/SLOctolyzer"],
    },
    {
        "id": "vessel_course",
        "title": "盘内血管走行",
        "nodes": ["CUP_RIM_MEASURE", "AV_FEATURES"],
        "finding": "not_established",
        "observation": (
            "可见血管从盘内向图像上、下方及两侧延伸和弯曲；"
            "没有辨认出可稳定定位的血管悬空或刺刀样转折。"
        ),
        "use": "保留走行观察；中央凹与方向未确定，不把图像左侧直接命名为鼻侧移位。",
        "sources": [AAO],
    },
    {
        "id": "peripapillary",
        "title": "盘周形态与其他解释",
        "nodes": ["ALTERNATIVES", "DISC_GEOMETRY"],
        "finding": "appearance_observed",
        "observation": (
            "可见纵向椭圆样盘区及不均匀明暗边界；"
            "灰度亮带与条带同时存在，未能单独划定明确的盘周萎缩边界。"
        ),
        "use": "记录外观与同图椭圆度；不由灰度亮带判定beta/gamma分型、近视度数或视神经苍白。",
        "sources": [AAO],
    },
    {
        "id": "rim",
        "title": "局灶盘沿形态",
        "nodes": ["CUP_RIM_MEASURE"],
        "finding": "not_observed",
        "observation": (
            "本次原图观察未辨认出明确局灶盘沿切迹；局部边界受血管和明暗变化影响。"
            "盘沿数值使用同图掩膜算法，不凭目视虚填像素值。"
        ),
        "use": "未观察到所定义的切迹，不增加该项支持；与杯盘测量共享同一图像来源。",
        "sources": [AAO],
    },
    {
        "id": "rnfl",
        "title": "RNFL外观",
        "nodes": ["RNFL_REFLECTANCE"],
        "finding": "not_localizable",
        "observation": (
            "盘周有放射状明暗纹理和横向条带；"  # Preserve the recorded observation verbatim.
            "未辨认出边界、走向均可稳定定位的楔形RNFL缺损。"
        ),
        "use": "没有形成明确缺损阳性观察，也不据此排除缺损；厚度仍需OCT或测量报告。",
        "sources": [AAO],
    },
    {
        "id": "hemorrhage",
        "title": "盘缘出血形态",
        "nodes": ["DISC_BLEED"],
        "finding": "not_established",
        "observation": (
            "盘周暗线多与血管走行相连；"  # Preserve the recorded observation verbatim.
            "本灰度图未识别到可与血管或成像暗影清楚分开的裂片样出血候选。"
        ),
        "use": "没有形成明确出血阳性观察；此结果不等同于已排除出血。",
        "sources": [AAO],
    },
    {
        "id": "quality",
        "title": "本图可观察范围",
        "nodes": ["IMAGE_QC"],
        "finding": "visible_artifacts",
        "observation": "视盘及主要血管可见；图像偏暗，有横向条带、亮度不均和下方暗影。",
        "use": "继续使用有同图来源的杯盘测量；中央凹、细微RNFL和小出血分别记录可辨认程度。",
        "sources": ["https://github.com/jaburke166/SLOctolyzer"],
    },
]
UNAVAILABLE = {
    "RNFL_THICKNESS": "没有OCT像素或有方法、分区的RNFL/GCC厚度报告。",
    "TEMPORAL_CHANGE": "没有同眼其他时点的可比影像。",
    "FIELD_VALUES": "没有视野点位、缺损图或检查报告。",
    "FIELD_QC": "没有视野可靠性和重复检查记录。",
    "IOP_VALUES": "没有实测眼压、方法及重复测量。",
    "CCT_VALUES": "没有角膜厚度测量。",
    "ANGLE_VALUES": "没有房角镜或前节检查。",
    "DISC_SIZE": "没有物理标尺或有来源的物理盘径。",
}


def _image(case, port, kind):
    uri = case["images"][port]
    if not uri.startswith("data:image/png;base64,") or len(uri) > 12_000_000:
        raise ValueError("completion_image_bound")
    raw = base64.b64decode(uri.split(",", 1)[1], validate=True)
    sha = "sha256:" + hashlib.sha256(raw).hexdigest()
    matches = [a for a in case["artifacts"] if a["artifact_type_code"] == kind]
    if len(matches) != 1 or matches[0]["checksum"] != sha:
        raise ValueError("completion_mask_binding")
    a = matches[0]
    if (
        a["scope"]["case_code"] != case["case_code"]
        or case["input"]["asset_code"] not in a["derived_from"]
    ):
        raise ValueError("completion_mask_scope")
    shape, pixels = grayscale_png(raw, geometry.MAX_PIXELS)
    if list(shape) != case["input"]["dimensions"]:
        raise ValueError("completion_mask_shape")
    return shape, pixels, {"sha256": sha[7:], "bytes": len(raw)}, a["asset_code"]


def build(value):
    case = value["case_evidence"]
    if case["case_code"] != CASE or case["input"]["checksum"] != IMAGE:
        raise ValueError("completion_review_scope")
    original = base64.b64decode(case["images"]["input"].split(",", 1)[1], validate=True)
    if "sha256:" + hashlib.sha256(original).hexdigest() != IMAGE:
        raise ValueError("completion_review_image")
    original_shape, _ = grayscale_png(original, geometry.MAX_PIXELS)
    if list(original_shape) != case["input"]["dimensions"]:
        raise ValueError("completion_review_shape")
    shape, disc, disc_ref, disc_code = _image(case, "segman_disc", "ARTIFACT:SLO_SEGMAN_DISC")
    _, cup, cup_ref, cup_code = _image(case, "segman_cup", "ARTIFACT:SLO_SEGMAN_CUP")
    results = {}
    for task, measurement in (
        ("rim", geometry.rim(shape, disc, cup)),
        ("ellipse", geometry.ellipse(shape, disc)),
    ):
        inputs = {"disc_mask": disc_ref}
        if task == "rim":
            inputs["cup_mask"] = cup_ref
        record = {
            "schema_version": geometry.VERSION,
            "task": task,
            "shape_hw": list(shape),
            "inputs": inputs,
            "correlation_group": "disc_mask:sha256:" + disc_ref["sha256"],
            "coordinate_assumption": "same_original_image_grid; square_pixels; no_resize_or_warp",
            "use": "research_measurement",
            "clinical_effect": None,
            "disease_probability": None,
            "measurement": measurement,
        }
        results[task] = {
            "record": record,
            "tool_code": TOOLS[task][0],
            "version": "1.0.0",
            "implementation": IMPLEMENTATION,
            "formula": FORMULAS[task],
            "sources": SOURCES[task],
            "method": "registered_algorithm_cpu_recalculation_from_saved_masks",
            "artifact_refs": [disc_code, cup_code] if task == "rim" else [disc_code],
        }
    observations = copy.deepcopy(OBSERVATIONS)
    for item in observations:
        item.update(
            status="已有结果",
            method="assistant_direct_image_review_in_place_of_vlm",
            deployed_vlm_executed=False,
            image_checksum=IMAGE,
            case_code=CASE,
            artifact_refs=[case["input"]["asset_code"]],
            numerical_confidence=None,
        )
    # One image's visual findings are not transferable to another case by rehash.
    f = case["results"]["fovea"]["observation"]["model"]
    fovea = {
        "selected_method": "assistant_visual_review",
        "review_completed": True,
        "selected_coordinate_xy": None,
        "localization_status": "not_localizable",
        "model_candidate_xy": f.get("fovea_xy"),
        "model_candidate_use": "historical_candidate_only",
        "anatomical_orientation": None,
        "isnt": None,
    }
    conditions = value["research_criteria"]["results"]
    summary = {
        "condition_count": len(conditions),
        "condition_outcomes": dict(Counter(r["comparison"] for r in conditions)),
        "clinical_diagnosis": None,
        "calibrated_disease_probability": None,
        "same_source_policy": "same_image_or_mask_is_not_an_independent_disease_vote",
    }
    return {
        "schema": VERSION,
        "case_code": CASE,
        "input_checksum": IMAGE,
        "visual_observations": observations,
        "geometry": results,
        "fovea": fovea,
        "unavailable_inputs": [{"node": k, "reason": v} for k, v in UNAVAILABLE.items()],
        "other_missing": ["家族史未提供", "没有可配对的同患者另一眼", "眼别及解剖方向未确定"],
        "summary": summary,
    }


def _project(value, completion):
    nodes = {n["key"]: n for n in value["nodes"]}
    for key in ("FOVEA_LOCATION", "FOVEA_MAP", "TASK_FOVEA"):
        p = nodes[key]["patient_evidence"]
        p.update(
            status="已有结果",
            summary="视觉复核已完成；本图未能定位中央凹。",
            next_action="将来接入VLM时复核同图；当前继续不依赖中央凹的测量。",
            images=["input"],
            observation_source={
                "method": "assistant_direct_image_review_in_place_of_vlm",
                "image_checksum": IMAGE,
            },
        )
        p["acquisition"] = {
            "category": "visual_observation",
            "label": "已有结果",
            "required_input": "本例原始SLO图像",
            "reason": "观察已完成，未形成可采用的坐标。",
        }
    for key in UNAVAILABLE:
        p = nodes[key]["patient_evidence"]
        p.update(status="当前输入无法取证", next_action="取得对应真实检查资料后再更新。")
    for node in nodes.values():
        node["patient_evidence"]["completion_observation_ids"] = [
            r["id"] for r in completion["visual_observations"] if node["key"] in r["nodes"]
        ]
    value["observation_summary"] = dict(
        Counter(n["patient_evidence"]["status"] for n in nodes.values())
    )


def apply(value):
    from apps.glaucoma_evidence_structure import apply as organize

    result = copy.deepcopy(value)
    result.pop("display_explanations", None)
    result.pop("case_graph_artifact", None)
    result["case_completion"] = build(result)
    _project(result, result["case_completion"])
    result = organize(result)
    result["title"] = "青光眼病例证据图 · data_06231 · 视觉复核与方向测量版"
    result["checksum"] = checksum({k: v for k, v in result.items() if k != "checksum"})
    return result


def validate(value):
    if "case_completion" not in value:
        return
    expected = build(value)
    if value["case_completion"] != expected:
        raise ValueError("completion_binding_mismatch")
    projected = copy.deepcopy(value)
    _project(projected, expected)
    if value["nodes"] != projected["nodes"]:
        raise ValueError("completion_projection_mismatch")


def main():
    from apps.evidence_graph_html import validate_snapshot, write_export

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("docs/exports/青光眼证据图.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace-current", action="store_true")
    parser.add_argument("--save-case-graph", action="store_true")
    args = parser.parse_args()
    if not args.replace_current and (
        args.output.exists() or args.output.with_suffix(".json").exists()
    ):
        parser.error("output exists")
    if args.snapshot.stat().st_size > 12_000_000:
        raise ValueError("completion_snapshot_bound")
    original = json.loads(args.snapshot.read_text())
    validate_snapshot(original)
    snapshot = apply(original)
    validate_snapshot(snapshot)
    if args.save_case_graph:
        from apps.glaucoma_case_storage import read_view, store_view

        artifact = store_view(snapshot)
        snapshot = read_view(CASE, artifact["asset_code"])
        snapshot["case_graph_artifact"] = artifact
        snapshot["checksum"] = checksum({k: v for k, v in snapshot.items() if k != "checksum"})
    write_export(snapshot, args.output, replace=args.replace_current)
    print(
        json.dumps(
            {
                "html": str(args.output),
                "nodes": len(snapshot["nodes"]),
                "visual_reviews": len(snapshot["case_completion"]["visual_observations"]),
                "geometry_tasks": list(snapshot["case_completion"]["geometry"]),
                "saved_artifact": snapshot.get("case_graph_artifact", {}).get("asset_code"),
                "checksum": snapshot["checksum"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
