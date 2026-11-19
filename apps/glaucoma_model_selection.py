"""A recorded assistant choice for this case, not a universal model ranking."""

import copy

from apps.glaucoma_evidence_plan import checksum

INPUT = "sha256:2ba00b68ec05340ae68a7b475786bff05c2153344cd9ab67ae88e58c6b42c9cd"
MASKS = {
    "ARTIFACT:" + kind: "sha256:" + digest
    for kind, digest in {
        "SLO_DISC_MASK": "b2283ac0310fb279ad594cd984e8ab6ee0eb83a442b6ec8b2f9095efeea2f705",
        "SLO_SEGMAN_DISC": "fb63ee3bfcce9296873f2e0442675fe73b985a7fed93a07dd4f37c332f95b574",
        "SLO_SEGMAN_CUP": "19527c584e21f1d887e82b9fe03a7c3ca302a7ee03d18b0dfa7bdabcc73192e7",
        "SLO_SEGMAN_RIM": "70b1832e2753ed8a0ee817b9732b4349d8c5f01a3f3c5faa938eaf5eaae0b1af",
    }.items()
}


def decision(value):
    c = value["case_evidence"]
    if c["input"]["checksum"] != INPUT:
        raise ValueError("model_selection_case_not_reviewed")
    refs = {}
    for kind, digest in MASKS.items():
        matches = [a for a in c["artifacts"] if a["artifact_type_code"] == kind]
        if len(matches) != 1:
            raise ValueError("model_selection_mask_ambiguous")
        a = matches[0]
        if a["checksum"] != digest or a["scope"]["case_code"] != c["case_code"]:
            raise ValueError("model_selection_mask_binding")
        refs[kind] = {"code": a["asset_code"], "checksum": digest}
    s = c["supplement"]
    call = c["results"]["avod"]["execution"]["call"]
    if (
        s["input_checksum"] != INPUT
        or s["case_code"] != c["case_code"]
        or s["registered_tool"].get("code") != "segman_slo_disc_cup_segmenter"
        or call["tool"] != "sloctolyzer_avod_segmenter"
        or c["results"]["avod"]["execution"]["input_checksum"] != INPUT
    ):
        raise ValueError("model_selection_tool_binding")
    return {
        "schema": "case-model-choice/1.0.0",
        "case_code": c["case_code"],
        "input_checksum": INPUT,
        "reviewer": "assistant_visual_review_2026-09-17",
        "scope": "本例视盘、杯盘及盘沿几何；不改变血管工具或其他病例默认值",
        "primary": {
            "name": "SegMAN-B",
            "tool": s["registered_tool"],
            "disc_area_px": s["mask_check"]["pixel_counts"]["disc"],
        },
        "alternative": {
            "name": "SLOctolyzer视盘",
            "tool": {"code": call["tool"], "version": call["version"]},
            "disc_area_px": c["results"]["avod"]["observation"]["model"]["foreground_pixels"][
                "disc_mask"
            ],
        },
        "evidence_refs": refs,
        "observations": [
            "已查看同一原图、SegMAN叠加图及两份视盘掩膜，均定位在同一视盘区，轮廓大小不一致。",
            "SLOctolyzer轮廓更平滑且较窄；SegMAN边界较不规则，但同时提供杯、盘沿及其合成视盘。",
        ],
        "reasons": [
            "本任务需要杯、盘沿和盘的一致定义，因此主用SegMAN整套结果，避免跨模型拼接分子分母。",
            "两模型的盘面积差异保留为模型分歧，不平均成第三份掩膜。",
            "没有本例人工真值，不以外观或既有少量Dice宣称SegMAN准确率胜出。",
        ],
        "selection_basis": "task_consistency_with_case_visual_review",
        "accuracy_winner": None,
    }


def apply(value):
    result = copy.deepcopy(value)
    if "model_selection" in value:
        validate(value)
        return result
    choice = decision(value)
    nodes = {n["key"]: n for n in result["nodes"]}
    for key in ("DISC_MASK", "DISC_GEOMETRY"):
        p = nodes[key]["patient_evidence"]
        p["alternatives"] = [copy.deepcopy(p)]
        p.update(
            summary="本例主用SegMAN-B视盘；SLOctolyzer保留对照",
            results=[
                {
                    "name": "主用SegMAN视盘面积",
                    "value": choice["primary"]["disc_area_px"],
                    "unit": "pixel",
                }
            ],
            tool=choice["primary"]["tool"],
            artifact_codes=[choice["evidence_refs"]["ARTIFACT:SLO_SEGMAN_DISC"]["code"]],
            images=["segman_disc", "disc_mask"],
        )
        nodes[key]["tools"] = [choice["primary"]["tool"], choice["alternative"]["tool"]]
    result["model_selection"] = choice
    result.pop("display_explanations", None)
    result.pop("case_graph_artifact", None)
    result["checksum"] = checksum({k: v for k, v in result.items() if k != "checksum"})
    return result


def validate(value):
    if "model_selection" not in value:
        return
    choice = decision(value)
    if value["model_selection"] != choice:
        raise ValueError("model_selection_result_mismatch")
    nodes = {n["key"]: n for n in value["nodes"]}
    for key in ("DISC_MASK", "DISC_GEOMETRY"):
        p = nodes[key]["patient_evidence"]
        if (
            p["tool"] != choice["primary"]["tool"]
            or p["artifact_codes"] != [choice["evidence_refs"]["ARTIFACT:SLO_SEGMAN_DISC"]["code"]]
            or p["results"]
            != [
                {
                    "name": "主用SegMAN视盘面积",
                    "value": choice["primary"]["disc_area_px"],
                    "unit": "pixel",
                }
            ]
        ):
            raise ValueError("model_selection_projection_mismatch")
