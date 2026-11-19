"""Acquisition routes for the selected FairVLMed SLO-only case view, not diagnoses."""

from collections import Counter

from apps.glaucoma_evidence_plan import checksum

SEGMAN = {
    "name": "ControlGlaucoma / SegMAN-B杯盘分割器",
    "github": "https://github.com/WANG-ZIHENG/ControlGlaucoma",
    "commit": "d51b0e259f87f42253fb92a746e0cbbd5c739813",
    "weight_repository": "https://huggingface.co/ZihengWang/ControlGlaucoma",
    "weight_revision": "04b995f3f345461241f6462f4b5848c172c67cb0",
    "weight_file": "segman_b/segman_b.pth",
    "weight_size": 622904973,
    "publisher_lfs_sha256": "66df9d0d9785cf443e5d3c31e3650c5e02d96a13674db40ed15ea44a00c5ac0d",
    "checked_on": "2026-09-16",
    "resource_check": "HTTP206, bytes0-31; remote metadata only, not full checksum verification",
    "classes": {"0": "background", "1": "rim", "2": "cup"},
    "disc_definition": "rim union cup",
    "entrypoint": "SegMAN/segmentation/tools/simple_inference.py",
    "config": "SegMAN/segmentation/local_configs/segman/base/segman_b_ade.py",
    "state": "public_checkpoint_not_installed",
    "training_required_for_published_inference": False,
    "local_inference_verified": False,
    "tool_registration": None,
}

UNAVAILABLE = {
    "RNFL_THICKNESS": ("OCT分区厚度或同病例结构化测量报告", "当前只有SLO，没有OCT或已接入厚度报告"),
    "TEMPORAL_CHANGE": ("同眼、带时点的随访影像", "本次输入只有一张图，未绑定可比较的随访"),
    "FIELD_VALUES": ("视野点位、缺损数值和方法", "当前未提供视野检查或结构化结果"),
    "FIELD_QC": ("视野可靠性指标及重复检查", "当前未提供视野检查记录"),
    "IOP_VALUES": ("眼压值、单位、方法和时点", "当前未提供眼压测量记录"),
    "CCT_VALUES": ("中央角膜厚度及测量方法", "当前未提供角膜厚度记录"),
    "ANGLE_VALUES": ("房角镜或相应前节检查", "当前未提供房角检查记录"),
}


def route(category, label, required, reason, *, action, **extra):
    return {
        "category": category,
        "label": label,
        "required_input": required,
        "reason": reason,
        "next_action": action,
        "scope": "current_selected_case_input",
        "automatic_retry": False,
        **extra,
    }


def apply_acquisition(value):
    nodes = {n["key"]: n for n in value["nodes"]}
    supplement = value["case_evidence"].get("supplement")
    gaps = {}
    for key, (required, reason) in UNAVAILABLE.items():
        gaps[key] = route(
            "unavailable_current_input",
            "当前输入无法取证",
            required,
            reason,
            action="本轮跳过；仅在补充对应同病例检查或可信报告后重新评估。",
            new_model_alone_can_resolve=False,
        )
    gaps["CUP_RIM_MEASURE"] = route(
        "tool_build_candidate",
        "可补工具·待搭建",
        "当前SLO图像→杯/盘沿/视盘mask",
        "已有原图和视盘mask，缺适用于SLO的视杯输出；已找到具体公开任务权重。",
        action="用户单独搭建SegMAN-B分割组件并验证，随后接入原杯盘比/盘沿算法。",
        model_candidate=SEGMAN,
        execution_ready=False,
    )
    for key, required in (
        ("RNFL_REFLECTANCE", "当前SLO图像的可判读纹理"),
        ("DISC_BLEED", "当前SLO图像的可判读盘周形态"),
    ):
        gaps[key] = route(
            "visual_review_pending",
            "待图像观察",
            required,
            "有原图，可以先做图像观察；观察不确定时保留具体原因。",
            action="由助手查看原图并记录定位、外观与不确定性；后续接入VLM。",
            model_candidate=None,
            execution_ready=False,
        )
    gaps["RISK_FIELDS"] = route(
        "unavailable_current_input",
        "当前输入无法取证",
        "基础人口学字段与家族史分开",
        "当前读回记录中没有人口学字段结果。",
        action="已有字段直接显示结果；未提供且无法取得的字段本轮跳过。",
        fields={
            "age": {
                "category": "unavailable_current_input",
                "label": "当前输入无法取证",
                "case_value": None,
            },
            "gender": {
                "category": "unavailable_current_input",
                "label": "当前输入无法取证",
                "case_value": None,
            },
            "race_ethnicity_language_maritalstatus": {
                "category": "evaluation_only",
                "label": "评价分组字段",
                "case_value": None,
            },
            "family_history": {
                "category": "unavailable_current_input",
                "label": "当前输入无法取证",
                "case_value": None,
            },
        },
    )
    gaps["DISC_GEOMETRY"] = route(
        "partial_observation",
        "像素结果已有／物理尺度缺失",
        "像素mask；物理单位另需标尺",
        "已有视盘mask与像素面积；当前未绑定像素到毫米的尺度。",
        action="使用已有像素结果；物理尺寸需原设备标尺。",
        fields={"pixel_area": "observed", "physical_scale": "unavailable_current_input"},
    )
    gaps["IMAGE_QC"] = route(
        "algorithm_pending",
        "格式已核验／质量算法待接入",
        "已有SLO图像",
        "格式和尺寸已核验；清晰度、遮挡等质量算法尚未接入。",
        action="复用已有算法优先；新增质量定义需单独确认，不需要另采OCT。",
        execution_ready=False,
    )
    if supplement:
        gaps["CUP_RIM_MEASURE"] = route(
            "observed",
            "已有结果",
            "同图SegMAN-B杯/盘沿/盘掩膜",
            "掩膜像素、类别、尺寸及同图绑定通过，几何算法已计算。",
            action="查看叠加图及各项测量。",
            registered_tool=supplement["registered_tool"],
        )
        gaps["RISK_FIELDS"] = route(
            "observed_fields",
            "逐字段已有结果／当前输入无法取证",
            "同图来源的基础人口学字段",
            "年龄、gender等已读取；家族史未提供。各字段分别保留结果与来源。",
            action="已提供字段直接使用；无家族史本轮跳过。",
            fields=supplement["demographics"]["fields"],
        )
        for key, review in supplement["visual_review"]["observations"].items():
            gaps[key] = route(
                "visual_observation",
                review["status"],
                "本例SLO原图",
                review["summary"],
                action=review["next_action"],
                method="assistant_visual_review",
                finding=review["finding"],
            )
    for key, acquisition in gaps.items():
        p = nodes[key]["patient_evidence"]
        p["acquisition"] = acquisition
        p["next_action"] = acquisition["next_action"]
        if key not in {"IMAGE_QC", "DISC_GEOMETRY"} and not (supplement and key == "RISK_FIELDS"):
            p.update(status=acquisition["label"], summary=acquisition["reason"])

    def propagate(key):
        node = nodes[key]
        p = node["patient_evidence"]
        children = [n for n in nodes.values() if n["parent"] == key]
        for child in children:
            propagate(child["key"])
        if children:
            p["children"] = [
                {"key": n["key"], "title": n["title"], "status": n["patient_evidence"]["status"]}
                for n in children
            ]
        if key not in gaps:
            if children:
                categories = [n["patient_evidence"]["acquisition"]["category"] for n in children]
                unavailable = all(c == "unavailable_current_input" for c in categories)
                p["acquisition"] = route(
                    "unavailable_current_input" if unavailable else "branch_summary",
                    "当前输入无法取证" if unavailable else "查看各子项取证路径",
                    "见子节点",
                    "子项均缺少当前输入之外的检查" if unavailable else "不同子项可以有不同取证路径",
                    action="本轮跳过；新增检查后再评估。"
                    if unavailable
                    else "优先推进现有输入可完成的子项。",
                    child_categories=sorted(set(categories)),
                )
                if unavailable:
                    p.update(
                        status="当前输入无法取证",
                        summary="当前未提供本分支所需检查",
                        next_action="本轮跳过；新增检查后再评估。",
                    )
                elif p["status"] == "待补充":
                    p.update(
                        status="取证路径已分类",
                        summary="具体缺项与工具路径见子节点",
                        next_action="优先推进现有输入可完成的子项。",
                    )
            else:
                p["acquisition"] = route(
                    "review_observation" if p["status"] == "待核对" else "observed",
                    "结果待核对" if p["status"] == "待核对" else "已有观察",
                    "已绑定产物",
                    p["summary"],
                    action=p["next_action"],
                )

    propagate("ROOT")
    nodes["ROOT"]["patient_evidence"]["next_action"] = (
        "利用已得到的模型、测量、人口学和视觉观察推进逐项研究判读；"
        "OCT、视野、眼压等当前无法取证项保留缺项，不作为本轮反复等待任务。"
    )
    policy = {
        "version": "fairvlmed-acquisition/1.1.0",
        "scope": "selected_case_slo_and_basic_demographics",
        "image_artifact": value["case_evidence"]["input"]["asset_code"],
        "dataset_fields_source": "packages/dataset_adapters/adapters.py:FAIR_FIELDS",
        "case_visibility_source": "case-demographics-supplement/1.0.0",
        "scope_statement": (
            "本次选定输入，无OCT/视野等及已接入检查文本；不是断言整个数据集永远无相关记录。"
        ),
        "missing_is_negative": False,
        "diagnostic_score_changed": False,
        "rules_checksum": checksum(gaps),
    }
    policy["checksum"] = checksum(policy)
    value["acquisition_policy"] = policy
    value["acquisition_requirements"] = list(gaps)
    value["acquisition_summary"] = dict(Counter(r["category"] for r in gaps.values()))
    value["observation_summary"] = dict(
        Counter(n["patient_evidence"]["status"] for n in nodes.values())
    )
