"""Patient observation projection: measurements do not become diagnostic votes."""

import copy
from collections import Counter

from apps.glaucoma_acquisition import apply_acquisition
from apps.glaucoma_evidence_plan import checksum

TASK_NODES = {
    "vessels": ("VESSEL_MASK", "TASK_VESSELS"),
    "avod": ("ARTERY_MASK", "VEIN_MASK", "DISC_MASK", "AV_FEATURES", "DISC_GEOMETRY", "TASK_AVOD"),
    "fovea": ("FOVEA_MAP", "FOVEA_LOCATION", "TASK_FOVEA"),
    "density": ("DENSITY", "TASK_DENSITY"),
    "width": ("WIDTH", "TASK_WIDTH"),
    "tortuosity": ("TORTUOSITY", "TASK_TORTUOSITY"),
    "fractal": ("FRACTAL", "TASK_FRACTAL"),
}
MISSING = {
    "CUP_RIM_MEASURE": (
        "缺工具",
        "需要同图视杯掩膜",
        "接入已验证适用于SLO的视杯模型后，调用现有杯盘比和盘沿算法。",
    ),
    "RNFL_THICKNESS": (
        "缺输入",
        "需要OCT及扫描/分区信息",
        "补充RNFL/GCC扫描或有来源的结构化测量报告。",
    ),
    "RNFL_REFLECTANCE": (
        "缺工具",
        "需要SLO神经纤维层外观分析",
        "接入外观分析工具；与OCT厚度分开记录。",
    ),
    "DISC_BLEED": (
        "缺工具",
        "需要出血候选检测及定位",
        "接入适用于该模态的出血检测工具，保留形态与其他解释。",
    ),
    "TEMPORAL_CHANGE": (
        "缺输入",
        "需要同眼不同时间的影像",
        "补充随访及采集时间，再进行配准和变化测量。",
    ),
    "FIELD_VALUES": ("缺输入", "需要视野检查结果", "补充检查点位、缺损数值及检测方法。"),
    "FIELD_QC": ("缺输入", "需要视野可靠性和重复检查", "补充固视、假阳性/假阴性等原始指标。"),
    "IOP_VALUES": ("缺输入", "需要眼压测量", "补充眼压值、单位、方法、眼别和时点。"),
    "CCT_VALUES": ("缺输入", "需要中央角膜厚度测量", "补充角膜厚度、单位和测量方法。"),
    "ANGLE_VALUES": ("缺输入", "需要房角观察", "补充房角镜或相应前节检查结果。"),
    "RISK_FIELDS": (
        "缺输入",
        "本次可见输入未绑定人口学/家族史字段",
        "使用已批准可见投影补充有来源的字段，不从标签推断。",
    ),
}
IMAGE_PORT = {
    "VESSEL_MASK": "vessel_mask",
    "ARTERY_MASK": "artery_mask",
    "VEIN_MASK": "vein_mask",
    "DISC_MASK": "disc_mask",
    "DISC_GEOMETRY": "disc_mask",
    "SLO_IMAGE": "input",
}


def project(snapshot, report):
    value = copy.deepcopy(snapshot)
    value.pop("checksum", None)
    if set(report["results"]) != set(TASK_NODES):
        raise ValueError("case_observation_task_set")
    if any(a["scope"]["case_code"] != report["case_code"] for a in report["artifacts"]):
        raise ValueError("case_observation_cross_scope")
    nodes = {n["key"]: n for n in value["nodes"]}
    titles = {
        "ROOT": "青光眼评估",
        "RNFL_REFLECTANCE": "SLO神经纤维层外观",
        "VESSEL_FEATURES": "血管几何观察",
        "AV_FEATURES": "动静脉分区观察",
        "MODEL_ASSESSMENT": "模型分类结果",
        "DISC_MASK": "视盘掩膜",
        "OUTPUT_QC": "输出核验",
        "FOVEA_LOCATION": "中央凹定位",
        "TASK_VESSELS": "血管分割工具",
        "TASK_AVOD": "动静脉/视盘分割工具",
        "TASK_FOVEA": "中央凹定位工具",
        "TASK_DENSITY": "血管面积占比算法",
        "TASK_WIDTH": "像素血管宽度算法",
        "TASK_TORTUOSITY": "血管曲折度算法",
        "TASK_FRACTAL": "分形维数算法",
    }
    for node in nodes.values():
        node["title"] = titles.get(node["key"], node["title"])
        node["patient_evidence"] = {
            "status": "待测量",
            "summary": "",
            "next_action": "",
            "results": [],
            "artifact_codes": [],
            "images": [],
        }
        if node["key"] in MISSING:
            status, summary, action = MISSING[node["key"]]
            node["patient_evidence"].update(status=status, summary=summary, next_action=action)
    for task, keys in TASK_NODES.items():
        item = report["results"][task]
        record = item["observation"]
        if item["execution"]["exit_code"] != 0 or record["task"] != task:
            raise ValueError("case_observation_not_successful")
        measurements = []
        if "measurement" in record:
            m = record["measurement"]
            measurements.append(
                {
                    "name": task,
                    "value": m["value"],
                    "unit": m["unit"],
                    "missing_reason": m["unavailable_reason"],
                    "formula": item["execution"]["manifest"].get("formula_spec"),
                }
            )
        else:
            model = record["model"]
            measurements.extend(
                {"name": port + "面积", "value": count, "unit": "pixel"}
                for port, count in model["foreground_pixels"].items()
            )
            if "fovea_xy" in model:
                missing = model.get("fovea_status") != "predicted"
                measurements.append(
                    {
                        "name": "中央凹(x,y)",
                        "value": None if missing else model["fovea_xy"],
                        "unit": "pixel",
                        "missing_reason": "未定位" if missing else None,
                    }
                )
        artifacts = [a for a in report["artifacts"] if a["extensions"]["component_task"] == task]
        for key in keys:
            node = nodes[key]
            relevant = measurements
            if key in IMAGE_PORT:
                relevant = [m for m in measurements if m["name"].startswith(IMAGE_PORT[key])]
            unavailable = relevant and all(m["value"] is None for m in relevant)
            node["patient_evidence"].update(
                status="无有效测量" if unavailable else "已有结果",
                summary=("本次输出未形成有效测量" if unavailable else "同图结果已生成并入库"),
                results=relevant,
                artifact_codes=[a["asset_code"] for a in artifacts],
                tool={
                    "code": item["execution"]["call"]["tool"],
                    "version": item["execution"]["call"]["version"],
                    "input_checksum": item["execution"]["input_checksum"],
                    "exit_code": 0,
                },
                images=[IMAGE_PORT[key]] if key in IMAGE_PORT else [],
            )
    nodes["SLO_IMAGE"]["patient_evidence"].update(
        status="已有结果",
        summary=report["sample"],
        images=["input"],
        results=[{"name": "尺寸(H,W)", "value": report["input"]["dimensions"], "unit": "pixel"}],
        artifact_codes=[report["input"]["asset_code"]],
    )
    nodes["OUTPUT_QC"]["patient_evidence"].update(
        status="已有结果", summary="输出格式、尺寸、摘要和同病例来源已核验"
    )
    fovea = report["results"]["fovea"]["observation"].get("model", {})
    point = fovea.get("fovea_xy")
    if point and fovea.get("fovea_status") == "predicted":
        h, w = report["input"]["dimensions"]
        for key in ("FOVEA_LOCATION", "FOVEA_MAP"):
            nodes[key]["patient_evidence"]["images"] = ["fovea_position"]
            if point[0] in (0, w - 1) or point[1] in (0, h - 1):
                nodes[key]["patient_evidence"].update(
                    status="待核对",
                    summary="模型定位落在图像边界",
                    next_action="核对图像视野是否包含中央凹；原始坐标和响应图已保留。",
                )
    nodes["DISC_GEOMETRY"]["patient_evidence"].update(
        summary="已有视盘掩膜及像素面积", next_action="物理尺寸需补充像素标尺；本轮数值单位为像素。"
    )
    nodes["IMAGE_QC"]["patient_evidence"].update(
        status="部分结果",
        summary="格式及尺寸已核验，成像质量评分待接入",
        next_action="补充清晰度、遮挡等有版本的质量算法。",
    )
    fairclip = report["fairclip"]["outputs"][0]
    nodes["MODEL_ASSESSMENT"]["patient_evidence"].update(
        status="已有结果",
        summary="FairCLIP既有分类输出（本轮复核，未重跑）",
        results=[
            {
                "name": "非青光眼模型分数",
                "value": fairclip["output"]["normalized_model_scores"][0],
                "unit": "1",
            },
            {
                "name": "青光眼模型分数",
                "value": fairclip["output"]["normalized_model_scores"][1],
                "unit": "1",
            },
        ],
        artifact_codes=[fairclip["source"]["artifact_code"]],
        next_action="分数校准未完成；与测量证据分别展示。",
    )

    from apps.glaucoma_supplement_view import project_supplement

    project_supplement(nodes, report)

    # Organizational parents summarize availability, never SUPPORTS/REFUTES.
    def aggregate(key):
        children = [n for n in nodes.values() if n["parent"] == key]
        for child in children:
            aggregate(child["key"])
        p = nodes[key]["patient_evidence"]
        if children:
            p["children"] = [
                {"key": n["key"], "title": n["title"], "status": n["patient_evidence"]["status"]}
                for n in children
            ]
        if p["summary"]:
            return
        available = sum(
            n["patient_evidence"]["status"]
            in {"已有结果", "部分结果", "待判读", "待核对", "已有观察", "已有观察·判读不确定"}
            for n in children
        )
        p.update(
            status="待判读"
            if available == len(children) and children
            else ("部分结果" if available else "待补充"),
            summary=f"直接子项：{available}/{len(children)}项已有结果"
            if children
            else "尚未完成测量",
            next_action="查看子节点的结果和补充项。",
        )

    aggregate("ROOT")
    nodes["ROOT"]["patient_evidence"].update(
        status="待判读",
        summary="已有影像测量及模型分类，按分支继续取证",
        next_action="补齐视杯、RNFL外观、出血检测及必要检查数据，再应用逐项判读规则。",
    )
    value.update(
        schema="evidence-graph-html-snapshot/2.1.0",
        title="青光眼病例证据图 · " + report["sample"],
        view_mode="case_observations",
        case_evidence=report,
        observation_summary=dict(Counter(n["patient_evidence"]["status"] for n in nodes.values())),
    )
    apply_acquisition(value)
    return {**value, "checksum": checksum(value)}
