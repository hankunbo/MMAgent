"""Six-section reading layout. It does not change stored evidence or rule results."""

VERSION = "evidence-graph-six-sections/1.0.0"

# Acquisition descriptions, not additional medical criteria or model approvals.
GROUPS = [
    (
        "ROOT",
        "原生SLO眼底图像、获准可见人口学；其他检查分别提供",
        "同一病例的原图、可见字段和各分支检查记录",
        "按各分支的实际工具与观察记录组织取证",
    ),
    (
        "STRUCTURE",
        "SLO眼底图像；OCT与随访影像为独立数据",
        "原图、同图分割掩膜；厚度和纵向变化另需对应检查",
        "分割与测量结合图像观察，分项选择方法",
    ),
    (
        "FUNCTION FIELD_PATTERN FIELD_REPEAT FIELD_VALUES FIELD_QC STRUCTURE_FUNCTION_MATCH",
        "标准自动视野检查及同眼结构记录",
        "视野点位、缺损数值、检查方法、可靠性和重复检查；对应结构需同眼绑定",
        "读取真实视野报告并与结构观察对照，不由SLO图像生成视野实测值",
    ),
    (
        "PRESSURE IOP_METHOD IOP_VALUES",
        "眼压测量记录；角膜厚度为配套测量",
        "眼压数值、单位、测量方法、眼别、时点与重复记录",
        "读取实际测压记录，保留方法和时点",
    ),
    (
        "CCT CCT_VALUES",
        "角膜厚度测量或结构化检查报告",
        "中央角膜厚度、单位、眼别及测量方法",
        "读取原检查数值；当前没有从SLO估算角膜厚度的工具",
    ),
    (
        "CONTEXT ALTERNATIVES",
        "SLO图像、可见人口学及相关检查记录",
        "图像质量、可见背景字段；近视、家族史、房角等需各自记录",
        "复用图像观察与已有几何测量，背景信息从原始字段读取",
    ),
    (
        "CUPDISC CUP_RIM_MEASURE",
        "同一原生SLO图像及其杯/盘分割掩膜",
        "同图、同尺寸的视杯和视盘边界，以及原图叠加检查",
        "使用本例已选择的分割结果计算杯盘比例和盘沿形态",
    ),
    (
        "RNFL",
        "SLO外观图像；OCT厚度需独立检查",
        "SLO原图用于外观观察；分区厚度需OCT或同病例测量报告",
        "图像观察与厚度读取分开进行",
    ),
    (
        "RNFL_THICKNESS",
        "OCT影像或结构化RNFL/GCC厚度报告",
        "分区厚度、单位、设备/方法、眼别和匹配参考数据",
        "读取或处理OCT检查；本例SLO不能提供实测厚度",
    ),
    (
        "RNFL_REFLECTANCE HEMORRHAGE DISC_BLEED IMAGE_QC "
        "DISC_VESSEL_COURSE PERIPAPILLARY_APPEARANCE",
        "原生SLO眼底图像",
        "保留原分辨率的同图盘区、盘周区域及必要局部放大",
        "读取已有助手实际看图记录；后续VLM可使用相同图像和观察定义",
    ),
    (
        "DISC_SIZE DISC_GEOMETRY DISC_SHAPE_CONTEXT",
        "SLO原图和同图视盘掩膜；物理尺寸另需标尺",
        "视盘边界、像素坐标；真实盘径需设备标尺，近视背景需屈光/眼轴资料",
        "复用已选视盘分割，计算几何量并结合背景记录",
    ),
    (
        "CHANGE TEMPORAL_CHANGE",
        "同眼、可比较的多时点影像或结构检查",
        "随访前后图像、检查日期、眼别及成像方法",
        "在可靠配对后进行配准和变化比较",
    ),
    (
        "ANGLE ANGLE_VALUES",
        "房角镜或相应前节检查",
        "房角观察、检查方法、眼别及检查时间",
        "读取实际房角记录；后节SLO图像不包含这一检查",
    ),
    (
        "RISK RISK_FIELDS",
        "获准可见的人口学和病史字段",
        "明确提供的年龄等字段；家族史需原始记录，未提供的保留缺项",
        "按字段来源读取，不从图像或隐藏标签补填",
    ),
    (
        "VESSEL_FEATURES VASCULAR_ASSOC",
        "SLO原图、血管及动静脉分割掩膜",
        "同图掩膜、测量区域、像素尺度和相应算法输出",
        "按口径、迂曲度、分叉角、密度和分形分别选择算法",
    ),
    (
        "AV_FEATURES ARTERY_MASK VEIN_MASK TASK_AVOD",
        "原生SLO图像及动静脉分区掩膜",
        "同图动脉、静脉分割结果和原图坐标对应关系",
        "使用本例保存的动静脉分割结果",
    ),
    (
        "FOVEA_LOCATION FOVEA_MAP TASK_FOVEA",
        "原生SLO眼底图像",
        "完整原图和黄斑可见区域；保留历史模型响应供对照",
        "当前采用已有视觉复核；历史模型点未作为选定坐标",
    ),
    (
        "MODEL_ASSESSMENT",
        "原生SLO图像及固定两类文本",
        "同病例原图、固定预处理、权重和类别顺序",
        "读取已保存的FairCLIP分类输出",
    ),
    (
        "SLO_IMAGE",
        "原生SLO眼底图像",
        "可见图像的尺寸、坐标、文件摘要和病例绑定",
        "复用登记图像；SLO不改称彩色眼底照片",
    ),
    (
        "VESSEL_MASK TASK_VESSELS",
        "原生SLO图像及同图血管掩膜",
        "血管分割输出与原图尺寸、坐标对齐",
        "使用已保存的血管分割结果",
    ),
    (
        "DISC_MASK",
        "原生SLO图像及同图视盘掩膜",
        "本例选定的视盘边界；其他分割结果只作对照",
        "沿用本例模型选择，不混用不同分割方案的计算分母",
    ),
    (
        "DENSITY WIDTH TORTUOSITY FRACTAL TASK_DENSITY TASK_WIDTH TASK_TORTUOSITY TASK_FRACTAL",
        "同图二值血管掩膜",
        "掩膜、测量区域和像素尺度；各算法保留自身公式与单位",
        "读取已保存的对应CPU算法结果",
    ),
    (
        "OUTPUT_QC",
        "工具输出文件及原输入引用",
        "输出格式、尺寸、摘要、病例与派生绑定",
        "读取已有输出检查记录",
    ),
    (
        "ARTERIOLAR_CALIBER",
        "SLO动脉掩膜及视盘掩膜",
        "按文献定义选取的ROI、动脉分支宽度和尺度信息",
        "已有全血管宽度算法仅提供基础值；标准CRAE计算尚需补齐",
    ),
    (
        "AV_TORTUOSITY",
        "SLO动脉/静脉掩膜和分段骨架",
        "分别标记的动静脉、分支端点、ROI及对应迂曲度定义",
        "现有弧弦比作为基础结果，文献分组方法需另行对齐",
    ),
    (
        "VENULAR_BRANCH_ANGLE",
        "同一SLO图像的静脉掩膜与视盘掩膜",
        "两个掩膜同尺寸、同坐标且来自同一图像；静脉与视盘均必需",
        "静脉骨架分叉识别、父子支选择、盘径归一化与尺度复核",
    ),
    (
        "VASCULAR_SKELETON_DENSITY",
        "SLO动静脉掩膜、骨架及ROI",
        "骨架长度、有效ROI面积，以及分组与尺度定义",
        "现有面积占比不替代骨架密度；对应算法尚需补齐",
    ),
    (
        "VASCULAR_FRACTAL",
        "SLO动静脉掩膜或其骨架",
        "与研究一致的掩膜/骨架、ROI、盒尺度及归一化方式",
        "已有盒计数结果，使用前核对文献测量定义",
    ),
    (
        "INTEREYE_COMPARISON",
        "同患者双眼SLO图像与杯/盘掩膜",
        "可靠的患者配对、左右眼标识和可比较的成像/分割方法",
        "双眼分别测量后比较；当前不把不同患者图像配成双眼",
    ),
]
PROFILES = {
    key: (modality, data, method) for keys, modality, data, method in GROUPS for key in keys.split()
}

RATIONALES = {
    "ROOT": (
        "指南与研究从视神经结构、视功能、眼压及适用背景提供青光眼相关信息，"
        "因此按这四个方向扩展取证分支，再汇集本例观察。"
    ),
    "STRUCTURE": (
        "文献将视神经头、盘沿、神经纤维层及相关血管形态与青光眼联系起来，"
        "因此分别检查这些结构，避免仅凭单一杯盘指标评价。"
    ),
    "FUNCTION": (
        "视野表现提供与结构观察互补的青光眼相关信息，因此扩展缺损模式、可靠性与重复性检查。"
    ),
    "PRESSURE": "眼压及其测量背景与青光眼风险评估有关，因此扩展测压方法、重复测量和角膜厚度背景。",
    "CONTEXT": (
        "近视等背景可能影响对青光眼相关表现的解释，"
        "因此同时核对房角、人口学与病史、其他眼病和测量伪影。"
    ),
}

ASSOCIATION_GAPS = {
    "ARTERIOLAR_CALIBER": (
        "缺少标准ROI内的CRAE、匹配尺度和参考分布；全局像素宽度不是CRAE，尚不能完成该项比较。"
    ),
    "AV_TORTUOSITY": (
        "缺少与文献一致的动静脉分组迂曲度、ROI和参考分布；现有全血管弧弦比不能直接套用研究分位数。"
    ),
    "VENULAR_BRANCH_ANGLE": (
        "缺少同模态、同算法的对照分布或已验证切点，尚不能判断本例角度是否属于关联方向上的偏小。"
    ),
    "VASCULAR_SKELETON_DENSITY": (
        "缺少分组骨架长度/ROI密度及匹配参考分布；现有血管面积占比不能作为骨架密度判读。"
    ),
    "VASCULAR_FRACTAL": (
        "缺少与文献一致的分组、ROI、尺度和参考分布，尚不能据当前全图盒计数判断疾病关联程度。"
    ),
    "DISC_VESSEL_COURSE": (
        "缺少可明确定位的异常走行及可靠鼻颞方向，当前没有形成该现象的明确支持观察。"
    ),
    "PERIPAPILLARY_APPEARANCE": (
        "缺少可明确划定的盘周萎缩边界及分型信息，当前外观不足以完成该项判读。"
    ),
    "DISC_SHAPE_CONTEXT": (
        "缺少屈光度、眼轴和可比的研究参照；已有二维椭圆度，尚不能判定近视相关改变或其疾病关联程度。"
    ),
    "INTEREYE_COMPARISON": "缺少同患者另一眼及可靠眼别信息，当前无法判读双眼不对称。",
    "STRUCTURE_FUNCTION_MATCH": (
        "缺少同眼视野结果、检查方法和可靠性记录，当前无法判读结构与功能是否对应。"
    ),
}


def build(value, explanations):
    """A display-only projection; all observed values remain in the snapshot."""
    nodes = {n["key"]: n for n in value["nodes"]}
    structure = value.get("evidence_structure", {})
    roles = structure.get("node_roles", {})
    associations = value.get("source_associations", {})
    relations = {r["from"]: r for r in associations.get("association_edges", [])}
    result = {}
    for key, node in nodes.items():
        p, e = node.get("patient_evidence", {}), explanations["nodes"][key]
        modality, data, method = PROFILES.get(
            key,
            ("本节点绑定的输入类型", "按已有取证记录提供对应输入", "使用已登记的方法或检查记录"),
        )
        selected = {key}
        pending = [key]
        while pending:
            parent = pending.pop()
            for edge in structure.get("main_edges", []):
                if edge["from"] == parent and edge["to"] not in selected:
                    selected.add(edge["to"])
                    pending.append(edge["to"])
        inputs = {d["from"] for d in structure.get("dependency_edges", []) if d["to"] in selected}
        tools = {}
        for other in nodes.values():
            if other["key"] not in selected | inputs:
                continue
            # A retired candidate stays in section 4, not the selected methods.
            if other["key"] in {"FOVEA_LOCATION", "FOVEA_MAP", "TASK_FOVEA"} and value.get(
                "case_completion"
            ):
                continue
            tool = other.get("patient_evidence", {}).get("tool")
            if tool and tool.get("code") and tool.get("version"):
                tools[(tool["code"], tool["version"])] = {
                    "code": tool["code"],
                    "version": tool["version"],
                }
        geometry = value.get("case_completion", {}).get("geometry", {})
        for task, consumers in (
            ("rim", {"CUP_RIM_MEASURE"}),
            ("ellipse", {"DISC_GEOMETRY", "DISC_SHAPE_CONTEXT"}),
        ):
            if (selected | inputs) & consumers and task in geometry:
                item = geometry[task]
                tools[(item["tool_code"], item["version"])] = {
                    "code": item["tool_code"],
                    "version": item["version"],
                }
        rationale = RATIONALES.get(key, e["rationale"])
        if key in relations:
            rationale = (
                relations[key]["direction"]
                + "因此扩展“"
                + node["title"]
                + "”检查，取得对应观察后与来源参照对照。"
            )
        result[key] = {
            "rationale": rationale,
            "modality": modality,
            "data_required": data,
            "method": method,
            "saved_tools": list(tools.values()),
        }
        if key in relations:
            branch = value.get("branch_geometry", {})
            summary = branch["summary"] if branch.get("node") == key else p.get("summary", "")
            if branch.get("node") != key:
                labels = {
                    "width": "全血管平均像素宽度",
                    "tortuosity": "全血管弧弦比",
                    "density": "血管面积占比",
                    "fractal": "全图盒计数分形维数",
                }
                measured = [
                    f"{labels.get(r['name'], r['name'])}：{r['value']} {r.get('unit', '')}"
                    for r in p.get("results", [])
                    if r.get("value") is not None
                ]
                if measured:
                    summary = "；".join(measured) + "。" + summary
            result[key]["association"] = {
                "reference": relations[key]["direction"],
                "observed": summary,
                "assessment": ASSOCIATION_GAPS[key],
            }
        role = roles.get(key, {})
        if role and role["role"] not in {"clinical_requirement", "research_association"}:
            targets = [nodes[k]["title"] for k in role.get("serves", [])]
            summary = role["result_summary"]
            result[key]["supporting"] = {
                "reference": role["demand"],
                "observed": summary,
                "assessment": (
                    "缺少“" + data + "”中相应检查资料，取得后再按对应检查参照判读。"
                    if role["result_status"] == "当前输入无法取证"
                    else "结果供“" + "、".join(targets) + "”使用，判读见这些检查。"
                    if targets
                    else p.get("next_action") or "按上述观察保留本项结果。"
                ),
            }
    return {"schema": VERSION, "nodes": result}
