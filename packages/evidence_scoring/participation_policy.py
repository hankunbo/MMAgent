"""V3 working review: fixed semantic groups, not counts of display nodes.

Importance numbers and the visual effect are explicitly heuristic. They are not
guideline weights, measured sensitivity, probabilities, or missing patient data.
"""

POLICY = {
    "version": "argument-participation/3.0.0",
    "direction_engine": "argument-review/2.0.0",
    "coverage": "sum(importance * acquired_slots / required_slots) / sum(importance)",
    "adjustment": "0.5 + coverage * (direction - 0.5)",
    "default_scope": "comprehensive_research",
    "importance_owner": "assistant_manual_research_design_not_clinical_weights",
    "unknown_direction": "never inferred from missing acquisition",
    "quality": "already represented by observation/applicability; do not multiply twice",
    "scenario": "coverage_only_completion_holds_existing_direction_fixed",
}

REVIEW_SCOPE = {
    "case_code": "CASE:ANON_85A135B787D34A23861AB252ADC26150",
    "input_checksum": "sha256:2ba00b68ec05340ae68a7b475786bff05c2153344cd9ab67ae88e58c6b42c9cd",
    "trace_checksum": "sha256:62419a412d1d8eb3d646e9a2ca2d6825584c316bb5f72d638e17eb00fe6f410f",
}

VISUAL_REVIEW = {
    "review_id": "manual-visible-rim/2026-09-20-v3",
    "observer": "assistant_direct_image_inspection_not_deployed_vlm",
    "input_checksum": REVIEW_SCOPE["input_checksum"],
    "overlay_checksum": "sha256:999d1035584951997edeac8f7aa9eaa9a2c90a2cd4de56f0c05ee34aba6c20c1",
    "scope": "原图可见盘区及同图分割叠加；未新增裁剪、模型或像素测量",
    "observed": "叠加图中的盘沿环连续；原图相应盘区部分边界可辨，未辨认出明确的大段盘沿中断。",
    "unresolved": "细微切迹被血管与明暗条带影响；掩膜连续不证明真实组织完整，未确定鼻颞方向。",
    "ratings": [0.30, 0.45, 0.35],
    "effect": -0.08,
    "lower": -0.25,
    "upper": 0.15,
    "reason": "仅对可见范围的粗大盘沿中断作弱反向研究评阅；不是确认细微切迹不存在。"
    "与杯盘掩膜同族去重，数值为本次工作选择。",
}


def group(key, title, comprehensive, image, slots, nodes, reason, action):
    return {
        "id": key,
        "title": title,
        "importance": {"comprehensive_research": comprehensive, "slo_image_research": image},
        "slots": slots,
        "nodes": nodes,
        "reason": reason,
        "next_action": action,
    }


# Each slot is an exact assessment definition; an OR of aliases is counted once.
# Their coverage says whether the REQUIRED measurement is acquired, not normal.
GROUPS = [
    group(
        "cup_rim",
        "杯盘及粗大盘沿形态",
        5.0,
        5.0,
        [["criterion:VCDR_GE_07"], ["criterion:RIM_RADIAL_LT_01"], ["visual:rim_continuity"]],
        ["CUP_RIM_MEASURE"],
        "形态测量与粗大观察同组；细微切迹单列。",
        "保留同图杯盘测量和可见范围观察。",
    ),
    group(
        "fine_rim",
        "细微局灶盘沿切迹",
        3.0,
        3.0,
        [["criterion:RIM_NOTCH"]],
        ["CUP_RIM_MEASURE"],
        "已有粗大轮廓不替代细微切迹的可定位检查。",
        "取得可辨认局部盘沿的图像或明确定位的视觉观察。",
    ),
    group(
        "rnfl_surface",
        "可定位RNFL外观",
        4.0,
        4.0,
        [["criterion:RNFL_DEFECT"]],
        ["RNFL_REFLECTANCE"],
        "条纹存在不等于已取得可定位RNFL缺损观察。",
        "补可区分纤维缺损与成像伪影的观察。",
    ),
    group(
        "rnfl_thickness",
        "分区RNFL/GCC厚度",
        5.0,
        0.0,
        [["descriptive:OCT_REFERENCE"]],
        ["RNFL_THICKNESS"],
        "综合结构评估有独立厚度域；纯SLO任务不要求OCT。",
        "需要同眼OCT测量或含方法与分区的报告；本图不能生成真实厚度。",
    ),
    group(
        "hemorrhage",
        "盘缘出血观察",
        3.0,
        2.0,
        [["criterion:DISC_HEMORRHAGE"]],
        ["DISC_BLEED"],
        "灰度暗线不能直接确认为出血或可靠排除出血。",
        "补能辨别出血性质及位置的图像/检查记录。",
    ),
    group(
        "field",
        "视功能与检查可靠性",
        5.0,
        0.0,
        [["criterion:FIELD"], ["descriptive:FIELD_RELIABILITY"]],
        ["FIELD_VALUES", "FIELD_QC", "STRUCTURE_FUNCTION_MATCH"],
        "视野结果和可靠性共同组成一组；结构功能对应不再重复计缺失。",
        "需要真实视野点位、方法及可靠性记录。",
    ),
    group(
        "pressure",
        "眼压及角膜厚度背景",
        3.0,
        0.0,
        [["criterion:IOP"], ["descriptive:CORNEA"]],
        ["IOP_VALUES", "CCT_VALUES"],
        "眼压与其解释背景成组；不从图像推测眼压。",
        "补测量眼压、方法/时间与角膜厚度资料。",
    ),
    group(
        "angle",
        "房角背景",
        2.0,
        0.0,
        [["descriptive:ANGLE"]],
        ["ANGLE_VALUES"],
        "综合任务包含亚型背景；纯SLO任务不要求房角。",
        "需要房角检查记录。",
    ),
    group(
        "longitudinal",
        "同眼纵向变化",
        2.0,
        0.0,
        [["criterion:SAME_EYE_CHANGE"]],
        ["TEMPORAL_CHANGE"],
        "单张图没有时间差，不能以单次形态替代进展。",
        "需要可靠同眼、同方法的不同时间检查。",
    ),
    group(
        "disc_context",
        "盘形态与物理大小背景",
        2.0,
        2.0,
        [["association:DISC_SHAPE_CONTEXT"], ["descriptive:DISC_SIZE_CONTEXT"]],
        ["DISC_GEOMETRY", "DISC_SHAPE_CONTEXT"],
        "像素椭圆度已测，但物理大小未提供，各占一个输入槽。",
        "补真实标尺或盘径检查；二维椭圆度不替代三维倾斜或近视度数。",
    ),
    group(
        "age",
        "年龄背景",
        1.0,
        1.0,
        [["criterion:AGE"]],
        ["RISK_FIELDS"],
        "风险背景与形态证据分开；年龄不再次作为疾病先验。",
        "复用已提供年龄。",
    ),
    group(
        "family",
        "明确家族史",
        1.0,
        0.0,
        [["criterion:FAMILY"]],
        ["RISK_FIELDS"],
        "其他人口学字段不替代明确家族史。",
        "需要明确家族史，缺失不是阴性家族史。",
    ),
    group(
        "vascular",
        "定义匹配的血管关联测量",
        1.0,
        1.0,
        [
            ["association:ARTERIOLAR_CALIBER"],
            ["association:AV_TORTUOSITY"],
            ["association:VENULAR_BRANCH_ANGLE"],
            ["association:VASCULAR_SKELETON_DENSITY"],
            ["association:VASCULAR_FRACTAL"],
        ],
        [
            "ARTERIOLAR_CALIBER",
            "AV_TORTUOSITY",
            "VENULAR_BRANCH_ANGLE",
            "VASCULAR_SKELETON_DENSITY",
            "VASCULAR_FRACTAL",
        ],
        "探索关联低重要性；通用面积密度/宽度/混合血管分形不冒充文献同定义测量。",
        "补匹配ROI、动静脉对象与方法的测量和个体参照。",
    ),
    group(
        "intereye",
        "双眼结构比较",
        2.0,
        0.0,
        [["association:INTEREYE_COMPARISON"]],
        ["INTEREYE_COMPARISON"],
        "当前单眼输入不能构成双眼不对称。",
        "补可靠同患者、配对眼别的图像。",
    ),
    group(
        "classifier",
        "固定分类模型输出",
        1.0,
        1.0,
        [["descriptive:CLASSIFICATION"]],
        ["MODEL_ASSESSMENT"],
        "输出可用不等于疾病判断充分；作为独立模型通道保留。",
        "使用既有输出，不重复运行模型。",
    ),
]

# The review binds to exact content above, not to the fact that a float exists.
ACQUIRED = {
    "criterion:VCDR_GE_07",
    "criterion:RIM_RADIAL_LT_01",
    "visual:rim_continuity",
    "association:DISC_SHAPE_CONTEXT",
    "criterion:AGE",
    "association:VENULAR_BRANCH_ANGLE",
    "descriptive:CLASSIFICATION",
}

TECH_RESOURCES = {
    "SLO_IMAGE": ["image"],
    "DISC_MASK": ["segman_disc"],
    "VESSEL_MASK": ["slo_vessel"],
    "ARTERY_MASK": ["slo_artery"],
    "VEIN_MASK": ["slo_vein"],
    "FOVEA_MAP": ["fovea"],
    "TASK_VESSELS": ["image", "slo_vessel"],
    "TASK_AVOD": ["image", "slo_artery", "slo_vein"],
    "TASK_FOVEA": ["image", "fovea"],
    "TASK_DENSITY": ["image", "slo_vessel"],
    "TASK_WIDTH": ["image", "slo_vessel"],
    "TASK_TORTUOSITY": ["image", "slo_vessel"],
    "TASK_FRACTAL": ["image", "slo_vessel"],
    "OUTPUT_QC": ["segman_disc", "segman_cup"],
}

QUALITY = {
    "descriptive:IMAGE_QUALITY": (
        0.40,
        "原图有条带、明暗不均及血管遮挡；可见粗轮廓，细微征象受限。已体现在相应O/A，不再乘一次。",
    ),
    "visual:quality": (0.40, "与IMAGE_QUALITY同一图像质量事实，不作为第二次惩罚。"),
    "descriptive:FOVEA": (0.15, "有模型响应但边界点未经可靠定位，当前不用于鼻颞轴。"),
    "visual:fovea": (0.15, "原图未能可靠定位中央凹；不以猜测坐标参与方向测量。"),
}

NOTES = [
    "疾病倾向D复用v2唯一事实/同族/同图去重；目录层级不增加论据。",
    "每项评阅明确疾病方向、质量或充分性作用；没有方向依据不强制非零。",
    "新看图记录只增加粗大盘沿连续性的弱反向评阅；细微切迹未确定，与掩膜同族不重复加分。",
    "Q为已冻结研究任务的加权输入覆盖，不是敏感度、准确率或临床检查标准。重要性由助手设定。",
    "综合任务与SLO任务分别报告Q；选用哪个范围必须明示，不能为提高总分静默切换。",
    "调整分=50+Q*(D-50)，缺失向中性收缩；Q只扣覆盖，质量已在O/A中处理。",
    "缺失情景只计算补齐覆盖并保持D不变时的调整分，不虚构阳性/阴性结果；疾病方向变化尚不能量化。",
    "参数范围只涵盖已评阅判断，不涵盖未知检查结果，不称完整临床置信区间。",
    "病例/源对象/原判读链精确绑定，新增评分不能改变Knowledge发布或正式医学图接受状态。",
]
