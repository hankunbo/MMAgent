"""Explicit working judgments for ONE frozen case, not universal medical cutoffs."""

POLICY = {
    "version": "argument-review/2.0.0",
    "weight": "observation * relation * applicability; not a probability",
    "effect": "signed manual case judgment [-1,1]; null is unassessed, zero is assessed neutral",
    "contribution": "support=w*max(e,0); refutation=w*max(-e,0)",
    "within_family": "max separately in each direction; repeated facts never add votes",
    "within_shared_input": "max + rho*(1-max)*(1-prod(1-other_families))",
    "shared_input_rho": 0.25,
    "across_blocks": "S=1-prod(1-s_block); R=1-prod(1-r_block)",
    "confidence": "0.5+0.5*(S-R), only if at least one judgment is available",
    "uncertainty": "conservative envelope: all known dimensions +/-0.10 and judgment bounds",
    "uncertainty_kind": "working_parameter_sensitivity_not_statistical_confidence_interval",
    "hierarchy": "unique descendant assessment IDs; grouping/dependency edges add no evidence",
    "anchors": {"0": "无明确方向", "0.2": "弱", "0.5": "中等", "0.8": "强"},
    "no_clinical_authorization": True,
}

# Exact inputs are part of the review, so these values CANNOT be transferred to
# another patient, changed measurements, altered source bindings or a new image.
REVIEW_BUNDLE = {
    "case_code": "CASE:ANON_85A135B787D34A23861AB252ADC26150",
    "input_checksum": "sha256:2ba00b68ec05340ae68a7b475786bff05c2153344cd9ab67ae88e58c6b42c9cd",
    "trace_checksum": "sha256:62419a412d1d8eb3d646e9a2ca2d6825584c316bb5f72d638e17eb00fe6f410f",
    "judgments": {
        "criterion:VCDR_GE_07": {
            "value": -0.15,
            "lower": -0.35,
            "upper": 0.20,
            "reason": (
                "本例VCDR 0.542，结合已保存盘沿面积占比0.721及未确认局灶切迹，"
                "相对明显杯扩大/盘沿丢失模式给予弱反向工作判断-0.15。"
                "不是由<0.7自动得到阴性；缺物理盘大小、双眼及纵向背景，"
                "合理评阅范围保留正反两侧[-0.35,0.20]。具体分值是本次助手判断，非文献效应量。"
            ),
        },
        "criterion:RIM_RADIAL_LT_01": {
            "value": -0.20,
            "lower": -0.40,
            "upper": 0.15,
            "reason": (
                "当前掩膜径向最小比例0.171，尚保留盘沿，较严重局灶盘沿损失模式弱反向。"
                "全圆二维径向结果不等同临床上下扇区检查，未确定解剖方向且无盘大小标尺；"
                "适用性仍为0.40，判断范围[-0.40,0.15]。不把0.1设为患病开关。"
            ),
        },
        "criterion:RIM_NOTCH": {
            "value": 0.0,
            "lower": -0.15,
            "upper": 0.15,
            "reason": (
                "已保存视觉记录未见明确切迹，但观察条件不足以确认其不存在；"
                "本次评阅为中性0而不是反驳。与杯盘/盘沿数值同属形态证据族。"
            ),
        },
        "criterion:AGE": {
            "value": 0.65,
            "lower": 0.45,
            "upper": 0.85,
            "reason": (
                "已提供年龄86岁，NEI来源将年龄增长列为风险背景。给予中等偏强的"
                "背景方向0.65；关系强度仍只有0.35，因此有效贡献远低于65分。"
                "这是风险背景评阅，不是86岁患病概率，也不与先验重复加入。"
            ),
        },
    },
    "model": {
        "id": "descriptive:CLASSIFICATION",
        "ratings": [0.75, 0.50, 0.65],
        "reason": (
            "既有FairCLIP输出来自同一SLO及固定二类配置，适合保留研究分类意见。"
            "采用(青光眼分数-非青光眼分数)/(两类分数之和)表达方向与差距；"
            "两类几乎并列，贡献应接近0。维度为工程工作评阅而非940例准确率，"
            "不将训练开发单例或未校准softmax解释为临床概率。"
        ),
    },
}

NOTES = [
    "三维权重评价证据可信程度；连续判读评价本例对青光眼研究命题的方向和程度。",
    "总分0—100以50为中性锚点，不是50%患病先验；完全没有可用判读时总分为空。",
    "风险/形态/模型分栏并在研究总分中显式汇总；总分不是已校准患病概率。",
    "同族取最大贡献，同图不同族只给予0.25的增量系数；该系数是可复验工作假设。",
    "父节点按唯一判读ID汇总，不反复乘组织层权重；不将计算依赖当医学支持关系。",
    "参数范围含三维±0.10及判读上下界，是敏感性包络，不是统计置信区间。",
    "反向工作判断不等于排除疾病；文献/方法无法匹配的项目保留准确缺项，不凭空赋方向。",
    "既有观察与工具产物不变；本次不重跑模型、不新看图、不改变Knowledge发布或03接受事件。",
    "移除分析只解释本评分的计算依赖，不声称识别了生物学因果效应。",
]
