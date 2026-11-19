"""First author review: numeric working assumptions, NOT measured accuracy.

Medical relations come from the bound judgment ledger, not this weight table.
Changing a judgment requires a new policy version and a new immutable graph.
"""

POLICY = {
    "version": "argument-review/1.0.0",
    "anchors": {
        "0.95": "直接提供的结构化事实，身份已绑定；不等于疾病预测准确率",
        "0.75": "有测量及执行来源，仍缺本例标注或重复性评估",
        "0.55": "部分域/协议吻合，存在明显适用性限制",
        "0.35": "间接关联、弱判别或观察条件受限",
        "missing": "输入或可比参照不存在时不赋默认分数",
    },
    "formula": (
        "weight=observation*relation*applicability; same-group max; "
        "S=1-prod(1-s); R=1-prod(1-r); conflict=min(S,R); net=S-R"
    ),
    "aggregation": "同图诊断证据共用相关组；年龄/家族史单独风险通道；组织边不加权、不重复投票",
    "absence": "not_met/not_observed不自动转为refute；第一版未配置任何排除性规则",
    "sensitivity": "各已知维度同时±0.10，截到[0,1]；工作参数敏感性区间，不是统计置信区间",
    "criterion_ratings": {
        # observation, relation, applicability; the latter two describe the
        # approved interpretation scope, not whether a condition is met.
        "VCDR_GE_07": [0.75, 0.65, 0.55],
        "RIM_RADIAL_LT_01": [0.70, 0.75, 0.40],
        "RIM_NOTCH": [0.35, 0.80, 0.45],
        "RNFL_DEFECT": [0.30, 0.80, 0.40],
        "DISC_HEMORRHAGE": [0.25, 0.65, 0.35],
        "SAME_EYE_CHANGE": [None, 0.85, None],
        "FIELD": [None, 0.85, None],
        "IOP": [None, 0.35, None],
        "AGE": [0.95, 0.35, 0.95],
        "FAMILY": [None, 0.50, None],
    },
    "criterion_reasons": {
        "VCDR_GE_07": (
            "同图SegMAN杯盘测量可复算；≥0.7来自转诊相关形态参照，"
            "原研究成像域与本例SLO不完全一致，且无物理盘大小。未达条件不排除青光眼。"
        ),
        "RIM_RADIAL_LT_01": (
            "有方向采样记录；当前全圆最小比例与文献特定扇区/临床观察并不等价，"
            "中央凹方向未确定，适用性下调。"
        ),
        "RIM_NOTCH": (
            "盘沿切迹是结构线索，但本次助手视觉复核受条带和边界影响；未见明确切迹不是确认不存在。"
        ),
        "RNFL_DEFECT": (
            "楔形缺损有文献依据；本例灰度/条带影响纤维观察，现有复核没有稳定定位该现象。"
        ),
        "DISC_HEMORRHAGE": (
            "盘缘裂片样出血有相关性且有其他解释；当前灰度图不易区别血管与出血，"
            "不能把未识别当作阴性。"
        ),
        "SAME_EYE_CHANGE": "纵向变化需同眼、可比方法及多个时点；当前仅单图。",
        "FIELD": "需可靠且可重复的视野缺损及空间对应；当前没有视野检查。",
        "IOP": "24 mmHg是所绑定NICE条款中的转诊条件，不是诊断阈值；当前没有眼压。",
        "AGE": (
            "已提供结构化年龄，可与所绑定NEI风险背景60岁比较；"
            "关系分仅表示风险背景，不能替代结构/功能证据。"
        ),
        "FAMILY": "家族史属于风险背景；当前未提供，不按无家族史处理。",
    },
}

NOTES = [
    "分数由助手按固定量表模拟LLM评阅，是可调整的研究工作值，不是训练或临床校准结果。",
    "支持/反驳均为0表示没有进入该通道的有效正负论据，不表示患病概率为0或50%。",
    "同一图像的相关医学线索先取组内最大贡献；父节点对唯一叶证据汇总，不再累乘父层。",
    "风险背景、医学形态判读和分类器原始输出分栏，不把年龄或模型softmax充当诊断证明。",
    "覆盖率只针对冻结的条件清单，不是疾病排除率；未知与未达阈值分别保留。",
    "证据移除仅做推理敏感性试验，包含技术依赖重算，不是生物学因果干预或重新运行工具。",
    "本版读取已保存的观察，不产生新的VLM看图记录，不改变03接受事件或Knowledge发布资格。",
]
