"""Read-only, source-led audit of the current 52-node glaucoma evidence export.

This is a review artifact, not a new patient observation or clinical graph update.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

REVIEWED_GRAPH_SHA256 = "sha256:3d77fd29c0bfc921a023e33242503425f85defcd95c77048420af9b3a631af4f"
SOURCES = {
    "A": {
        "title": "AAO POAG PPP：视盘、RNFL、鉴别与结构功能对应（PDF 24—27页）",
        "url": "https://www.aao.org/education/preferred-practice-pattern/primary-open-angle-glaucoma-ppp",
        "receipt": "knowledge_node_audit_aao_p24_27_20260918.json；"
        "截断后段补取knowledge_node_audit_aao_p26_27_20260918.json",
    },
    "N": {
        "title": "NICE NG81 Recommendations：检查方法及适用条件",
        "url": "https://www.nice.org.uk/guidance/ng81/chapter/Recommendations",
        "receipt": "原图knowledge/source_chunks中的精确已归档引用",
    },
    "E": {
        "title": "EGS：径向盘沿/盘径比例的定义与示例",
        "url": "https://eugs.org/news/rim-disc-ratio-is-a-more-sensitive-parameter-than-cup-disc-ratio-2011-06-30-tip-of-the-month",
        "receipt": "原图research_criteria中的来源；本轮联网复核",
    },
    "I1": {
        "title": "Harizman 2006：ISNT与正常/青光眼的区分",
        "url": "https://pubmed.ncbi.nlm.nih.gov/17102005/",
        "receipt": "knowledge_node_audit_isnt_2006_20260918.json",
    },
    "I2": {
        "title": "Chan 2013：HRT盘沿面积ISNT的诊断表现",
        "url": "https://pubmed.ncbi.nlm.nih.gov/24049722/",
        "receipt": "knowledge_node_audit_isnt_2013_20260918.json",
    },
    "R": {
        "title": "SLO观察RNFL：不同光谱成像的比较研究",
        "url": "https://pubmed.ncbi.nlm.nih.gov/30977322/",
        "receipt": "knowledge_node_audit_rnfl_slo_20260918.json；摘要核对，未完成同设备适用性验证",
    },
    "P": {
        "title": "盘周萎缩与RNFL缺损的空间关系研究",
        "url": "https://pubmed.ncbi.nlm.nih.gov/32908683/",
        "receipt": "knowledge_node_audit_ppa_20260918.json；"
        "摘要使用OCT分型与彩色/无赤光照片，未完成全文核验",
    },
    "G": {
        "title": "原图工具输出、来源绑定与判读定义",
        "url": "../exports/青光眼证据图.json",
        "receipt": "本次输入文件SHA256与逐节点精确引用见配套JSON",
    },
    "B": {
        "title": "本轮22主题×3来源的原始检索结果",
        "url": "../exports/knowledge_node_audit_broad_20260918.json",
        "receipt": "有效SearchRun/SearchHit已由原入口保存Knowledge；其余任务明确拒绝/失败",
    },
}

# key: disposition, finding, next_action, sources. Each current node is explicit.
FINDINGS = {
    "ROOT": (
        "汇总",
        "有10项研究判读，但不是10条独立患者证据。",
        "关联同图、同模型和同测量的共同来源，保留支持/反驳/缺项分栏；不按节点数量投票。",
        "A G",
    ),
    "STRUCTURE": (
        "补充检查",
        "杯盘和RNFL已有分支；盘内血管形态、盘周萎缩尚未细分。",
        "优先补充同图形态观察，再决定是否需要新算法。",
        "A P",
    ),
    "FUNCTION": (
        "缺输入",
        "当前没有视野报告。",
        "保留结构—功能对应关系的需求；不以缺视野否定已取得的结构结果。",
        "A N",
    ),
    "PRESSURE": (
        "缺输入",
        "SLO不提供眼压。",
        "取得真实眼压记录后才比较方法和重复值；正常眼压也不排除青光眼。",
        "A N",
    ),
    "CONTEXT": (
        "汇总",
        "有年龄背景及图像/模型结果；非全部都是疾病判据。",
        "分开风险背景、替代解释、质量和模型预测，不将它们都算病变。",
        "A G",
    ),
    "CUPDISC": (
        "补充检查",
        "已有杯盘比、径向盘沿比及切迹观察。",
        "补查上下盘沿分布、杯垂直伸长及盘内血管走行；已关联的测量不重复加权。",
        "A I1 I2",
    ),
    "RNFL": (
        "补充检查",
        "已有SLO外观观察，尚无RNFL/GCC厚度。",
        "补记局灶楔形暗带的位置/走向与弥漫对比下降，区分可见形态和实际层厚。",
        "A R",
    ),
    "HEMORRHAGE": (
        "复核已有结果",
        "已有图像观察，但灰度SLO对血液颜色信息有限。",
        "明确盘缘条状/火焰状候选及位置；看不清与未观察到分开，不能当排除证据。",
        "A G",
    ),
    "DISC_SIZE": (
        "背景依赖",
        "盘大小影响杯大小的解释，不是孤立阳性检查。",
        "保留为杯盘/盘沿判读的背景依赖，不新增无尺度的像素面积病变阈值。",
        "A E",
    ),
    "CHANGE": (
        "缺输入",
        "仅有本次单眼单时点输入。",
        "同眼历史影像到位后比较同方法配准变化。",
        "A N",
    ),
    "FIELD_PATTERN": (
        "缺输入",
        "没有视野点位图或报告。",
        "若取得报告，记录弓形、鼻侧阶梯及与盘沿/RNFL的空间对应。",
        "A N",
    ),
    "FIELD_REPEAT": (
        "缺输入",
        "没有检查可靠性或重复视野。",
        "与FIELD_PATTERN共享同一次检查，不另作独立疾病加分项。",
        "A N",
    ),
    "IOP_METHOD": (
        "缺输入",
        "没有测压方法和时点。",
        "未来需数值、单位、方法、测量时间及重复值同时绑定。",
        "N A",
    ),
    "CCT": (
        "缺输入",
        "没有角膜测厚输入。",
        "厚度用于风险/眼压解释；不从眼底亮度推算，也不凭简单公式修正真实眼压。",
        "A N",
    ),
    "ANGLE": (
        "缺输入",
        "后节SLO不能显示前房角。",
        "房角检查或前节资料到位后分型；不以单张眼底图声称开角。",
        "A N",
    ),
    "RISK": (
        "部分已有",
        "年龄有值，家族史未提供。",
        "年龄作背景，家族史保持缺项；人群统计关联不等于个体诊断。",
        "A G",
    ),
    "ALTERNATIVES": (
        "补充检查",
        "有若干血管算法，但尚未系统列盘倾斜/近视性改变、盘周萎缩等替代解释。",
        "新增形态观察候选；色泽苍白需彩色信息，不能由灰度强弱代替。",
        "A P",
    ),
    "CUP_RIM_MEASURE": (
        "复核已有结果",
        "VCDR约0.542、最薄径向盘沿比约0.171；现有比较未达到各自预设界限。",
        "保留测量，补记0.7阈值的来源模态；EGS<0.1示例的区域定义不能由未知眼别的全局最小值替代。",
        "E G",
    ),
    "RNFL_THICKNESS": (
        "缺输入",
        "无OCT像素或带分区/方法的厚度报告。",
        "当前输入无法取证；新增SLO分割工具也不能恢复真实分层厚度。",
        "A N",
    ),
    "RNFL_REFLECTANCE": (
        "复核已有结果",
        "有外观记录，但原记录包含条纹干扰及局灶边界不清。",
        "可重新观察暗带候选和干扰位置；不把not_observed编码解释成RNFL正常。",
        "A R G",
    ),
    "DISC_BLEED": (
        "复核已有结果",
        "尚无明确盘缘出血阳性观察。",
        "区分形态候选、血管交叉/阴影；灰度不足时不生成确定阴性。",
        "A G",
    ),
    "DISC_GEOMETRY": (
        "计算依赖",
        "已有视盘轮廓，缺物理尺度。",
        "供同图比值、ROI和分区；不将像素盘径转换成毫米。",
        "E G",
    ),
    "TEMPORAL_CHANGE": (
        "缺输入",
        "缺同眼不同时间且可配准的影像。",
        "不能用别的病例或数据集图片组成纵向变化。",
        "A G",
    ),
    "FIELD_VALUES": (
        "缺输入",
        "无阈值点、偏差图或检查方法。",
        "取得真实视野后再执行条件，不由SLO生成假的临床视野。",
        "A N",
    ),
    "FIELD_QC": (
        "缺输入",
        "无可靠性指标或复测。",
        "只评价对应检查的可用性，不直接支持/反驳青光眼。",
        "A N",
    ),
    "IOP_VALUES": (
        "缺输入",
        "没有已提供测量值。",
        "不把NICE转诊界限当统一诊断界限或模型预测值。",
        "N G",
    ),
    "CCT_VALUES": ("缺输入", "没有实测角膜厚度。", "不使用人群平均数补成患者值。", "A N"),
    "ANGLE_VALUES": (
        "缺输入",
        "无房角镜或前节检查。",
        "仅在真实观察到位后运行亚型相关条件。",
        "A N",
    ),
    "RISK_FIELDS": (
        "部分已有",
        "现有年龄86；家族史等未输入。",
        "只读可见投影内字段；族群/语言可用于分组评价，不自动成为疾病加分。",
        "A G",
    ),
    "IMAGE_QC": (
        "补充检查",
        "技术格式检查不能说明所有局部形态都可判读。",
        "分别记录视盘/RNFL区域覆盖、清晰度、照明和条纹，影响对应条件而非一刀切否定整图。",
        "G",
    ),
    "VESSEL_FEATURES": (
        "探索结果",
        "已有几何值，但目前未绑定本模态可复验的疾病阈值。",
        "保留探索侧栏；优先补的是盘内血管位置现象，不是增加全局血管统计量。",
        "A B G",
    ),
    "AV_FEATURES": (
        "计算依赖",
        "已分动静脉；并无通过验证的当前疾病判读公式。",
        "保留掩膜及模型来源，不把动静脉比或高血压标志直接替代青光眼证据。",
        "B G",
    ),
    "FOVEA_LOCATION": (
        "待复核定位",
        "现有峰值x=511落在512宽图的边缘，不能直接用作可靠黄斑方位。",
        "先查输出约定、裁剪变换和实际成像范围；核准前不据此推定鼻/颞或眼别。",
        "G",
    ),
    "MODEL_ASSESSMENT": (
        "已有结果",
        "FairCLIP有模型分数，临床概率未校准。",
        "保留模型/权重/输入/两类文本绑定；同张图的预测与形态证据属于相关来源。",
        "G",
    ),
    "SLO_IMAGE": (
        "已有结果",
        "单张可见SLO已绑定病例。",
        "严格区分灰度SLO、彩色眼底、OCT与OCTA，检查裁剪和方向元数据。",
        "G R",
    ),
    "VESSEL_MASK": (
        "计算依赖",
        "已有同图血管输出。",
        "可供位置候选定位；不要将分割掩膜直接叫视神经损伤。",
        "A G",
    ),
    "ARTERY_MASK": (
        "计算依赖",
        "已有动脉输出。",
        "观察分错及交叉点；后续疾病算法须另有定义和验证。",
        "G",
    ),
    "VEIN_MASK": ("计算依赖", "已有静脉输出。", "与动脉/原图共同解释，不单独计疾病证据。", "G"),
    "DISC_MASK": (
        "计算依赖",
        "存在SLOctolyzer及SegMAN方案，杯盘需保持同图同定义。",
        "SegMAN配套杯盘用于比值；其他视盘输出可做一致性复核，不平均两个定义制造新值。",
        "G E",
    ),
    "FOVEA_MAP": (
        "计算依赖",
        "原始响应图不是已经核准的中央凹坐标。",
        "查看响应峰、坐标约定和变换，再决定可否提供解剖方向。",
        "G",
    ),
    "DENSITY": (
        "探索结果",
        "11.67%是掩膜面积/整图像素。",
        "不能套用OCTA径向盘周毛细血管密度阈值；FOV和ROI须一致才可比较。",
        "B G",
    ),
    "WIDTH": (
        "探索结果",
        "约6.11px是像素算法结果。",
        "无成像尺度不能换算微米，更不能套用血管口径临床阈值。",
        "B G",
    ),
    "TORTUOSITY": (
        "探索结果",
        "约1.0797是既有分支定义的弧弦比中位数。",
        "与其他论文的曲率积分/弯曲密度不是同一指标；无匹配阈值不新增判读。",
        "B G",
    ),
    "FRACTAL": (
        "探索结果",
        "约1.5270为既有盒计数结果。",
        "广搜发现相关研究；仍需核对二值化、尺度范围、ROI和验证人群，不能立即给诊断界限。",
        "B G",
    ),
    "TASK_VESSELS": (
        "工具依赖",
        "SLOctolyzer分割已执行。",
        "复用已有输出；只有输入/版本/QC变化才需重跑。",
        "G",
    ),
    "TASK_AVOD": (
        "工具依赖",
        "动静脉/视盘模型有输出。",
        "不把同一多头模型产物当三个独立诊断来源。",
        "G",
    ),
    "TASK_FOVEA": (
        "工具依赖",
        "工具执行成功，但位置结果落边缘待复核。",
        "先复核既有产物和预后处理，不先下载新模型。",
        "G",
    ),
    "TASK_DENSITY": (
        "工具依赖",
        "算法已运行且有输出。",
        "保持当前公式和ROI来源；疾病意义另需匹配证据。",
        "G B",
    ),
    "TASK_WIDTH": (
        "工具依赖",
        "算法已运行且有输出。",
        "记录像素单位，不更名为物理血管宽度。",
        "G B",
    ),
    "TASK_TORTUOSITY": (
        "工具依赖",
        "算法已运行且有输出。",
        "保留分支筛选、端点和汇总方法。",
        "G B",
    ),
    "TASK_FRACTAL": (
        "工具依赖",
        "算法已运行且有输出。",
        "保留盒尺度与骨架/掩膜版本，不能通过改参数拟合当前病例标签。",
        "G B",
    ),
    "OUTPUT_QC": (
        "计算依赖",
        "输出格式/范围核验与医学判读不是同一项。",
        "补查中央凹落边缘、杯盘包含关系等特定产物问题；失败只影响依赖项。",
        "G",
    ),
}

PROPOSALS = [
    {
        "key": "RIM_DISTRIBUTION",
        "parent": "CUPDISC",
        "title": "上下盘沿分布与ISNT",
        "availability": "现有图像可补形态；精确定向待核验",
        "condition": "I/S/N/T盘沿次序及局灶变薄；不同研究分别用宽度与区域面积，不混用",
        "inputs": "杯盘掩膜、原图、可靠眼别/方向",
        "action": "先观察；若要新增象限测量算法，先确认公式、分区和准入条件",
        "sources": ["A", "I1", "I2"],
        "implemented": False,
    },
    {
        "key": "DISC_VESSEL_COURSE",
        "parent": "CUPDISC",
        "title": "盘内血管鼻侧移位与悬空",
        "availability": "现有SLO和血管/视盘输出可补观察",
        "condition": "盘沿退缩相关的血管位置改变，不能把正常弯曲或血管本身当阳性",
        "inputs": "同图血管、盘杯掩膜、原图；鼻侧命名还需可靠方向",
        "action": "优先人工代VLM观察候选，记录位置和可见程度；不添加新模型",
        "sources": ["A"],
        "implemented": False,
    },
    {
        "key": "PERIPAPILLARY_BACKGROUND",
        "parent": "ALTERNATIVES",
        "title": "盘周萎缩、倾斜与近视性解释",
        "availability": "单张SLO可观察部分外观",
        "condition": "盘周边界及盘形态与近视/青光眼均可能有关；beta/gamma分型不能单凭普通灰度定论",
        "inputs": "原图、视盘边界；精细分型需要相应成像/报告",
        "action": "设背景/替代解释分支，不与杯盘证据重复加分",
        "sources": ["A", "P"],
        "implemented": False,
    },
    {
        "key": "INTEREYE_ASYMMETRY",
        "parent": "CUPDISC",
        "title": "同一患者双眼杯盘/盘沿不对称",
        "availability": "当前输入无法取证",
        "condition": "同方法双眼结构差异，考虑盘大小；本轮不添加统一差值阈值",
        "inputs": "可靠同患者双眼、眼别和同定义测量",
        "action": "只建立需求；不拿任意另一张图配对",
        "sources": ["A"],
        "implemented": False,
    },
    {
        "key": "STRUCTURE_FUNCTION",
        "parent": "FUNCTION",
        "title": "结构损伤与视野区域对应",
        "availability": "当前输入无法取证",
        "condition": "同眼盘沿/RNFL异常与相应视野区域匹配；不能要求早期必有两者",
        "inputs": "同眼结构观察、可靠视野图及日期",
        "action": "用跨分支对应关系而非复制新一票阳性",
        "sources": ["A", "E"],
        "implemented": False,
    },
    {
        "key": "OTHER_OPTIC_NEUROPATHY",
        "parent": "ALTERNATIVES",
        "title": "非青光眼性视神经/视网膜解释",
        "availability": "部分形态可观察，颜色/视野条件缺输入",
        "condition": "与凹陷不相称的苍白、垂直中线相关视野或其他视网膜病变等需核验替代解释",
        "inputs": "对应彩色影像、视野/报告及病史；灰度SLO不能提供真实颜色",
        "action": "区分本图可观察现象和本图不能提供的资料",
        "sources": ["A"],
        "implemented": False,
    },
]


def digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def build(graph_bytes, search_bytes):
    graph = json.loads(graph_bytes)
    broad = json.loads(search_bytes)
    nodes = graph["nodes"]
    keys = [n["key"] for n in nodes]
    if len(keys) != len(set(keys)) or set(keys) != set(FINDINGS):
        raise ValueError("audit_node_set_changed_requires_review")
    if digest(graph_bytes) != REVIEWED_GRAPH_SHA256:
        raise ValueError("audit_graph_snapshot_changed_requires_review")
    rows = []
    for node in nodes:
        state, finding, action, sources = FINDINGS[node["key"]]
        record = node["database"]["record"]
        rows.append(
            {
                "key": node["key"],
                "title": node["title"],
                "parent": node["parent"],
                "node_ref": {k: record[k] for k in ("kind", "code", "version", "checksum")},
                "disposition": state,
                "finding": finding,
                "next_action": action,
                "sources": sources.split(),
                "patient_observation_added": False,
            }
        )
    tasks = broad["topic_results"]
    result = {
        "schema": "glaucoma-node-audit/1.0.0",
        "date": "2026-09-18",
        "graph_file_checksum": digest(graph_bytes),
        "search_file_checksum": digest(search_bytes),
        "scope": "review_only_not_new_patient_results_or_medical_rules",
        "coverage": {"reviewed": len(rows), "total": len(nodes), "unreviewed": []},
        "search_summary": {
            k: broad[k]
            for k in (
                "topic_count",
                "search_task_count",
                "provider_hit_count",
                "unique_identity_group_count",
            )
        },
        "search_statuses": dict(Counter(t["status"] for t in tasks)),
        "search_incomplete": [
            {"topic": t["topic"], "status": t["status"], "request_key": t["request_key"]}
            for t in tasks
            if t["status"] not in ("completed", "partial")
        ],
        "sources": SOURCES,
        "nodes": rows,
        "proposals": PROPOSALS,
        "new_models_installed": [],
        "new_algorithms_registered": [],
        "graph_criteria_or_patient_results_changed": False,
    }
    result["checksum"] = digest(
        json.dumps(
            result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )
    return result


def markdown(report):
    lines = [
        "# 青光眼证据图：52节点广搜复核（2026-09-18）",
        "",
        "## 结论",
        "",
        "原52个节点全部核对。建议补6类关系/观察，其中优先利用现有图像的是盘沿分布、盘内血管位置、盘周形态；双眼、视野、OCT等缺输入的项目不能靠再跑同张图补齐。",
        "",
        "本次完成文献检索和逐节点复核；未新增患者观察、模型/算法、医学阈值或改变旧HTML/病例结果。新分支在下表中是明确的待实施建议。",
        "",
        "## 检索覆盖",
        "",
        f"通过原Knowledge入口运行{report['search_summary']['topic_count']}个主题、{report['search_summary']['search_task_count']}次来源检索，取得{report['search_summary']['provider_hit_count']}条命中、{report['search_summary']['unique_identity_group_count']}组精确标识。",
        "55次返回各5条，partial表示达到数量上限；7次Crossref网络失败，4次PubMed元数据被现有校验拒绝。不是穷尽搜索，也不是231篇都已读全文或适用于本例。详情及精确SearchRun/SearchHit见配套原始JSON。",
        "",
        "额外通过已安装AAO取文工具阅读完整指南，保存第24—27页；通过PubMed取文工具核对两篇ISNT研究。未新增来源连接器。",
        "",
        "## 三处优先修正的理解",
        "",
        "1. 当前VCDR未达0.7，只表示未满足这一研究比较条件，不是排除青光眼；"
        "其来源模态和终点须随值保留。",
        "2. 径向盘沿0.171来自既有掩膜计算。EGS的<0.1是带位置解释的示例；"
        "未知眼别/定位未核准时，不能把全局最小值改称确定下颞象限测量。",
        "3. 原图中RNFL、出血的not_observed仍可能伴随图像难判读。"
        "标题可以叫“已有结果”，但内容不能由“看不清/未见”改写为“已经排除”。",
        "",
        "## 新分支建议",
        "",
        "| 建议 | 条件/现象 | 当前输入 | 先做什么 | 来源 |",
        "|---|---|---|---|---|",
    ]
    for p in report["proposals"]:
        lines.append(
            f"| {p['title']} | {p['condition']} | {p['availability']} | "
            f"{p['action']} | {', '.join(p['sources'])} |"
        )
    lines += [
        "",
        "ISNT不能设置成通用确诊开关：2006年照片研究发现关联；2013年HRT区域面积研究中正常眼也常违反次序，特异度仅15.7%。两项研究测量方法不同，不能把其统计表现直接套到SLO掩膜。[I1/I2]",
        "",
        "近期广搜还有分形维数、血管几何、OCTA及AI研究。当前算法的整图占比、像素宽度、弧弦比并不自动等于论文的局部毛细血管密度、物理口径或曲率指标；先核对方法，不增加未经证实的疾病阈值。[B]",
        "",
        "## 逐节点清单",
        "",
        "| 节点键／检查项 | 归类 | 核对结果 | 下一动作 | 依据 |",
        "|---|---|---|---|---|",
    ]
    for n in report["nodes"]:
        lines.append(
            f"| {n['key']} · {n['title']} | {n['disposition']} | {n['finding']} | "
            f"{n['next_action']} | {', '.join(n['sources'])} |"
        )
    lines += ["", "## 来源与可复查记录", ""]
    for key, s in SOURCES.items():
        lines.append(f"- {key}：[ {s['title']} ]({s['url']})。记录：{s['receipt']}。")
    lines += [
        "",
        "## 是否需要你另搭模型",
        "",
        "本轮前三项形态补查优先复用现有SLO、SegMAN杯盘和SLOctolyzer血管产物，不要求新增模型。精确ISNT/盘周分区等新算法要先确认定义再登记；双眼/OCT/视野缺输入不能用新模型假造。",
        "",
        "本报告审查现有观察文本与保存产物元数据，没有重新目视判读原患者图像，未执行GPU或重新推理。因此不能把新增观察建议视为已发现该患者有相应病变。",
        "",
        "新的研究论据需要由原归档/候选/审核流程变成医学规则；工具说明登记与文献搜索记录不等于Published医学规则。",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    for p in (args.json, args.markdown):
        if p.exists() or not p.parent.is_dir():
            parser.error("output must be new in existing directory")
    report = build(args.graph.read_bytes(), args.search.read_bytes())
    for p, body in (
        (args.json, json.dumps(report, ensure_ascii=False, indent=2) + "\n"),
        (args.markdown, markdown(report)),
    ):
        with p.open("x", encoding="utf-8") as out:
            out.write(body)
    print(json.dumps(report["coverage"], ensure_ascii=False))


if __name__ == "__main__":
    main()
