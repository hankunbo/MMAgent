"""Readable, source-linked interpretation of a graph; never diagnostic scoring."""

VERSION = "evidence-explanation/1.0.0"
NICE_URL = "https://www.nice.org.uk/guidance/ng81/chapter/Recommendations"

# Authoring summaries are explicitly separate from verbatim archived passages.
# Each tuple: authoring rationale, concrete information requested, interpretation.
TOPICS = {
    "ROOT": (
        "从结构表现、功能表现、眼压背景和其他解释四个方向组织取证；四类是本项目对资料的归纳。",
        "汇集本病例已有观察，分别记录支持线索、相反线索和尚无法取得的信息。",
        "综合各分支的具体发现；当前没有启用数值权重融合或自动诊断结论。",
    ),
    "STRUCTURE": (
        "指南要求观察视神经及眼底，因此把相关形态、结构和变化组织为一个取证分支。",
        "检查盘杯、盘沿、神经纤维层、盘出血、视盘大小及同眼变化。",
        "结合具体形态与测量背景逐项解释，不把结构分支本身当成一次阳性发现。",
    ),
    "FUNCTION": (
        "视野检查提供功能方面的信息，与结构观察互相补充。",
        "取得视野点位、缺损模式及可靠性、重复性信息。",
        "有实际视野记录后再解释功能表现；单张眼底图不提供实测视野。",
    ),
    "PRESSURE": (
        "眼压及相关测量背景值得单列，避免将压力信息与结构损伤混为一谈。",
        "取得眼压、测量方法、时点和中央角膜厚度。",
        "眼压是相关背景；高眼压不单独确诊，正常眼压也不单独排除。",
    ),
    "CONTEXT": (
        "同一种表现可能有不同解释，因此同时组织亚型、风险和其他解释的信息。",
        "核对房角、明确的人口学与家族史、近视及其他可能影响解释的情况。",
        "用背景修饰具体观察的解释，不把每个背景字段都自动变成疾病加分项。",
    ),
    "CUPDISC": (
        "杯盘比例与盘沿形态需要结合视盘大小解释，因此一起组织取证。",
        "同一图像中取得视杯、视盘、盘沿及杯盘几何测量。",
        "展示真实测量及形态；尚未配置针对本模态的支持/反驳阈值。",
    ),
    "RNFL": (
        "神经纤维层结构值得观察；OCT厚度和SLO外观是两种不同的信息。",
        "有OCT时取得带分区的厚度；只有SLO时观察可辨认的外观线索。",
        "厚度需要实际OCT或测量报告；不能把二维明暗转换成厚度。",
    ),
    "HEMORRHAGE": (
        "盘周出血可以提供相关线索，同时需要考虑非青光眼解释。",
        "观察候选出血的位置、形态、图像可辨认性和其他可能原因。",
        "分别记录发现、未发现和无法辨认；后两者不能互相替代。",
    ),
    "DISC_SIZE": (
        "视盘大小影响杯盘外观的解释，需要保留测量背景。",
        "取得视盘边界、像素面积及可获得的物理标尺。",
        "像素大小是本图测量，缺少标尺时不直接作为真实毫米大小。",
    ),
    "CHANGE": (
        "重复观察可以帮助识别变化，比较需要同眼和可比的检查方法。",
        "取得有时点的同眼随访影像或结构检查，配准后比较。",
        "单张影像不产生纵向变化结果；保留方法差异和测量底限。",
    ),
    "FIELD_PATTERN": (
        "视野评估是功能取证的具体入口。",
        "取得检测策略、点位、缺损数值及其空间分布。",
        "将实际缺损模式与来源支持的解释对应，不由眼底分类分数代填。",
    ),
    "FIELD_REPEAT": (
        "重复检测和一致的检测策略有助于解释视野结果。",
        "取得可靠性指标、原始检测方法及重复检查记录。",
        "判断已有功能结果是否可比较；可靠性检查不是独立疾病证据。",
    ),
    "IOP_METHOD": (
        "眼压值必须连同测量方法和时点保留，才能解释和比较。",
        "取得眼压值、单位、眼别、方法和测量时间。",
        "记录压力背景，不从图像推造眼压数值。",
    ),
    "CCT": (
        "角膜厚度是指南列出的检查信息，与眼压及风险背景一起保留。",
        "取得中央角膜厚度、单位及测量方法。",
        "保留实际测量；没有已绑定公式时不自动进行眼压校正。",
    ),
    "ANGLE": (
        "房角检查提供疾病类型与解释背景。",
        "取得房角镜或适当前节检查的实际观察。",
        "普通后节SLO图不能替代房角检查。",
    ),
    "RISK": (
        "风险背景值得记录，但每个字段是否参与判断需有各自依据。",
        "读取实际提供的人口学字段及明确家族史，并分别标记缺失。",
        "NICE相关条款讨论眼高压者未来视力损害风险，不是通用患病概率公式。",
    ),
    "ALTERNATIVES": (
        "相似的形态也可能来自近视等其他情况；图像质量还会影响观察。",
        "记录其他解释、图像局限，以及可由现有工具获得的辅助观察。",
        "辅助测量先保留为观察，只有具体解释有依据时才影响疾病命题。",
    ),
    "IMAGE_QC": (
        "图像是否可测量是工程取证前提，不是一条诊断依据。",
        "检查格式、尺寸、清晰度、可见区域及可能的伪影。",
        "技术检查通过表示相应格式或质量条件满足，不表示医学正常。",
    ),
    "VESSEL_FEATURES": (
        "已有血管掩膜可提供可复算的几何观察，作为探索性辅助信息。",
        "计算面积占比、像素宽度、曲折度和分形维数，并保留计算区域。",
        "当前仅展示测量；尚无本项目已绑定的青光眼数值判读规则。",
    ),
    "AV_FEATURES": (
        "动静脉分区用于区分后续测量的对象。",
        "取得同图动脉、静脉掩膜及各自测量范围。",
        "分割结果是观察基础，不直接赋予青光眼支持分数。",
    ),
    "FOVEA_LOCATION": (
        "中央凹定位为图像区域与几何关系提供参考。",
        "取得坐标、响应输出，核对定位是否在合理的可见区域。",
        "边界坐标等异常需核对；定位本身不是青光眼判定。",
    ),
    "MODEL_ASSESSMENT": (
        "固定分类模型提供另一条图像模式观察，与几何测量分开保存。",
        "保留两类模型分数、预测类别、固定模型与输入来源。",
        "模型分数按原定义展示；不冒充校准患病概率或替代其他分支观察。",
    ),
}
ALIASES = {
    "CUP_RIM_MEASURE": "CUPDISC",
    "RNFL_THICKNESS": "RNFL",
    "RNFL_REFLECTANCE": "RNFL",
    "DISC_BLEED": "HEMORRHAGE",
    "DISC_GEOMETRY": "DISC_SIZE",
    "TEMPORAL_CHANGE": "CHANGE",
    "FIELD_VALUES": "FIELD_PATTERN",
    "FIELD_QC": "FIELD_REPEAT",
    "IOP_VALUES": "IOP_METHOD",
    "CCT_VALUES": "CCT",
    "ANGLE_VALUES": "ANGLE",
    "RISK_FIELDS": "RISK",
}
TERMS = {
    "STRUCTURE": ("optic nerve", "fundus"),
    "CUPDISC": ("optic nerve",),
    "RNFL": ("optic nerve",),
    "DISC_SIZE": ("optic nerve",),
    "FUNCTION": ("visual field",),
    "FIELD_PATTERN": ("visual field",),
    "FIELD_REPEAT": ("visual field", "measurement strategy"),
    "PRESSURE": ("iop", "corneal thickness"),
    "IOP_METHOD": ("iop",),
    "CCT": ("corneal thickness",),
    "ANGLE": ("gonioscopy",),
    "CHANGE": ("optic nerve",),
}
FACTS = {
    "ROOT": "指南列出了视野、视神经与眼底、眼压、房角和中央角膜厚度等检查。",
    "STRUCTURE": "指南要求进行视神经与眼底评估，并记录视神经头影像。",
    "FUNCTION": "指南要求通过标准自动视野检查评估视功能，必要时重复检测。",
    "PRESSURE": "指南列出了眼压和中央角膜厚度测量；NEI说明眼压与青光眼风险有关。",
    "CONTEXT": "来源涉及房角检查、眼高压者风险背景，以及近视与青光眼相似表现。",
    "CUPDISC": "指南将视神经与眼底检查列入评估。",
    "RNFL": "所引指南要求记录视神经头影像，提供结构观察的上层依据。",
    "CHANGE": "指南讨论按临床需要重复视神经头评估，并在变化时保存新的影像基线。",
    "FIELD_PATTERN": "指南将标准自动视野检查列入诊断评估。",
    "FIELD_REPEAT": "指南讨论重复视野检测，并在已有缺损时使用相同的测量策略。",
    "IOP_METHOD": "指南指定了眼压测量方法。",
    "CCT": "指南将中央角膜厚度测量列入检查。",
    "ANGLE": "指南将房角镜检查列入检查。",
    "RISK": "相关来源讨论家族史等风险背景；NICE所引段落限于眼高压者未来视力损害风险。",
    "ALTERNATIVES": "Dalhousie说明近视和青光眼可能有相似表现，使解释更困难。",
}


def sources(snapshot, node):
    """Select relevant bound passages, not the whole shared diagnosis list."""
    result = []
    chunks = {c["ref"]["checksum"]: c for c in snapshot.get("source_chunks", [])}
    key = ALIASES.get(node["key"], node["key"])
    for binding in node.get("knowledge", []):
        chunk = chunks.get(binding.get("chunk_ref", {}).get("checksum"))
        if chunk:
            text = chunk["text"]
            if chunk.get("position", {}).get("kind") == "heading":
                continue
            if text.strip().endswith(":"):
                continue
            if key in TERMS and not any(t in text.casefold() for t in TERMS[key]):
                continue
            result.append(
                {
                    "title": "NICE 青光眼诊疗指南",
                    "text": text,
                    "url": NICE_URL,
                    "reference": binding,
                }
            )
        elif binding.get("page_id"):
            page = next(
                (p for p in snapshot.get("historical_pages", []) if p["id"] == binding["page_id"]),
                None,
            )
            if page:
                allowed = binding.get("paragraph_checksums", [])
                for paragraph in page.get("paragraphs", []):
                    if paragraph.get("text_checksum") in allowed:
                        result.append(
                            {
                                "title": {
                                    "nei": "NEI 青光眼介绍",
                                    "egs": "EGS 视盘观察",
                                    "dalhousie": "Dalhousie 近视与青光眼",
                                }.get(page["id"], page["id"]),
                                "text": paragraph["text"],
                                "url": page["url"],
                                "reference": binding,
                            }
                        )
    for page in snapshot.get("egs_pages", []):
        if page["id"] not in node.get("egs_sources", []):
            continue
        for chunk in page["chunks"]:
            if len(chunk["text"]) > 100 and any(
                t in chunk["text"].casefold() for t in page["binding"]["terms"]
            ):
                result.append(
                    {
                        "title": page["title"],
                        "text": chunk["text"],
                        "url": page["url"],
                        "reference": chunk["ref"],
                        "summary": page["binding"]["interpretation"],
                    }
                )
    unique = {(r["url"], r["text"]): r for r in result}
    return list(unique.values())


def explain(snapshot):
    nodes = {}
    for node in snapshot["nodes"]:
        key = ALIASES.get(node["key"], node["key"])
        rationale, demand, interpretation = TOPICS.get(
            key,
            (
                "该节点记录取证流程中的具体输入、工具或产物。",
                "取得并核对“" + node.get("title", node["key"]) + "”及其同图来源。",
                "按工具输出定义使用该结果；医学解释见上游证据需求。",
            ),
        )
        evidence = node.get("patient_evidence", {})
        criterion_refs = (
            snapshot.get("research_criteria", {})
            .get("node_summaries", {})
            .get(node["key"], {})
            .get("criterion_ids", [])
        )
        if criterion_refs:
            interpretation = (
                "下列条件沿取证路径下发到测量节点，叶节点的对照结果沿原路径返回。"
                "各级引用同一结果，不重复计分；条件是否满足与适用性分别列出。"
            )
        next_action = evidence.get("next_action", "")
        if criterion_refs and node["key"] in {"ROOT", "STRUCTURE", "CUPDISC", "CUP_RIM_MEASURE"}:
            next_action = "查看杯盘比条件对照；结合盘沿形态、视盘大小及SLO测量适用性继续判读。"
        if snapshot.get("research_criteria", {}).get("schema") in {
            "case-research-criteria/1.2.0",
            "case-research-criteria/1.3.0",
        }:
            interpretation = ""
            next_action = ""
            reviews = snapshot["research_criteria"]["visual_review"]["observations"]
            visual_code = {
                "RNFL_REFLECTANCE": "RNFL_DEFECT",
                "DISC_BLEED": "DISC_HEMORRHAGE",
                "IMAGE_QC": "IMAGE_QUALITY",
            }.get(node["key"])
            if visual_code:
                evidence = {**evidence, "summary": reviews[visual_code]["observation"]}
        citations = sources(snapshot, node)
        facts = list(dict.fromkeys(c["summary"] for c in citations if c.get("summary")))
        if citations and key in FACTS:
            facts.insert(0, FACTS[key])
        if not facts:
            facts = (
                ["相关来源的具体内容见下方原文。"]
                if citations
                else ["本节点的直接依据是工具定义、输入或执行产物；没有绑定独立医学原文。"]
            )
        structure = snapshot.get("evidence_structure", {})
        role = structure.get("node_roles", {}).get(node["key"])
        if role:
            rationale, demand = role["rationale"], role["demand"]
            # New disease paths cite the actual criterion, not merely a broad
            # diagnosis checklist inherited from the old authoring tree.
            citations = role["sources"]
            facts = list(
                dict.fromkeys(
                    c.get("fact") or c.get("summary") or c.get("text")
                    for c in citations
                    if c.get("fact") or c.get("summary") or c.get("text")
                )
            ) or [rationale]
            interpretation, next_action = "", ""
            evidence = {**evidence, "summary": role["result_summary"]}
        nodes[node["key"]] = {
            "source_facts": facts,
            "sources": citations,
            "rationale": rationale,
            "demand": demand,
            "result_summary": evidence.get("summary") or "尚无病例结果。",
            "interpretation": interpretation,
            "next_action": next_action,
        }
    return {
        "version": "evidence-explanation/1.4.0"
        if "evidence_structure" in snapshot
        else "evidence-explanation/1.3.0"
        if snapshot.get("research_criteria", {}).get("schema") == "case-research-criteria/1.3.0"
        else "evidence-explanation/1.2.0"
        if snapshot.get("research_criteria", {}).get("schema") == "case-research-criteria/1.2.0"
        else "evidence-explanation/1.1.0"
        if "research_criteria" in snapshot
        else VERSION,
        "authorship": "project_source_informed_synthesis",
        "nodes": nodes,
    }
