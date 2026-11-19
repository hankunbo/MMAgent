"""Source-led disease paths, distinct from computation and interpretation support.

This is a persisted, reconstructible topology projection, not a CSS filter or a
new clinical acceptance event. Original authoring nodes and observations remain
available for exact reference and historical replay.
"""

import copy
from collections import Counter

from apps.glaucoma_evidence_plan import checksum

VERSION = "evidence-led-structure/1.0.0"
POLICY = {
    "version": "source-led-topology/1.0.0",
    "main_tree": "union_of_source_defined_criterion_paths",
    "dependency_direction": "input_or_context_to_served_check",
    "medical_effect": "not_assigned",
    "counting": "each_criterion_once_not_per_ancestor_or_dependency",
    "original_topology": "preserved_authoring_history_not_active_disease_tree",
}

# Dependencies are justified by the measurement definition or a source, not by
# the mere existence of a tool. Shared dependencies are referenced, not cloned.
DEPENDENCIES = {
    "DISC_SIZE": (
        "interpretation_context",
        ["CUP_RIM_MEASURE"],
        "interpretation_modifier",
        "EGS说明同一杯盘比在大小不同的视盘上含义不同；盘大小修饰杯盘与盘沿的解释。",
        "取得带物理标尺或临床测量方法的盘径；现有像素掩膜不能确定毫米盘大小。",
    ),
    "DISC_GEOMETRY": (
        "computation_dependency",
        ["CUP_RIM_MEASURE"],
        "measurement_input",
        "同图盘径为杯盘比和径向盘沿比例提供分母；像素面积留作测量记录。",
        "核对主选分割的盘径、边界与同图坐标，不把像素面积当疾病阈值。",
    ),
    "DISC_MASK": (
        "computation_dependency",
        ["CUP_RIM_MEASURE"],
        "measurement_input",
        "杯盘比=杯垂直径/盘垂直径；盘沿比=径向盘沿宽度/同轴盘径。视盘边界是计算输入。",
        "使用同图、同一分割体系的杯和盘边界；本例主选SegMAN-B，保留SLOctolyzer对照。",
    ),
    "FIELD_REPEAT": (
        "interpretation_context",
        ["FIELD_VALUES"],
        "validity_check",
        "视野判据要求可重复的缺损，重复检测用于确认模式是否稳定。",
        "取得同眼重复检查及策略信息；仅一次或没有视野记录时保留缺项。",
    ),
    "FIELD_QC": (
        "computation_dependency",
        ["FIELD_VALUES"],
        "validity_check",
        "检查可靠性决定该次视野能否用于缺损模式判读，不单独作为疾病线索。",
        "读取视野报告的可靠性、检测方法和重复性，不从眼底影像推造。",
    ),
    "CCT": (
        "interpretation_context",
        ["IOP_VALUES"],
        "interpretation_modifier",
        "中央角膜厚度属于眼压评估的背景资料，与压平测量一起解释。",
        "取得实际角膜厚度；当前没有眼压校正公式，不计算虚构的校正眼压。",
    ),
    "CCT_VALUES": (
        "interpretation_context",
        ["IOP_VALUES"],
        "interpretation_modifier",
        "角膜厚度数值服务眼压解释，不因单列数据节点而增加一项疾病投票。",
        "需要角膜测厚或对应报告，SLO不能提供角膜厚度。",
    ),
    "IMAGE_QC": (
        "computation_dependency",
        ["CUP_RIM_MEASURE", "RNFL_REFLECTANCE", "DISC_BLEED", "TEMPORAL_CHANGE"],
        "validity_check",
        "图像可辨认性决定哪些形态与测量可以读取，是视觉取证的前提。",
        "对具体待看区域核对清晰度、遮挡和伪影，随访还需检查可比性。",
    ),
    "SLO_IMAGE": (
        "computation_dependency",
        ["CUP_RIM_MEASURE", "RNFL_REFLECTANCE", "DISC_BLEED", "TEMPORAL_CHANGE"],
        "measurement_input",
        "原SLO提供本例可观察的结构；纵向比较还需要另一时点的同眼影像。",
        "精确引用当前病例的原图与摘要，不与其他病例或不同模态配对。",
    ),
    "OUTPUT_QC": (
        "computation_dependency",
        ["CUP_RIM_MEASURE"],
        "validity_check",
        "杯、盘掩膜和测量的尺寸、包含关系及来源一致性是比例计算的有效性前提。",
        "复用原工具输出检查和同图绑定结果；质控结果本身不判断青光眼。",
    ),
}
DEFERRED = {
    "RNFL_THICKNESS": (
        "OCT厚度有结构检查依据，但还没有本例OCT和匹配设备/人群的判读参考。",
        "补实际OCT或测量报告、分区及匹配参考库，再登记明确判读条件。",
    ),
    "ANGLE": (
        "房角检查用于亚型区分；当前主树未定义亚型结论，且本例没有前节检查。",
        "补房角观察及具体亚型适用条件，再挂到相应评估目标。",
    ),
    "ANGLE_VALUES": (
        "后节SLO没有可读取的房角结果。",
        "需要房角镜或适用前节检查资料和明确的亚型判读规则。",
    ),
    "ALTERNATIVES": (
        "近视等其他解释有来源依据，但宽泛的鉴别目录不是一个可执行判据。",
        "逐项补充可观察的鉴别现象及来源，不按已有工具倒推疾病需求。",
    ),
}
AUXILIARY = {
    "VESSEL_FEATURES",
    "AV_FEATURES",
    "FOVEA_LOCATION",
    "MODEL_ASSESSMENT",
    "VESSEL_MASK",
    "ARTERY_MASK",
    "VEIN_MASK",
    "FOVEA_MAP",
    "DENSITY",
    "WIDTH",
    "TORTUOSITY",
    "FRACTAL",
    "TASK_VESSELS",
    "TASK_AVOD",
    "TASK_FOVEA",
    "TASK_DENSITY",
    "TASK_WIDTH",
    "TASK_TORTUOSITY",
    "TASK_FRACTAL",
}


def _unique_sources(sources):
    return list({checksum(s): copy.deepcopy(s) for s in sources}.values())


def _result_summary(node, group, criterion_ids, results, demand):
    """Do not reuse old authoring-tree child counts as disease completeness."""
    p = node.get("patient_evidence", {})
    status = p.get("status", "取证需求")
    if group == "main":
        selected = [r for r in results if r["criterion_id"] in criterion_ids]
        comparisons = Counter(r["comparison"] for r in selected)
        labels = {
            "met": "达到条件",
            "not_met": "未达到条件",
            "not_observed": "已观察未见所定义现象",
            "unavailable": "缺输入未检查",
        }
        summary = (
            f"关联{len(selected)}项判读条件（按条件计数）："
            + "；".join(labels.get(k, k) + f"{v}项" for k, v in comparisons.items())
            + "。"
        )
        missing = comparisons.get("unavailable", 0)
        status = (
            "当前输入无法取证"
            if missing == len(selected)
            else ("部分已有结果" if missing else "已有结果")
        )
        return summary, status
    if node["key"] == "DISC_SIZE":
        return "本例未提供物理盘径或标尺；已有像素边界供比例计算使用。", "当前输入无法取证"
    summary = p.get("summary")
    if not summary or "直接子项" in summary:
        summary = demand
    if status in {"已有观察", "已有观察·判读不确定"}:
        status = "已有结果"
    return summary, status


def build(value):
    from apps.evidence_graph_explanation import sources as bound_sources
    from apps.glaucoma_research_criteria import EGS

    c = value.get("research_criteria", {})
    if c.get("schema") != "case-research-criteria/1.3.0":
        raise ValueError("evidence_structure_requires_criteria_1_3")
    nodes = {n["key"]: n for n in value["nodes"]}
    rules = {r["id"]: r for r in c["policy"]["criteria"]}
    main = set()
    ids = {key: [] for key in nodes}
    for result in c["results"]:
        rule = rules[result["criterion_id"]]
        if not rule.get("sources") or not rule.get("condition"):
            raise ValueError("evidence_structure_source_condition_required")
        path = result["path"]
        if not path or path[0] != "ROOT" or path[-1] != rule["leaf"]:
            raise ValueError("evidence_structure_criterion_path_mismatch")
        for i, key in enumerate(path):
            if key not in nodes or nodes[key]["parent"] != (path[i - 1] if i else None):
                raise ValueError("evidence_structure_criterion_path_mismatch")
            main.add(key)
            ids[key].append(rule["id"])
    if len(rules) != len(c["results"]) or not main:
        raise ValueError("evidence_structure_incomplete_criteria")
    roles, edges, dependencies = {}, [], []
    for key, node in nodes.items():
        if key in main:
            relevant = [rules[rid] for rid in ids[key]]
            sources = _unique_sources(s for r in relevant for s in r["sources"])
            role, group, serves = "clinical_requirement", "main", []
            names = "、".join(dict.fromkeys(r["title"] for r in relevant))
            rationale = "因为来源给出了“" + names + "”的相关条件，沿本路径取得对应观察。"
            demand = "核对以下来源定义的阈值或现象：" + "；".join(
                dict.fromkeys(r["condition"] for r in relevant)
            )
            if node["parent"] is not None:
                edges.append(
                    {
                        "from": node["parent"],
                        "to": key,
                        "relation": "organizes_source_backed_criteria",
                        "criterion_ids": ids[key],
                    }
                )
        elif key in DEPENDENCIES:
            role, targets, relation, rationale, demand = DEPENDENCIES[key]
            group = "dependency"
            serves = [target for target in targets if target in main]
            if not serves:
                raise ValueError("evidence_structure_orphan_dependency")
            ids[key] = list(dict.fromkeys(i for target in serves for i in ids[target]))
            sources = bound_sources(value, node)
            if key in {"DISC_SIZE", "DISC_GEOMETRY", "DISC_MASK"}:
                sources = [EGS["onh"], *sources]
            sources = _unique_sources(sources)
            for target in serves:
                dependencies.append(
                    {
                        "from": key,
                        "to": target,
                        "relation": relation,
                        "criterion_ids": ids[target],
                        "purpose": rationale,
                        "sources": sources,
                        "basis_kind": "source_informed_context"
                        if role == "interpretation_context"
                        else "measurement_definition_or_validity_requirement",
                    }
                )
        elif key in DEFERRED:
            role, group, serves = "deferred_check", "deferred", []
            rationale, demand = DEFERRED[key]
            sources = bound_sources(value, node)
        elif key in AUXILIARY:
            role, group, serves = "exploratory_result", "auxiliary", []
            rationale = "保留已运行工具的输出；当前没有与本例输入、方法一致的来源判读条件。"
            demand = "如用于疾病证据，先补相关来源和可检验条件，再登记为主树需求。"
            if key == "MODEL_ASSESSMENT":
                rationale = "独立展示FairCLIP的模型研究结果，与来源条件的逐项判读分开。"
                demand = "读取模型原始类别分数及输入/模型来源，不把分数拆成独立医学证据投票。"
            sources = bound_sources(value, node)
        else:
            raise ValueError("evidence_structure_unclassified_node")
        roles[key] = {
            "role": role,
            "group": group,
            "criterion_ids": ids[key],
            "rationale": rationale,
            "demand": demand,
            "serves": serves,
            "sources": sources,
        }
        summary, status = _result_summary(node, group, ids[key], c["results"], demand)
        roles[key].update(result_summary=summary, result_status=status)
    result = {
        "schema": VERSION,
        "policy": POLICY,
        "policy_checksum": checksum(POLICY),
        "criteria_checksum": checksum(c),
        "main_node_keys": [key for key in nodes if key in main],
        "main_edges": edges,
        "dependency_edges": dependencies,
        "node_roles": roles,
        "counts": dict(Counter(r["group"] for r in roles.values())),
    }
    return {**result, "checksum": checksum(result)}


def apply(value):
    result = copy.deepcopy(value)
    result.pop("display_explanations", None)
    result.pop("case_graph_artifact", None)
    result["evidence_structure"] = build(result)
    result["checksum"] = checksum({k: v for k, v in result.items() if k != "checksum"})
    return result


def validate(value):
    if "evidence_structure" not in value:
        return
    if "source_associations" in value:
        from apps.glaucoma_associations import structure

        if value["evidence_structure"] != structure(value):
            raise ValueError("association_structure_binding_mismatch")
        return
    if value["evidence_structure"].get("schema") != VERSION:
        raise ValueError("unsupported_evidence_structure")
    if value["evidence_structure"] != build(value):
        raise ValueError("evidence_structure_binding_mismatch")
