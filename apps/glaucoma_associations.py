"""Source-backed research associations beside (not disguised as) decision rules."""

import argparse
import copy
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from apps.glaucoma_association_sources import DIRECTORY
from apps.glaucoma_association_sources import verify as verify_sources
from apps.glaucoma_evidence_bindings import dump, ref, write_new
from apps.glaucoma_evidence_plan import checksum
from db.knowledge import models as k
from packages.contracts import NodeType

VERSION = "case-source-associations/1.0.0"
STRUCTURE_VERSION = "evidence-led-structure/1.1.0"
# Literature statements are project paraphrases, not numerical patient rules.
FACTS = {
    "caliber": (
        "Blue Mountains人群随访：较窄CRAE与10年新发开角型青光眼相关；"
        "每降低1个标准差，调整OR为1.77（95%CI 1.12–2.79）。CRVE未显示独立关联。"
    ),
    "geometry": (
        "Singapore Malay横断面研究：较低动、静脉迂曲度及较小静脉分支角与青光眼相关；"
        "其中分形维数降低的结果针对眼高压，不能改写成该研究已证明青光眼分形关联。"
    ),
    "density": (
        "UK Biobank前瞻研究：基线较低的动、静脉血管骨架密度及分形维数与后续青光眼发生相关。"
        "该结果是随访风险关联，不是本例的诊断概率。"
    ),
    "null_geometry": (
        "2023年匹配研究各纳入23例NTG、POAG和对照；"
        "CRAE、CRVE、动静脉比、迂曲度和分形参数未见显著组间差异。"
        "未检出差异不等于证明没有关联。"
    ),
    "tilt_lee": "Lee等在近视POAG队列中报告倾斜视盘组进展较少；这是特定人群与倾斜定义下的进展结果。",
    "tilt_sawada": (
        "Sawada等的近视OAG双眼比较发现，较大倾斜等视盘形态与进展有关；"
        "不能与其他研究合并成统一单向权重。"
    ),
    "aao": (
        "AAO列出杯盘不对称、盘沿变薄、盘内血管鼻侧移位、环盘血管裸露及beta区盘周萎缩等观察，"
        "并要求结合结构和视野；近视倾斜视盘也属于鉴别背景。"
    ),
}


def definition(key, parent, title, source_ids, demand, **extra):
    return {
        "key": key,
        "parent": parent,
        "title": title,
        "source_ids": source_ids,
        "demand": demand,
        **extra,
    }


DEFINITIONS = [
    definition(
        "VASCULAR_ASSOC",
        "STRUCTURE",
        "视网膜血管形态关联",
        ["caliber", "geometry", "density", "null_geometry"],
        "分别检查动脉口径、动静脉迂曲度、分支角、骨架密度及分形维数。",
        relationship="organizes_associations",
        method="summary",
        dependencies=[],
    ),
    definition(
        "ARTERIOLAR_CALIBER",
        "VASCULAR_ASSOC",
        "视网膜动脉口径",
        ["caliber", "null_geometry"],
        "取得标准盘周区域的动脉口径及CRAE；保留测量区域、血管选择与尺度。",
        relationship="population_association",
        method="measurement",
        observation="WIDTH",
        dependencies=["ARTERY_MASK", "VEIN_MASK", "DISC_MASK", "WIDTH"],
        direction="较窄CRAE与较高新发OAG风险相关；并非所有队列复现。",
        gap="需补标准ROI、动静脉分离后的CRAE算法及尺度核对；全局像素宽度不是CRAE。",
    ),
    definition(
        "AV_TORTUOSITY",
        "VASCULAR_ASSOC",
        "动静脉迂曲度",
        ["geometry", "null_geometry"],
        "分别取得动脉与静脉迂曲度，固定分支筛选和计算公式。",
        relationship="population_association",
        method="measurement",
        observation="TORTUOSITY",
        dependencies=["ARTERY_MASK", "VEIN_MASK", "TORTUOSITY"],
        direction="Singapore Malay研究中较低迂曲度与青光眼相关；另有未检出差异的研究。",
        gap="已有全血管弧弦比中位数；需对齐文献公式、ROI及动静脉分组，不能直接套其分位数。",
    ),
    definition(
        "VENULAR_BRANCH_ANGLE",
        "VASCULAR_ASSOC",
        "静脉分支角",
        ["geometry"],
        "取得静脉分叉点、两子支方向及角度，记录交叉点排除规则。",
        relationship="population_association",
        method="algorithm_missing",
        dependencies=["VEIN_MASK", "VESSEL_MASK"],
        direction="较小静脉分支角与青光眼存在统计关联。",
        gap="本图已有静脉掩膜；分叉识别、交叉排除与角度汇总算法待确认实现，当前没有角度结果。",
    ),
    definition(
        "VASCULAR_SKELETON_DENSITY",
        "VASCULAR_ASSOC",
        "血管骨架密度",
        ["density"],
        "在一致ROI内分别计算动静脉骨架长度密度，并说明归一化方法。",
        relationship="population_association",
        method="measurement",
        observation="DENSITY",
        dependencies=["ARTERY_MASK", "VEIN_MASK", "DENSITY"],
        direction="较低骨架密度与后续青光眼发生相关。",
        gap="现有数值是血管面积占比；需补骨架长度/ROI密度算法，不能直接更名为VSD。",
    ),
    definition(
        "VASCULAR_FRACTAL",
        "VASCULAR_ASSOC",
        "血管分形维数关联",
        ["density", "null_geometry"],
        "取得定义、尺度区间和ROI明确的动静脉分形维数。",
        relationship="population_association",
        method="measurement",
        observation="FRACTAL",
        dependencies=["ARTERY_MASK", "VEIN_MASK", "FRACTAL"],
        direction="前瞻研究中较低分形维数与新发青光眼相关；小型匹配研究未检出组间差异。",
        gap="已有全图盒计数结果；需核对文献的骨架/掩膜、尺度、ROI与归一化，不能直接比较不同定义的数值。",
    ),
    definition(
        "DISC_VESSEL_COURSE",
        "CUPDISC",
        "盘内血管走行",
        ["aao"],
        "观察盘内血管位置及环盘血管裸露；鼻侧移位需先确定眼别与解剖方向。",
        relationship="morphological_association",
        method="visual",
        observation="vessel_course",
        dependencies=["SLO_IMAGE", "AV_FEATURES", "FOVEA_LOCATION"],
        direction="AAO将鼻侧移位和环盘血管裸露列为可能提示青光眼性损伤的形态。",
    ),
    definition(
        "PERIPAPILLARY_APPEARANCE",
        "STRUCTURE",
        "盘周萎缩外观",
        ["aao"],
        "观察盘周萎缩边界和分布；beta/gamma分型与普通亮带分开记录。",
        relationship="morphological_association",
        method="visual",
        observation="peripapillary",
        dependencies=["SLO_IMAGE", "IMAGE_QC"],
        direction="AAO将beta区盘周萎缩列为相关观察；不能把本例灰度亮带直接认定为beta区。",
    ),
    definition(
        "DISC_SHAPE_CONTEXT",
        "CONTEXT",
        "视盘椭圆形态与近视背景",
        ["tilt_lee", "tilt_sawada", "aao"],
        "保存二维椭圆形态和图像方向，结合近视资料解释。",
        relationship="interpretation_context",
        method="ellipse",
        dependencies=["DISC_MASK", "DISC_GEOMETRY"],
        direction="倾斜/近视形态会影响解释；不同队列的进展关联方向不一致。二维椭圆度不是三维倾斜角。",
    ),
    definition(
        "INTEREYE_COMPARISON",
        "CUPDISC",
        "双眼杯盘与盘沿不对称",
        ["aao"],
        "比较同一患者双眼的杯盘与盘沿，并核对盘大小、眼别和测量方法。",
        relationship="morphological_association",
        method="input_missing",
        dependencies=["CUP_RIM_MEASURE"],
        direction="AAO将不对称杯凹列为相关观察；单张单眼图不能完成双眼比较。",
        gap="当前没有可配对的同患者另一眼及可靠眼别信息。",
    ),
    definition(
        "STRUCTURE_FUNCTION_MATCH",
        "FUNCTION",
        "结构与视野对应",
        ["aao"],
        "对照盘沿/RNFL异常位置与同眼视野缺损区域，记录一致与不一致。",
        relationship="cross_modal_consistency",
        method="input_missing",
        dependencies=["RNFL_REFLECTANCE", "CUP_RIM_MEASURE", "FIELD_VALUES", "FIELD_QC"],
        direction="结构与功能需共同评价，变化可能不同步；不一致不自动排除青光眼。",
        gap="当前没有同眼视野结果、检查方法和可靠性记录。",
    ),
]
KEYS = {d["key"] for d in DEFINITIONS}


def citations(definition, bundle):
    return [
        {
            "id": name,
            "title": bundle["sources"][name]["title"],
            "url": bundle["sources"][name]["url"],
            "fact": FACTS[name],
            "reference": bundle["sources"][name]["reference"],
            "chunk_refs": [c["reference"] for c in bundle["sources"][name]["chunks"]],
            "representation": bundle["sources"][name]["representation"],
        }
        for name in definition["source_ids"]
    ]


def knowledge_records(value, bundle):
    """New immutable template and nodes; no false SUPPORTS/AND edges or rules."""
    old = next(s["record"] for s in value["records"] if s["record"]["kind"] == "template")
    # The saved predecessor stays in the extension, so validation is idempotent.
    if "source_associations" in value:
        old = value["source_associations"]["base_template"]
    template = k.parse_record_json(k.KnowledgeKind.TEMPLATE, json.dumps(old))
    concepts = [
        k.seal(
            k.Concept,
            code="CONCEPT:GLAUCOMA:" + d["key"],
            version="1.0.0",
            synthetic=template.synthetic,
            name=d["title"],
            source_refs=tuple(ref(c["reference"]) for c in citations(d, bundle)),
        )
        for d in DEFINITIONS
    ]
    definitions = [
        k.TemplateNode(
            node_code="EVIDENCE:GLAUCOMA:" + d["key"],
            node_type=NodeType.EVIDENCE,
            concept_code=c.code,
            required=False,
        )
        for d, c in zip(DEFINITIONS, concepts, strict=True)
    ]
    fields = {
        name: getattr(template, name)
        for name in type(template).model_fields
        if name
        not in {"checksum", "version", "supersedes", "nodes", "source_refs", "graph_template"}
    }
    sources = tuple(
        dict.fromkeys(
            (
                *template.source_refs,
                *(ref(bundle["sources"][key]["reference"]) for key in sorted(bundle["sources"])),
            )
        )
    )
    extended = k.seal(
        k.Template,
        version="1.6.0",
        supersedes=template.reference(),
        nodes=(*template.nodes, *definitions),
        source_refs=sources,
        graph_template=None,
        **fields,
    )
    added = []
    for d, concept, node in zip(DEFINITIONS, concepts, definitions, strict=True):
        refs = tuple(ref(c["reference"]) for c in citations(d, bundle))
        chunks = tuple(ref(c) for s in citations(d, bundle) for c in s["chunk_refs"])
        record = k.seal(
            k.Node,
            code=node.node_code,
            version="1.0.0",
            synthetic=template.synthetic,
            template_ref=extended.reference(),
            concept_ref=concept.reference(),
            definition=node,
            source_refs=refs,
            binding_refs=chunks,
        )
        added.append(record)
    # Original nodes retain original exact refs; the extension template includes
    # their definitions, not an invented clinical instantiation or upgraded run.
    requirements = [
        k.seal(
            k.EvidenceRequirement,
            code=n.code + ":REQUIREMENT",
            version="1.0.0",
            synthetic=template.synthetic,
            node_ref=n.reference(),
            artifact_type_code=n.code + ":ASSESSMENT",
            source_refs=n.source_refs,
            binding_refs=n.binding_refs,
            acceptance_rule_ref=None,
        )
        for n in added
    ]
    return template, extended, tuple([*concepts, extended, *added, *requirements])


def observed(d, value):
    base = {n["key"]: n for n in value["nodes"] if n["key"] not in KEYS}
    c = value["case_evidence"]
    result = {
        "status": "已有结果",
        "results": [],
        "images": [],
        "artifact_codes": [],
        "completion_observation_ids": [],
        "next_action": d.get("gap", ""),
        "correlation_group": c["input"]["checksum"],
        "patient_support_score": None,
    }
    method = d["method"]
    if method == "measurement":
        p = base[d["observation"]]["patient_evidence"]
        for field in ("results", "images", "artifact_codes", "tool"):
            if field in p:
                result[field] = copy.deepcopy(p[field])
        result["summary"] = "已有同图基础测量；" + d["gap"]
        result["measurement_equivalence"] = "not_established"
    elif method == "visual":
        o = next(
            o
            for o in value["case_completion"]["visual_observations"]
            if o["id"] == d["observation"]
        )
        result.update(
            summary=o["observation"],
            images=["input"],
            artifact_codes=o["artifact_refs"],
            observation_source={"method": o["method"], "image_checksum": o["image_checksum"]},
            visual_observation=copy.deepcopy(o),
            finding=o["finding"],
        )
    elif method == "ellipse":
        item = value["case_completion"]["geometry"]["ellipse"]
        m = item["record"]["measurement"]
        result.update(
            summary="已有二维视盘椭圆测量；近视资料和三维倾斜角未提供。",
            artifact_codes=item["artifact_refs"],
            images=["segman_disc"],
            results=[
                {"name": "短轴/长轴", "value": m["minor_over_major"], "unit": "1"},
                {"name": "长轴/短轴", "value": m["major_over_minor"], "unit": "1"},
                {"name": "图像长轴方向", "value": m["major_axis_angle_degrees"], "unit": "°"},
            ],
            tool={"code": item["tool_code"], "version": item["version"]},
        )
    elif method == "input_missing":
        result.update(status="当前输入无法取证", summary=d["gap"])
    elif method == "algorithm_missing":
        result.update(status="算法待补", summary=d["gap"], images=["vein_mask"])
    else:
        result.update(
            status="部分已有结果", summary="新增5项血管关联；4项有基础测量，分支角算法待补。"
        )
    return result


def build(value, bundle, *, version=None):
    from apps.glaucoma_tool_evidence import VERSION as bound_version

    version = version or value.get("source_associations", {}).get("schema", VERSION)
    if version not in {VERSION, bound_version}:
        raise ValueError("unsupported_association_version")
    verify_sources(bundle)
    case = value["case_evidence"]
    base_template, template, records = knowledge_records(value, bundle)
    nodes, edges, associations, dependencies = [], [], [], []
    for d in DEFINITIONS:
        record = next(
            r for r in records if r.kind is k.KnowledgeKind.NODE and r.code.endswith(":" + d["key"])
        )
        sources = citations(d, bundle)
        node = {
            "key": d["key"],
            "parent": d["parent"],
            "title": d["title"],
            "node_code": record.code,
            "node_type": "EVIDENCE",
            "database": {"record": dump(record), "status": "draft"},
            "knowledge": [
                {"source_ref": s["reference"], "chunk_refs": s["chunk_refs"]} for s in sources
            ],
            "patient_evidence": observed(d, value),
        }
        nodes.append(node)
        edges.append(
            {
                "id": "association-organizes:" + d["parent"] + ":" + d["key"],
                "parent": d["parent"],
                "child": d["key"],
                "relation": "organizes_source_association",
                "causal": False,
                "statement": "根据所列文献组织取证主题，不表示父项导致子项。",
                "sources": sources,
                "numeric_weight": None,
            }
        )
        for source_key in d["dependencies"]:
            dependencies.append(
                {
                    "from": source_key,
                    "to": d["key"],
                    "relation": "association_measurement_input",
                    "criterion_ids": [],
                    "purpose": d["demand"],
                    "sources": sources,
                    "basis_kind": "measurement_definition_not_disease_vote",
                }
            )
        if d["method"] != "summary":
            associations.append(
                {
                    "id": "association:" + d["key"],
                    "from": d["key"],
                    "to": "ROOT",
                    "relation": d["relationship"],
                    "direction": d["direction"],
                    "sources": sources,
                    "study_result_role": "population_or_guideline_not_patient_validation",
                    "counter_source_ids": [s for s in d["source_ids"] if s == "null_geometry"],
                    "universal_threshold": None,
                    "numeric_weight": None,
                    "patient_support_score": None,
                    "causal": False,
                    "correlation_group": case["input"]["checksum"],
                }
            )
    flows = None
    if version == bound_version:
        from apps.glaucoma_tool_evidence import projection

        flows = [projection(value, nodes, dependencies)]
    result = {
        "schema": version,
        "case_code": case["case_code"],
        "input_checksum": case["input"]["checksum"],
        "base_template": dump(base_template),
        "extension_template": dump(template.reference()),
        "sources": bundle,
        "nodes": nodes,
        "organization_edges": edges,
        "association_edges": associations,
        "dependency_edges": dependencies,
        "definition_checksum": checksum(DEFINITIONS),
        "source_interpretation_checksum": checksum(FACTS),
        "knowledge_records": [dump(r) for r in records],
        "counts": {
            "added_nodes": len(nodes),
            "organization_edges": len(edges),
            "medical_associations": len(associations),
            "input_links": len(dependencies),
        },
        "scoring": "reuse_existing_measurements; association_not_cutoff; no_duplicate_image_votes",
    }
    if flows is not None:
        result["tool_flows"] = flows
    return {**result, "checksum": checksum(result)}


def structure(value):
    from apps.glaucoma_evidence_structure import build as original

    base = copy.deepcopy(value)
    base["nodes"] = [n for n in base["nodes"] if n["key"] not in KEYS]
    base.pop("source_associations", None)
    result = original(base)
    extension = value["source_associations"]
    result["schema"] = STRUCTURE_VERSION
    result["association_checksum"] = extension["checksum"]
    result["association_edges"] = extension["association_edges"]
    for node, d in zip(extension["nodes"], DEFINITIONS, strict=True):
        key, p = node["key"], node["patient_evidence"]
        result["main_node_keys"].append(key)
        result["node_roles"][key] = {
            "role": "research_association",
            "group": "main",
            "criterion_ids": [],
            "rationale": d.get(
                "direction", "不同研究将血管形态与青光眼联系起来，因此分项取证并保留研究差异。"
            ),
            "demand": d["demand"],
            "serves": [],
            "sources": citations(d, extension["sources"]),
            "result_summary": p["summary"],
            "result_status": p["status"],
        }
    result["main_edges"] += [
        {
            "from": e["parent"],
            "to": e["child"],
            "relation": e["relation"],
            "criterion_ids": [],
            "sources": e["sources"],
        }
        for e in extension["organization_edges"]
    ]
    result["dependency_edges"] += extension["dependency_edges"]
    from apps.glaucoma_tool_evidence import STRUCTURE_VERSION as bound_structure
    from apps.glaucoma_tool_evidence import VERSION as bound_version
    from apps.glaucoma_tool_evidence import structure_projection

    if extension["schema"] == bound_version:
        result["schema"] = bound_structure
        structure_projection(result, extension)
    result["counts"] = dict(Counter(r["group"] for r in result["node_roles"].values()))
    result["checksum"] = checksum({k: v for k, v in result.items() if k != "checksum"})
    return result


def apply(value, bundle, *, version=None):
    from apps.glaucoma_case_completion import _project

    result = copy.deepcopy(value)
    result.pop("display_explanations", None)
    result.pop("case_graph_artifact", None)
    extension = build(result, bundle, version=version)
    result["source_associations"] = extension
    result["nodes"] = [n for n in result["nodes"] if n["key"] not in KEYS] + extension["nodes"]
    result["relationships"] = [
        e for e in result["relationships"] if not e["id"].startswith("association-organizes:")
    ] + extension["organization_edges"]
    _project(result, result["case_completion"])
    result["evidence_structure"] = structure(result)
    result["title"] = "青光眼病例证据图 · data_06231 · 文献关联扩展版"
    result["checksum"] = checksum({k: v for k, v in result.items() if k != "checksum"})
    return result


def validate(value):
    extension = value.get("source_associations")
    if extension is None:
        return
    expected = build(value, extension["sources"])
    if extension != expected:
        raise ValueError("association_binding_mismatch")
    added = [n for n in value["nodes"] if n["key"] in KEYS]
    if added != expected["nodes"]:
        raise ValueError("association_node_projection")
    edges = [e for e in value["relationships"] if e["id"].startswith("association-organizes:")]
    if edges != expected["organization_edges"]:
        raise ValueError("association_edge_projection")


def main():
    from apps.evidence_graph_html import validate_snapshot, write_export
    from apps.glaucoma_association_sources import retrieve
    from apps.glaucoma_case_storage import read_view, store_view
    from apps.knowledge_source_tools import repository
    from apps.patient_service_assembly import OBJECT_ROOTS, check_object_root
    from packages.object_store import LocalObjectStore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-case-graph", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("output must be new")
    value = json.loads(args.snapshot.read_text())
    validate_snapshot(value)
    repo = repository("editor")
    check_object_root(OBJECT_ROOTS["knowledge"], "knowledge")
    store = LocalObjectStore(OBJECT_ROOTS["knowledge"])
    bundle = retrieve(repo, store)
    result = apply(value, bundle)
    validate_snapshot(result)
    records = tuple(
        k.parse_record_json(k.KnowledgeKind(r["kind"]), json.dumps(r))
        for r in result["source_associations"]["knowledge_records"]
    )
    repo.put_many(
        records, idempotency_key="glaucoma-associations:" + checksum([dump(r) for r in records])
    )
    for record in records:
        saved = repo.get(record.reference())
        if saved.withdrawn or saved.record != record:
            raise ValueError("association_knowledge_readback")
    if args.save_case_graph:
        artifact = store_view(result)
        result = read_view(value["case_evidence"]["case_code"], artifact["asset_code"])
        result["case_graph_artifact"] = artifact
        result["checksum"] = checksum({k: v for k, v in result.items() if k != "checksum"})
    write_export(result, args.output)
    receipt = {
        "completed_at": datetime.now(UTC).isoformat(),
        "nodes": len(result["nodes"]),
        **result["source_associations"]["counts"],
        "knowledge_records": len(records),
        "knowledge_template": result["source_associations"]["extension_template"],
        "artifact": result.get("case_graph_artifact"),
        "html": str(args.output),
        "checksum": result["checksum"],
    }
    write_new(DIRECTORY / "registration.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
