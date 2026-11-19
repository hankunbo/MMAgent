"""Bind saved tool facts into the association graph; no inference or new scoring."""

import argparse
import copy
import json
from pathlib import Path

from apps.glaucoma_branch_measurement import NODE
from apps.glaucoma_branch_measurement import validate as validate_measurement

VERSION = "case-source-associations/1.1.0"
STRUCTURE_VERSION = "evidence-led-structure/1.2.0"
PORT_NODES = {"vein_mask": "VEIN_MASK", "disc_mask": "DISC_MASK"}


def projection(value, nodes, dependencies):
    """Rebuild every displayed binding from the exact saved measurement inputs."""
    from apps.vessel_branch_tool import FORMULA

    if "branch_geometry" not in value:
        raise ValueError("association_measurement_required")
    validate_measurement(value)
    branch = value["branch_geometry"]
    record, artifact = branch["record"], branch["artifact"]
    case = value["case_evidence"]
    original = {n["key"]: n for n in value["nodes"]}
    projected = {n["key"]: n for n in nodes}
    target = projected[NODE]["patient_evidence"]
    inputs = []
    for port, key in PORT_NODES.items():
        binding = record["input_bindings"][port]
        source = next(a for a in case["artifacts"] if a["checksum"] == binding["sha256"])
        evidence = original[key]["patient_evidence"]
        if source["asset_code"] not in evidence["artifact_codes"]:
            raise ValueError("association_selected_mask_mismatch")
        producer = evidence.get("tool")
        if not producer or not producer.get("code") or not producer.get("version"):
            raise ValueError("association_mask_producer_required")
        inputs.append(
            {
                "port": port,
                "node": key,
                "required": True,
                "artifact_code": source["asset_code"],
                "checksum": source["checksum"],
                "artifact_type_code": source["artifact_type_code"],
                "dimensions_hw": source["dimensions"],
                "producer": copy.deepcopy(producer),
            }
        )
    measured = [j for j in record["junctions"] if j["status"] == "measured"]
    target.update(
        status="已有结果",
        summary=branch["summary"],
        results=[
            {"name": j["junction_id"] + " 静脉分叉角", "value": j["angle_degrees"], "unit": "°"}
            for j in measured
        ]
        + [
            {"name": "静脉分叉角中位数", "value": record["median_degrees"], "unit": "°"},
            {"name": "有效分叉数量", "value": record["count"], "unit": "个"},
        ],
        artifact_codes=[artifact["asset_code"]],
        tool=copy.deepcopy(branch["tool"]),
        images=["vein_mask", "segman_disc"],
        next_action="",
        measurement_equivalence="population_study_protocol_not_established",
    )
    parent = projected["VASCULAR_ASSOC"]["patient_evidence"]
    parent.update(
        summary="5项血管关联：静脉分叉角已完成；其余4项有基础测量，仍需匹配文献测量定义。",
        artifact_codes=[artifact["asset_code"]],
        result_node_refs=[NODE],
    )
    # Both masks are required. A whole-vessel mask cannot stand in for the disc.
    kept = [d for d in dependencies if d["to"] != NODE]
    source_edges = [d for d in dependencies if d["to"] == NODE]
    for item in inputs:
        kept.append(
            {
                **copy.deepcopy(source_edges[0]),
                "from": item["node"],
                "input_port": item["port"],
                "input_artifact_code": item["artifact_code"],
                "input_checksum": item["checksum"],
                "input_combination": "AND",
                "purpose": "同图静脉掩膜用于分叉骨架；选定视盘掩膜用于父干根定位和盘径尺度。",
            }
        )
    dependencies[:] = kept
    return {
        "schema": "case-tool-evidence-flow/1.0.0",
        "target_node": NODE,
        "path": ["ROOT", "STRUCTURE", "VASCULAR_ASSOC", NODE],
        "case_code": case["case_code"],
        "image_ref": {k: case["input"][k] for k in ("asset_code", "checksum")},
        "required_inputs": inputs,
        "input_combination": "AND",
        "tool": copy.deepcopy(branch["tool"]),
        "formula": copy.deepcopy(FORMULA),
        "measurement_checksum": record["checksum"],
        "policy_checksum": record["policy_checksum"],
        "output_ref": {k: artifact[k] for k in ("asset_code", "checksum", "artifact_type_code")},
        "result_summary": branch["summary"],
        "interpretation": "记录与青光眼有关联的静脉分叉角；本例尚不作增减疾病支持度的数值判定。",
        "comparison_requirement": (
            "需要同模态、同算法的对照分布或已验证切点，才能判断本例是否偏小。"
        ),
        "disease_threshold": None,
        "patient_support_score": None,
        "aggregation": "reference_same_artifact_once; not_independent_votes",
        "correlation_group": case["input"]["checksum"],
    }


def structure_projection(result, extension):
    """Move actually consumed masks into the dependency view and link consumers."""
    flow = extension["tool_flows"][0]
    for edge in extension["dependency_edges"]:
        role = result["node_roles"][edge["from"]]
        if role["group"] == "auxiliary":
            role["role"], role["group"] = "computation_dependency", "dependency"
            role["rationale"] = "为有文献关联的检查提供输入或基础测量，具体比较方法见对应检查。"
            role["demand"] = "复用同图已保存的结果；核对计算区域、定义和单位是否符合对应研究。"
        role["serves"] = list(dict.fromkeys([*role["serves"], edge["to"]]))
    for item in flow["required_inputs"]:
        role = result["node_roles"][item["node"]]
        role["role"], role["group"] = "computation_dependency", "dependency"
        role["rationale"] = (
            "静脉掩膜区分测量对象，提供分叉骨架；它是静脉分叉角的实际计算输入。"
            if item["port"] == "vein_mask"
            else role["rationale"] + " 本图同时将它绑定为静脉分叉角的实际计算输入。"
        )
        role["demand"] = (
            "使用同图静脉分割；与选定视盘边界一起计算分叉角。"
            if item["port"] == "vein_mask"
            else role["demand"] + " 同一视盘还用于静脉父干根定位和尺度归一化。"
        )
    for key in flow["path"][:-1]:
        result["node_roles"][key]["tool_result_refs"] = [flow["output_ref"]]
        result["node_roles"][key]["tool_result_summary"] = flow["result_summary"]


def upgrade(value):
    from apps.evidence_graph_html import validate_snapshot
    from apps.glaucoma_associations import apply

    validate_snapshot(value)
    result = apply(value, value["source_associations"]["sources"], version=VERSION)
    result["title"] = "青光眼病例证据图 · 工具证据贯通版"
    from apps.glaucoma_evidence_plan import checksum

    result["checksum"] = checksum({k: v for k, v in result.items() if k != "checksum"})
    validate_snapshot(result)
    return result


def main():
    from apps import vessel_branch_tool as tool
    from apps.evidence_graph_html import write_export
    from apps.glaucoma_case_storage import read_view, store_view
    from apps.glaucoma_evidence_plan import checksum
    from apps.knowledge_source_tools import repository
    from db.knowledge import models as k

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-code", required=True)
    p.add_argument("--from-case-graph", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.output.with_suffix(".json").exists():
        p.error("output must be new")
    value = read_view(a.case_code, a.from_case_graph)
    # Reading historical graphs remains independent of current tool eligibility;
    # creating this new binding additionally checks the installed current tool.
    spec = tool.load_verified()
    with tool.native_connection() as connection:
        if tool.rows(connection) != [
            (tool.VERSION, True, tool.IMPLEMENTATION, True, "smoke_tested", tool.checksum(spec))
        ]:
            raise ValueError("association_current_tool_not_available")
    repo = repository("editor")
    for source in value["source_associations"]["sources"]["sources"].values():
        for raw in [source["record"], *(c["record"] for c in source["chunks"])]:
            record = k.parse_record_json(k.KnowledgeKind(raw["kind"]), json.dumps(raw))
            saved = repo.get(record.reference())
            if saved.withdrawn or saved.record != record:
                raise ValueError("association_current_source_mismatch")
    result = upgrade(value)
    artifact = store_view(result)
    result = read_view(a.case_code, artifact["asset_code"])
    result["case_graph_artifact"] = artifact
    result["checksum"] = checksum({k: v for k, v in result.items() if k != "checksum"})
    write_export(result, a.output)
    print(json.dumps({"artifact": artifact, "html": str(a.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
