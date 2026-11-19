"""Glaucoma Draft authoring and technical dependencies, never a patient graph.

Reuses the original Knowledge models, archive and repository. The display tree
does not invent AND/SUPPORTS rules. Tool bindings are catalog observations, not
Grounding or execution authorization. No patient data or model execution here.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from db.knowledge import models as k
from packages.contracts import EdgeType, NodeType

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "docs/experiments/glaucoma-demand-plan-v1/plan.json"
VERSION = "1.2.0"
TEMPLATE = "TEMPLATE:GLAUCOMA:RESEARCH_DEMAND_PLAN"

# A source section justifies looking for evidence, not a diagnosis threshold.
# Specific refinements not established by NG81 are explicitly authoring proposals.
TOPICS = (
    ("CUPDISC", "STRUCTURE", "盘杯及盘沿形态", "1.2.1", "missing_cup_model"),
    ("RNFL", "STRUCTURE", "RNFL及黄斑神经层结构", "1.2.4", "missing_OCT_pixels"),
    ("HEMORRHAGE", "STRUCTURE", "盘出血等形态线索", None, "missing_SLO_detector"),
    ("DISC_SIZE", "STRUCTURE", "盘大小与测量背景", None, "pixel_mask_only_no_physical_scale"),
    ("CHANGE", "STRUCTURE", "同眼纵向变化", "1.5.6–1.5.7", "missing_matched_followup"),
    ("FIELD_PATTERN", "FUNCTION", "视野缺损模式", "1.2.1", "missing_visual_field"),
    ("FIELD_REPEAT", "FUNCTION", "视野可靠性与重复性", "1.5.3–1.5.5", "missing_visual_field"),
    ("IOP_METHOD", "PRESSURE", "眼压方法与重复测量", "1.2.1", "missing_tonometry"),
    ("CCT", "PRESSURE", "角膜厚度背景", "1.2.1", "missing_pachymetry"),
    ("ANGLE", "CONTEXT", "房角及亚型区分", "1.2.1", "missing_angle_exam"),
    ("RISK", "CONTEXT", "年龄与家族史等风险背景", "1.2.6", "approved_visible_metadata_only"),
    ("ALTERNATIVES", "CONTEXT", "其他眼病或测量伪影的解释", None, "interpretation_rule_missing"),
)

# These are narrower product demands. All remain unobserved for a patient.
OBSERVATIONS = (
    (
        "CUP_RIM_MEASURE",
        "CUPDISC",
        "同图杯盘比与盘沿测量",
        "cup_and_disc_masks",
        "missing_cup_model",
    ),
    (
        "RNFL_THICKNESS",
        "RNFL",
        "有方法及分区的RNFL/GCC厚度",
        "OCT_volume_or_report",
        "missing_input",
    ),
    (
        "RNFL_REFLECTANCE",
        "RNFL",
        "SLO神经纤维层外观线索（不是厚度）",
        "SLO_image",
        "missing_validated_method",
    ),
    (
        "DISC_BLEED",
        "HEMORRHAGE",
        "出血候选位置及误检解释",
        "image_with_domain_evidence",
        "missing_tool",
    ),
    ("DISC_GEOMETRY", "DISC_SIZE", "视盘mask及像素尺度背景", "disc_mask", "registered_component"),
    (
        "TEMPORAL_CHANGE",
        "CHANGE",
        "同眼同方法配准前后变化",
        "paired_images_with_time",
        "missing_input",
    ),
    ("FIELD_VALUES", "FIELD_PATTERN", "视野点位、缺损模式及检查方法", "perimetry", "missing_input"),
    ("FIELD_QC", "FIELD_REPEAT", "视野可靠性指标与重复结果", "perimetry_metadata", "missing_input"),
    ("IOP_VALUES", "IOP_METHOD", "眼压数值、单位、方法和时点", "tonometry", "missing_input"),
    ("CCT_VALUES", "CCT", "角膜厚度、单位及测量方法", "pachymetry", "missing_input"),
    ("ANGLE_VALUES", "ANGLE", "房角检查及亚型相关观察", "gonioscopy", "missing_input"),
    (
        "RISK_FIELDS",
        "RISK",
        "获准可见的人口学与明确家族史",
        "approved_metadata",
        "field_specific_availability",
    ),
    (
        "IMAGE_QC",
        "ALTERNATIVES",
        "图像格式、大小、质量及测量局限",
        "SLO_image",
        "technical_checks_only",
    ),
    (
        "VESSEL_FEATURES",
        "ALTERNATIVES",
        "血管几何辅助观察（不是青光眼判定规则）",
        "vessel_mask",
        "registered_component",
    ),
    (
        "AV_FEATURES",
        "ALTERNATIVES",
        "动静脉分区辅助观察",
        "artery_and_vein_masks",
        "registered_component",
    ),
    (
        "FOVEA_LOCATION",
        "ALTERNATIVES",
        "中央凹位置与图像空间背景",
        "fovea_response",
        "registered_component",
    ),
    (
        "MODEL_ASSESSMENT",
        "ALTERNATIVES",
        "独立模型分类输出与医学证据分栏",
        "confirmed_model_output",
        "existing_FairCLIP_run_not_imported",
    ),
)

PRODUCTS = (
    ("SLO_IMAGE", "IMAGE_QC", "同一病例的原生SLO可见图像", "slo:ImagePNG"),
    ("VESSEL_MASK", "VESSEL_FEATURES", "同图血管mask", "slo:VesselMask"),
    ("ARTERY_MASK", "AV_FEATURES", "同图动脉mask", "slo:ArteryMask"),
    ("VEIN_MASK", "AV_FEATURES", "同图静脉mask", "slo:VeinMask"),
    ("DISC_MASK", "DISC_GEOMETRY", "同图视盘mask（不含视杯）", "slo:OpticDiscMask"),
    ("FOVEA_MAP", "FOVEA_LOCATION", "中央凹原始响应图", "slo:FoveaHeatmap"),
    ("DENSITY", "VESSEL_FEATURES", "整图血管面积占比", "slo:Observation"),
    ("WIDTH", "VESSEL_FEATURES", "全局像素宽度估计", "slo:Observation"),
    ("TORTUOSITY", "VESSEL_FEATURES", "开放血管段弧弦比中位数", "slo:Observation"),
    ("FRACTAL", "VESSEL_FEATURES", "盒计数分形维数", "slo:Observation"),
)

TASKS = (
    ("vessels", "VESSEL_FEATURES", "SLO_IMAGE", ("VESSEL_MASK",)),
    ("avod", "AV_FEATURES", "SLO_IMAGE", ("ARTERY_MASK", "VEIN_MASK", "DISC_MASK")),
    ("fovea", "FOVEA_LOCATION", "SLO_IMAGE", ("FOVEA_MAP",)),
    ("density", "VESSEL_FEATURES", "VESSEL_MASK", ("DENSITY",)),
    ("width", "VESSEL_FEATURES", "VESSEL_MASK", ("WIDTH",)),
    ("tortuosity", "VESSEL_FEATURES", "VESSEL_MASK", ("TORTUOSITY",)),
    ("fractal", "VESSEL_FEATURES", "VESSEL_MASK", ("FRACTAL",)),
)


def checksum(value):
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return "sha256:" + hashlib.sha256(raw.encode()).hexdigest()


def build(source_ref, tool_bindings, *, synthetic=False):
    """Deterministic authoring records. Exact original classes, no medical rules."""
    if source_ref.kind is not k.KnowledgeKind.SOURCE_VERSION:
        raise ValueError("source_version_required")
    descriptions = [("ROOT", None, "青光眼评估目标，尚未裁定", NodeType.GOAL)]
    descriptions += [
        (key, "ROOT", title, NodeType.EVIDENCE)
        for key, title in (
            ("STRUCTURE", "视神经结构"),
            ("FUNCTION", "视功能"),
            ("PRESSURE", "眼压背景"),
            ("CONTEXT", "适用背景与其他解释"),
        )
    ]
    descriptions += [(a, b, c, NodeType.EVIDENCE) for a, b, c, _, _ in TOPICS]
    descriptions += [(a, b, c, NodeType.EVIDENCE) for a, b, c, _, _ in OBSERVATIONS]
    descriptions += [(a, b, c, NodeType.DATA) for a, b, c, _ in PRODUCTS]
    descriptions += [("TASK_" + a.upper(), b, "调用 " + a, NodeType.TOOL) for a, b, _, _ in TASKS]
    descriptions += [("OUTPUT_QC", "IMAGE_QC", "输出格式与来源绑定检查（非临床QC）", NodeType.QC)]
    grounded = {"ROOT", "STRUCTURE", "FUNCTION", "PRESSURE", "CONTEXT"}
    grounded.update(a for a, _, _, section, _ in TOPICS if section)

    def record(cls, code, key=None, **fields):
        return k.seal(
            cls,
            code=code,
            version=VERSION,
            synthetic=synthetic,
            source_refs=(source_ref,) if key in grounded else (),
            **fields,
        )

    concepts = [
        record(k.Concept, "CONCEPT:GLAUCOMA:" + a, a, name=c) for a, _, c, _ in descriptions
    ]
    prefixes = {
        NodeType.GOAL: "GOAL",
        NodeType.EVIDENCE: "EVIDENCE",
        NodeType.DATA: "ARTIFACT",
        NodeType.TOOL: "TOOL",
        NodeType.QC: "EVIDENCE",
    }
    definitions = tuple(
        k.TemplateNode(
            node_code=f"{prefixes[d]}:GLAUCOMA:{a}",
            node_type=d,
            concept_code=c.code,
            required=a == "ROOT",
        )
        for (a, _, _, d), c in zip(descriptions, concepts, strict=True)
    )
    codes = {a: n.node_code for (a, *_), n in zip(descriptions, definitions, strict=True)}
    edges = []
    for task, _, inp, outs in TASKS:
        tool = codes["TASK_" + task.upper()]
        edges.append(
            k.GraphEdge(
                source_node_code=codes[inp],
                target_node_code=tool,
                edge_type=EdgeType.REQUIRES_INPUT,
            )
        )
        edges.extend(
            k.GraphEdge(
                source_node_code=tool, target_node_code=codes[out], edge_type=EdgeType.PRODUCES
            )
            for out in outs
        )
    for key, *_ in PRODUCTS:
        if key != "SLO_IMAGE":
            edges.append(
                k.GraphEdge(
                    source_node_code=codes[key],
                    target_node_code=codes["OUTPUT_QC"],
                    edge_type=EdgeType.VALIDATED_BY,
                )
            )
    template = record(
        k.Template,
        TEMPLATE,
        "ROOT",
        goal_node_code=codes["ROOT"],
        nodes=definitions,
        edges=tuple(edges),
        rule_refs=(),
    )
    nodes = [
        record(
            k.Node,
            n.node_code,
            a,
            template_ref=template.reference(),
            concept_ref=c.reference(),
            definition=n,
        )
        for (a, *_), n, c in zip(descriptions, definitions, concepts, strict=True)
    ]
    by_code = {n.code: n for n in nodes}
    requirements = [
        record(
            k.EvidenceRequirement,
            n.code + ":REQUIREMENT",
            node_ref=n.reference(),
            artifact_type_code=n.code + ":ASSESSMENT",
            acceptance_rule_ref=None,
        )
        for n in nodes
        if n.definition.node_type is NodeType.EVIDENCE
    ]
    edge_records = [
        record(
            k.Edge,
            f"EVIDENCE:GLAUCOMA:RESEARCH_EDGE:{i}",
            template_ref=template.reference(),
            source_node_ref=by_code[e.source_node_code].reference(),
            target_node_ref=by_code[e.target_node_code].reference(),
            definition=e,
        )
        for i, e in enumerate(edges)
    ]
    records = [*concepts, template, *nodes, *requirements, *edge_records]
    for r in records:
        k.revalidate_record(r)
    result = {
        "schema": "glaucoma-demand-authoring/1.0.0",
        "version": VERSION,
        "synthetic": synthetic,
        "status": "Draft",
        "patient_graph_created": False,
        "medical_effect": None,
        "template_ref": template.reference().model_dump(mode="json"),
        "source_ref": source_ref.model_dump(mode="json"),
        "counts": dict(Counter(r.kind.value for r in records)),
        "records": [r.model_dump(mode="json") for r in records],
        "display_tree": [
            {"key": a, "parent": b, "title": c, "node_code": codes[a]}
            for a, b, c, _ in descriptions
        ],
        "topics": [
            {
                "key": a,
                "source_section": s,
                "basis": "NG81 evidence category only"
                if s
                else "authoring proposal; specific medical source not bound",
                "gap": gap,
            }
            for a, _, _, s, gap in TOPICS
        ],
        "observations": [
            {
                "key": a,
                "requires": inp,
                "readiness": status,
                "patient_observed": False,
                "medical_rule": None,
            }
            for a, _, _, inp, status in OBSERVATIONS
        ],
        "product_types": {codes[a]: dtype for a, _, _, dtype in PRODUCTS},
        "tool_bindings": tool_bindings,
        "correlation": (
            "same input and derived masks share a source group; never sum as independent evidence"
        ),
        "limits": [
            "display_tree is organization, not AND/OR or SUPPORTS",
            "proposed ASSESSMENT types are authoring placeholders, not registered capabilities",
            "catalog binding is component readiness, not a Grounding mapping or execution grant",
            "vessel metrics have no accepted glaucoma interpretation rule",
            "NG81 COAG/OHT context is not a universal subtype diagnosis formula",
            "no patient input, observations, diagnosis, publication or model execution",
        ],
    }
    return {**result, "checksum": checksum(result)}


def load(path):
    result = json.loads(path.read_text())
    digest = result.pop("checksum")
    if digest != checksum(result):
        raise ValueError("plan_checksum_mismatch")
    ref = k.KnowledgeReference.model_validate_json(json.dumps(result["source_ref"]))
    rebuilt = build(ref, result["tool_bindings"], synthetic=result["synthetic"])
    if rebuilt != {**result, "checksum": digest}:
        raise ValueError("plan_does_not_match_versioned_builder")
    return rebuilt


def inspect_tools():
    from apps.slo_tool_registry import IMPLEMENTATION, TOOLS, native_connection
    from apps.slo_tool_registry import VERSION as tool_version

    result = []
    with native_connection() as connection:
        for task, (code, *_) in TOOLS.items():
            manifest = connection.execute(
                "SELECT medgraph.get_tool_execution_manifest(%s,%s,%s)",
                (code, tool_version, IMPLEMENTATION),
            ).fetchone()[0]
            if manifest is None:
                raise ValueError("registered_tool_missing")
            state = connection.execute(
                "SELECT enabled,validation_status FROM medgraph.v_tool_execution_manifest "
                "WHERE tool_code=%s AND version=%s AND implementation_key=%s",
                (code, tool_version, IMPLEMENTATION),
            ).fetchone()
            result.append(
                {
                    "task": task,
                    "tool_code": code,
                    "version": tool_version,
                    "implementation_key": IMPLEMENTATION,
                    "enabled": state[0],
                    "validation_status": state[1],
                    "manifest_checksum": checksum(manifest),
                    "manifest": manifest,
                }
            )
    return sorted(result, key=lambda r: r["task"])


def archive_nice(repo):
    from apps.glaucoma_knowledge_intake import CONFIG, persist_discovery
    from apps.knowledge_source_tools import NICE_PROXY
    from apps.patient_service_assembly import OBJECT_ROOTS, check_object_root
    from packages.knowledge_archive import Bibliography, FetchInput
    from packages.knowledge_connectors import (
        ExplicitProxyHTTPSTransport,
        GuidelineSiteConnector,
        SafeHTTPClient,
        SourceLocator,
        load_configuration,
    )
    from packages.object_store import LocalObjectStore

    check_object_root(OBJECT_ROOTS["knowledge"], "knowledge")
    discovery = persist_discovery(repo)
    hit = next(
        k.SearchHit.model_validate_json(json.dumps(s["record"]))
        for s in discovery["records"]
        if s["record"]["kind"] == "search_hit"
        and s["record"]["discovery"]["provider_record_id"] == "NG81-RECOMMENDATIONS"
    )
    config = load_configuration(CONFIG).providers[0]
    connector = GuidelineSiteConnector(
        config, SafeHTTPClient(config, ExplicitProxyHTTPSTransport(NICE_PROXY))
    )
    doc = connector.fetch(SourceLocator(record_id="NG81-RECOMMENDATIONS"))
    receipt = repo.archive_hit(
        hit.reference(),
        FetchInput(
            document=doc,
            metadata_response_checksum=doc.response.response_checksum,
            bibliography=Bibliography(
                language="en",
                institutions=("NICE",),
                canonical_url="https://www.nice.org.uk/guidance/ng81/chapter/Recommendations",
            ),
        ),
        LocalObjectStore(OBJECT_ROOTS["knowledge"]),
    )
    ref = k.KnowledgeReference.model_validate_json(json.dumps(receipt["version_ref"]))
    return ref, {
        "archive": receipt,
        "fetch": doc.response.metadata.model_dump(mode="json"),
        "object_root": str(OBJECT_ROOTS["knowledge"]),
        "transport": NICE_PROXY.receipt(),
    }


def persist(repo, plan):
    source = repo.get(k.KnowledgeReference.model_validate_json(json.dumps(plan["source_ref"])))
    if source.withdrawn or source.status is k.KnowledgeStatus.DEPRECATED:
        raise ValueError("source_unavailable")
    records = tuple(
        k.parse_record_json(k.KnowledgeKind(r["kind"]), json.dumps(r)) for r in plan["records"]
    )
    snapshots = repo.put_many(records, idempotency_key="glaucoma-demand-plan:" + plan["checksum"])
    return {
        "records": len(snapshots),
        "statuses": dict(Counter(s.status.value for s in snapshots)),
        "all_exact": all(s.record == r for s, r in zip(snapshots, records, strict=True)),
        "template": repo.get(
            k.KnowledgeReference.model_validate_json(json.dumps(plan["template_ref"]))
        ).model_dump(mode="json"),
        "patient_graph_created": False,
        "published": False,
    }


def source_excerpt(repo, plan, query):
    """Bounded inspection of archived source bytes, not FTS/Chunk or a Claim."""
    from bs4 import BeautifulSoup

    from apps.patient_service_assembly import OBJECT_ROOTS, check_object_root
    from packages.object_store import LocalObjectStore
    from packages.object_store import checksum as object_checksum

    if not 1 <= len(query) <= 100:
        raise ValueError("bounded_authoring_query_required")
    source_ref = k.KnowledgeReference.model_validate_json(json.dumps(plan["source_ref"]))
    snapshot = repo.get(source_ref)
    if snapshot.withdrawn or snapshot.status is k.KnowledgeStatus.DEPRECATED:
        raise ValueError("source_unavailable")
    check_object_root(OBJECT_ROOTS["knowledge"], "knowledge")
    record = snapshot.record
    data = LocalObjectStore(OBJECT_ROOTS["knowledge"]).read_bounded(record.object_uri, 1_048_576)
    if len(data) != record.size_bytes or object_checksum(data) != record.content_checksum:
        raise ValueError("archived_source_bytes_mismatch")
    if record.media_type != "text/html":
        raise ValueError("html_source_required")
    soup = BeautifulSoup(data, "html.parser")
    for element in soup(("script", "style", "noscript")):
        element.decompose()
    text = soup.get_text(" ", strip=True)
    offset = text.casefold().find(query.casefold())
    return {
        "source_ref": plan["source_ref"],
        "content_checksum": record.content_checksum,
        "operation": "bounded_archived_page_find",
        "query": query,
        "found": offset >= 0,
        "excerpt": text[offset : offset + 1000] if offset >= 0 else None,
        "fts_index_created": False,
        "medical_claim_created": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "register", "show", "tree", "source"))
    parser.add_argument("--plan", type=Path, default=DEFAULT)
    parser.add_argument("--node", help="display-tree key, e.g. VESSEL_FEATURES")
    parser.add_argument(
        "--query", default="optic nerve", help="source text query, not patient data"
    )
    args = parser.parse_args(argv)
    from apps.knowledge_source_tools import repository

    repo = repository("editor")  # Authoring channel; never impersonates Published reads.
    if args.action == "prepare":
        if args.plan.exists():
            raise ValueError("preserve_existing_plan_use_new_version")
        bindings = inspect_tools()
        source_ref, receipt = archive_nice(repo)
        result = build(source_ref, bindings)
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        with args.plan.open("x") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
        with args.plan.with_name("source-receipt.json").open("x") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2)
        print(
            json.dumps(
                {"plan": str(args.plan), "counts": result["counts"], "source_archived": True}
            )
        )
        return 0
    plan = load(args.plan)
    if args.action == "source":
        result = source_excerpt(repo, plan, args.query)
    elif args.action == "register":
        if plan["synthetic"] or inspect_tools() != plan["tool_bindings"]:
            raise ValueError("real_catalog_changed_or_synthetic_plan")
        result = persist(repo, plan)
    else:
        ref = plan["template_ref"]
        if args.node:
            code = next(n["node_code"] for n in plan["display_tree"] if n["key"] == args.node)
            r = next(r for r in plan["records"] if r["kind"] == "node" and r["code"] == code)
            ref = {key: r[key] for key in ("kind", "code", "version", "checksum")}
        snapshot = repo.get(k.KnowledgeReference.model_validate_json(json.dumps(ref)))
        if args.action == "tree":
            print(f"Knowledge {snapshot.status.value} | {TEMPLATE} | {VERSION}")
            print("取证目录；不表示患者已检查、临床因果或AND/OR诊断规则。")

            def display(parent, depth=0):
                for node in plan["display_tree"]:
                    if node["parent"] == parent:
                        print("  " * depth + node["title"] + " [" + node["key"] + "]")
                        display(node["key"], depth + 1)

            display(None)
            return 0
        result = {
            "database_record": snapshot.model_dump(mode="json"),
            "counts": plan["counts"],
            "display_tree": plan["display_tree"],
            "observations": plan["observations"],
            "patient_graph_created": False,
            "medical_effect": None,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
