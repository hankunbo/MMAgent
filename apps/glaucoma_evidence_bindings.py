"""Traceable Draft node/relationship bindings; never a clinical scoring authority.

Knowledge references stay in Knowledge. Patient observations stay in Runtime;
this authoring ledger links to them separately and never publishes or executes.
"""

import argparse
import json
from pathlib import Path

from apps.glaucoma_evidence_plan import DEFAULT, ROOT, TASKS, checksum, inspect_tools, load
from db.knowledge import models as k
from packages.object_store import LocalObjectStore
from packages.object_store import checksum as byte_checksum

DESTINATION = ROOT / "docs/experiments/glaucoma-bindings-v1"
VERSION = "1.3.0"
SECTIONS = ("1.2.1", "1.2.4", "1.2.6", "1.5.3", "1.5.4", "1.5.5", "1.5.6", "1.5.7")
# Only the stated category is justified. This map does not assert each proposed
# fine-grained measurement is mentioned in NG81, or has a diagnostic coefficient.
CATEGORY = {
    "ROOT": ("1.2.1",),
    "STRUCTURE": ("1.2.1", "1.2.4"),
    "FUNCTION": ("1.2.1",),
    "PRESSURE": ("1.2.1",),
    "CONTEXT": ("1.2.1", "1.2.6"),
    "CUPDISC": ("1.2.1",),
    "CUP_RIM_MEASURE": ("1.2.1",),
    "RNFL": ("1.2.4",),
    "RNFL_THICKNESS": ("1.2.4",),
    "RNFL_REFLECTANCE": ("1.2.4",),
    "CHANGE": ("1.5.6", "1.5.7"),
    "TEMPORAL_CHANGE": ("1.5.6", "1.5.7"),
    "FIELD_PATTERN": ("1.2.1",),
    "FIELD_VALUES": ("1.2.1",),
    "FIELD_REPEAT": ("1.5.3", "1.5.4", "1.5.5"),
    "FIELD_QC": ("1.5.3", "1.5.5"),
    "IOP_METHOD": ("1.2.1",),
    "IOP_VALUES": ("1.2.1",),
    "CCT": ("1.2.1",),
    "CCT_VALUES": ("1.2.1",),
    "ANGLE": ("1.2.1",),
    "ANGLE_VALUES": ("1.2.1",),
    "RISK": ("1.2.6",),
    "RISK_FIELDS": ("1.2.6",),
}
LIMITS = {
    "1.2.1": "COAG及相关情况的检查类别；不提供本树各测量的阈值或支持权重。",
    "1.2.4": "视神经头基线影像；未规定RNFL/GCC厚度算法，SLO不是OCT体数据。",
    "1.2.6": "OHT初诊的未来视力损害风险背景；不是普通人群患病概率公式，未列全部人口学字段。",
    "1.5.3": "有临床指征时复查视野；不提供可靠性数值阈值。",
    "1.5.4": "OHT/疑似COAG且既往正常视野的复查语境。",
    "1.5.5": "已有缺损时保持视野测量策略；不是自动判定可靠性的算法。",
    "1.5.6": "有临床指征时复查视神经头。",
    "1.5.7": "检出视神经头变化后保存新基线；未给图像配准或变化阈值。",
}
PARTIAL = {
    "CUP_RIM_MEASURE",
    "RNFL",
    "RNFL_THICKNESS",
    "RNFL_REFLECTANCE",
    "FIELD_QC",
    "RISK_FIELDS",
}
FETCHES = {
    "egs": ROOT / ".local/knowledge-page-calls/egs-x_kl0vez/receipt.json",
    "nei": ROOT / ".local/knowledge-page-calls/nei-ylcladk2/receipt.json",
    "dalhousie": ROOT / ".local/knowledge-page-calls/dalhousie-_ldniess/receipt.json",
}
PAGE_MATCHES = {
    "egs": ("Small discs", "large discs", "disc size", "cup/disc"),
    "nei": ("family history", "high eye pressure", "optic nerve"),
    "dalhousie": ("diagnosis of glaucoma is challenging", "share the same features"),
}
PAGE_NODES = {
    "egs": {"DISC_SIZE", "DISC_GEOMETRY", "CUPDISC", "CUP_RIM_MEASURE"},
    "nei": {"ROOT", "RISK", "RISK_FIELDS", "PRESSURE", "FUNCTION"},
    "dalhousie": {"ALTERNATIVES", "CONTEXT"},
}


def ref(value):
    return k.KnowledgeReference.model_validate_json(json.dumps(value))


def dump(value):
    return value.model_dump(mode="json")


def write_new(path, value):
    """Idempotent exact retry; never replace a different historical artifact."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("binding_file_conflict")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def index_nice(repo, plan):
    from apps.patient_service_assembly import OBJECT_ROOTS, check_object_root
    from packages.knowledge_index.models import IndexConfig, SectionChunkPolicy
    from packages.knowledge_index.parsers import parse

    source = repo.get(ref(plan["source_ref"]))
    if source.withdrawn or source.status is k.KnowledgeStatus.DEPRECATED:
        raise ValueError("source_unavailable")
    check_object_root(OBJECT_ROOTS["knowledge"], "knowledge")
    store = LocalObjectStore(OBJECT_ROOTS["knowledge"])
    body = store.read_bounded(source.record.object_uri, 2_000_000)
    if (
        len(body) != source.record.size_bytes
        or byte_checksum(body) != source.record.content_checksum
    ):
        raise ValueError("source_bytes_mismatch")
    parsed = parse(body, source.record.media_type, language="en")
    if parsed.status != "parsed":
        raise ValueError("source_parse_failed")
    paths = tuple(
        sorted(
            {
                b.position.section_path
                for b in parsed.document.blocks
                if b.position.section_path and b.position.section_path[-1] in SECTIONS
            }
        )
    )
    if {p[-1] for p in paths} != set(SECTIONS):
        raise ValueError("required_source_sections_missing")
    policy = SectionChunkPolicy(source_checksum=source.record.content_checksum, section_paths=paths)
    context = k.seal(
        k.ApplicabilityCondition,
        code="EVIDENCE:GLAUCOMA:SOURCE_BINDING_CONTEXT",
        version="1.0.0",
        source_refs=(source.record.reference(),),
        population=("adults",),
        conditions=("COAG", "OHT", "suspected_COAG"),
        scope_note="NG81 adult authoring context; not patient eligibility or a diagnosis rule",
    )
    repo.put(context)
    receipt = json.loads(DEFAULT.with_name("source-receipt.json").read_text())["archive"]
    built = repo.build_document_index(
        receipt,
        store,
        IndexConfig(
            mode="fts_only",
            vector_weight=0.0,
            fts_weight=1.0,
            vector_unavailable_reason="real_embedding_provider_not_configured_for_authoring",
        ),
        chunk_policy=policy,
        applicability_refs=(context.reference(),),
        targets=("glaucoma_authoring",),
    )
    if built["status"] != "indexed":
        raise ValueError("section_index_not_built")
    chunks = []
    for value in built["chunk_refs"]:
        snapshot = repo.get(ref(value))
        c = snapshot.record
        text = store.read_bounded(c.object_uri, 16384)
        if (
            snapshot.withdrawn
            or byte_checksum(text) != c.content_checksum
            or len(text) != c.size_bytes
        ):
            raise ValueError("chunk_bytes_mismatch")
        chunks.append(
            {
                "ref": value,
                "source_ref": dump(c.source_version_ref),
                "location": dump(c.location),
                "position": dump(c.derivation.position),
                "object_uri": c.object_uri,
                "text_checksum": c.content_checksum,
                "text": text.decode(),
            }
        )
    return built, chunks


def fixed_page_evidence():
    """Actual existing tool receipts; explicitly NOT archived Knowledge SourceVersions."""
    from bs4 import BeautifulSoup

    from apps.knowledge_source_tools import SITES

    result = []
    for key, path in FETCHES.items():
        receipt = json.loads(path.read_text())
        body = path.with_name("body.bin").read_bytes()
        if (
            receipt["status"] != "succeeded"
            or byte_checksum(body) != "sha256:" + receipt["content_sha256"]
            or len(body) != receipt["byte_count"]
        ):
            raise ValueError("page_receipt_bytes_mismatch")
        soup = BeautifulSoup(body, "html.parser")
        for item in soup(("script", "style", "noscript")):
            item.decompose()
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all(("p", "li"))]
        selected = [
            {"paragraph_index": i, "text": text, "text_checksum": byte_checksum(text.encode())}
            for i, text in enumerate(paragraphs)
            if len(text) <= 4000 and any(term.lower() in text.lower() for term in PAGE_MATCHES[key])
        ]
        if not selected:
            raise ValueError("page_location_not_found:" + key)
        result.append(
            {
                "id": key,
                "kind": "verified_existing_tool_fetch_not_source_version",
                "provider_ref": receipt["provider_ref"],
                "url": "https://" + SITES[key][1] + SITES[key][2],
                "receipt": str(path.relative_to(ROOT)),
                "receipt_checksum": byte_checksum(path.read_bytes()),
                "body_checksum": byte_checksum(body),
                "paragraphs": selected,
                "extraction": "beautifulsoup-p-li/1.0.0",
                "source_version_ref": None,
                "scope": "education/research; qualitative context, not a numeric diagnostic weight",
            }
        )
    return result


def tool_tasks(key):
    tasks = []
    for task, observation, inp, outs in TASKS:
        if key in (observation, inp, *outs, "TASK_" + task.upper()):
            tasks.append(task)
    return sorted(tasks)


def build_bindings(plan, chunks, pages):
    if any(c["source_ref"] != plan["source_ref"] for c in chunks):
        raise ValueError("foreign_source_chunk")
    by_key = {n["key"]: n for n in plan["display_tree"]}
    tools = {t["task"]: t for t in plan["tool_bindings"]}
    nodes = []
    for key, n in by_key.items():
        selected = [c for c in chunks if c["position"]["section_path"][-1] in CATEGORY.get(key, ())]
        knowledge = [
            {
                "kind": "archived_guideline_category",
                "chunk_ref": c["ref"],
                "source_ref": c["source_ref"],
                "location": c["location"],
                "text_checksum": c["text_checksum"],
                "scope": LIMITS[c["position"]["section_path"][-1]],
            }
            for c in selected
        ]
        knowledge += [
            {
                "kind": "fixed_page_receipt",
                "page_id": p["id"],
                "body_checksum": p["body_checksum"],
                "paragraph_checksums": [x["text_checksum"] for x in p["paragraphs"]],
                "scope": p["scope"],
            }
            for p in pages
            if key in PAGE_NODES[p["id"]]
        ]
        tool = [
            {
                "task": t,
                "tool_code": tools[t]["tool_code"],
                "version": tools[t]["version"],
                "implementation_key": tools[t]["implementation_key"],
                "manifest_checksum": tools[t]["manifest_checksum"],
                "meaning": "input/output capability; not a patient result or glaucoma association",
            }
            for t in tool_tasks(key)
            if t in tools
        ]
        medical_status = (
            "partial_category_only"
            if key in PARTIAL
            else "qualitative_source_bound"
            if knowledge
            else "not_established"
        )
        nodes.append(
            {
                **n,
                "knowledge": knowledge,
                "tools": tool,
                "memory": {"status": "not_implemented", "references": []},
                "patient_observation": "separate_Runtime_binding_required",
                "medical_basis_status": medical_status,
                "association_weight": None,
                "weight_reason": "no_approved_quantitative_mapping",
                "gap": "specific_medical_relation_source_missing"
                if not knowledge and not tool
                else "fine_grained_rule_not_established"
                if key in PARTIAL
                else None,
            }
        )
    by_key = {n["key"]: n for n in nodes}
    relationships = []
    for n in nodes:
        if n["parent"] is None:
            continue
        relationships.append(
            {
                "id": "organizes:" + n["parent"] + ":" + n["key"],
                "parent": n["parent"],
                "child": n["key"],
                "relation": "organizes_evidence_request",
                "causal": False,
                "knowledge": n["knowledge"],
                "tools": n["tools"],
                "basis_status": n["medical_basis_status"],
                "organization_owner": "glaucoma-research-authoring/1.0.0",
                "score": None,
                "score_reason": "organization_is_not_medical_support",
            }
        )
    code_keys = {n["node_code"]: n["key"] for n in nodes}
    template = next(r for r in plan["records"] if r["kind"] == "template")
    for i, edge in enumerate(template["edges"]):
        a, b = code_keys[edge["source_node_code"]], code_keys[edge["target_node_code"]]
        evidence = by_key[a]["tools"] + by_key[b]["tools"]
        unique = {e["task"]: e for e in evidence}
        relationships.append(
            {
                "id": f"technical:{i}",
                "parent": a,
                "child": b,
                "relation": edge["edge_type"],
                "causal": False,
                "knowledge": [],
                "tools": [unique[t] for t in sorted(unique)],
                "basis_status": "technical_contract",
                "organization_owner": "glaucoma-research-authoring/1.0.0",
                "score": None,
                "score_reason": "technical_dependency_not_disease_weight",
            }
        )
    # These are source-backed research statements, not executable SUPPORTS edges.
    # Direction here means how to read the statement, never causality or a weight.
    assertions = (
        (
            "STRUCTURE",
            "ROOT",
            "recommended_assessment",
            "视神经/眼底检查属于COAG相关诊断取证类别。",
            "nice",
        ),
        (
            "FUNCTION",
            "ROOT",
            "recommended_assessment",
            "标准自动视野检查属于COAG相关诊断取证类别。",
            "nice",
        ),
        (
            "IOP_METHOD",
            "ROOT",
            "recommended_assessment",
            "眼压测量属于诊断取证，单一眼压值不是本图的诊断公式。",
            "nice",
        ),
        ("CCT", "ROOT", "recommended_assessment", "中央角膜厚度是建议获取的检查信息。", "nice"),
        ("ANGLE", "ROOT", "recommended_assessment", "房角检查用于相关情况的诊断取证。", "nice"),
        (
            "CHANGE",
            "STRUCTURE",
            "longitudinal_comparison",
            "有指征时复查视神经头，变化后保存新的影像基线。",
            "nice",
        ),
        (
            "FIELD_REPEAT",
            "FIELD_PATTERN",
            "measurement_context",
            "复查需考虑既有视野状态，并保持已检出缺损的测量策略。",
            "nice",
        ),
        (
            "RISK",
            "ROOT",
            "risk_context",
            "OHT的未来视力损害风险需考虑眼压、CCT、家族史、预期寿命；不等于患病概率。",
            "nice",
        ),
        (
            "DISC_SIZE",
            "CUPDISC",
            "interpretation_modifier",
            "杯盘比的解释受视盘大小影响；本文裂隙灯物理测量不等于mask像素尺寸。",
            "egs",
        ),
        (
            "ALTERNATIVES",
            "ROOT",
            "alternative_explanation",
            "近视和青光眼可具有相似表现，需要区分而非按同一形态直接判病。",
            "dalhousie",
        ),
        (
            "PRESSURE",
            "ROOT",
            "qualitative_association",
            "高眼压与青光眼风险相关，但高眼压不充分、正常眼压也不能排除。",
            "nei",
        ),
    )
    for a, b, relation, statement, source in assertions:
        evidence = [
            e
            for e in by_key[a]["knowledge"]
            if (source == "nice" and e["kind"] == "archived_guideline_category")
            or e.get("page_id") == source
        ]
        if not evidence:
            continue  # No invented source binding when a source has not been observed.
        relationships.append(
            dict(
                id=f"research:{a}:{b}:{relation}",
                parent=a,
                child=b,
                relation=relation,
                causal=False,
                statement=statement,
                knowledge=evidence,
                tools=[],
                basis_status="qualitative_source_bound",
                organization_owner="glaucoma-research-authoring/1.0.0",
                score=None,
                score_reason="source_does_not_supply_a_validated_patient_weight",
            )
        )
    value = {
        "schema": "glaucoma-provenance-bindings/1.0.0",
        "plan_checksum": plan["checksum"],
        "nodes": nodes,
        "relationships": relationships,
        "pages": pages,
        "memory_available": False,
        "medical_scoring_enabled": False,
        "clinical_claims_created": False,
        "source_chunks": chunks,
    }
    return {**value, "checksum": checksum(value)}


def revised_records(plan, bindings):
    """Append original-model Draft versions; do not add an ASSOCIATES contract enum."""
    bound = {
        n["node_code"]: tuple(ref(e["chunk_ref"]) for e in n["knowledge"] if "chunk_ref" in e)
        for n in bindings["nodes"]
    }
    mapping = {}
    records = []
    all_chunks = tuple(ref(c["ref"]) for c in bindings["source_chunks"])
    for raw in plan["records"]:
        old = k.parse_record_json(k.KnowledgeKind(raw["kind"]), json.dumps(raw))
        fields = {
            name: getattr(old, name)
            for name in type(old).model_fields
            if name not in ("kind", "checksum", "version", "supersedes")
        }
        for name, value in list(fields.items()):
            if isinstance(value, k.KnowledgeReference):
                fields[name] = mapping.get(value, value)
        refs = bound.get(old.code, bound.get(old.code.removesuffix(":REQUIREMENT"), ()))
        if isinstance(old, k.Template):
            refs = all_chunks
        if hasattr(old, "binding_refs"):
            fields["binding_refs"] = refs
            fields["source_refs"] = (ref(plan["source_ref"]),) if refs else ()
        if isinstance(old, k.Concept):
            node = next(n for n in bindings["nodes"] if old.code == "CONCEPT:GLAUCOMA:" + n["key"])
            fields["source_refs"] = (
                (ref(plan["source_ref"]),)
                if any("chunk_ref" in e for e in node["knowledge"])
                else ()
            )
        new = k.seal(type(old), version=VERSION, supersedes=old.reference(), **fields)
        records.append(new)
        mapping[old.reference()] = new.reference()
    return tuple(records)


def verify_bindings(plan, bindings):
    for c in bindings["source_chunks"]:
        if byte_checksum(c["text"].encode()) != c["text_checksum"]:
            raise ValueError("chunk_text_checksum_mismatch")
    for p in bindings["pages"]:
        for q in p["paragraphs"]:
            if byte_checksum(q["text"].encode()) != q["text_checksum"]:
                raise ValueError("page_text_checksum_mismatch")
    if build_bindings(plan, bindings["source_chunks"], bindings["pages"]) != bindings:
        raise ValueError("bindings_do_not_match_builder")
    return bindings


def existing_case_binding():
    """Read-only historical evidence joined back to original Runtime and object bytes."""
    from apps.research_single_case import credential_boundary
    from db.runtime.repository.research import ResearchRunRepository
    from db.runtime.repository.tool_execution import RuntimeToolExecutions

    path = ROOT / ".local/07-j-fairclip-shared-gpu-v1"
    observed = json.loads((path / "recover-7teoaw8k/observed.json").read_text())
    raw = (path / "recover-7teoaw8k/command.log").read_bytes()
    if observed["exit_code"] != 0 or byte_checksum(raw) != "sha256:" + observed["log_sha256"]:
        raise ValueError("historical_report_receipt_mismatch")
    report = json.loads(raw)
    if checksum({k: v for k, v in report.items() if k != "checksum"}) != report["checksum"]:
        raise ValueError("historical_report_checksum_mismatch")
    target = credential_boundary().resolve_target("runtime")
    context = ResearchRunRepository(target).read_research_run(report["run_id"])
    fact = RuntimeToolExecutions(target).recover(report["run_id"], report["execution_key"])
    if (
        not fact
        or fact["status"] != "CONFIRMED"
        or fact["plan_checksum"] != report["plan_checksum"]
    ):
        raise ValueError("confirmed_runtime_execution_required")
    bundle = json.loads((path / "installed-bundle.json").read_text())
    outputs = []
    for entry in report["model_outputs"]:
        source = entry["source"]
        if (
            source["scope"] != {**dump(context.request.scope), "run_id": context.run_id}
            or source["attempt_id"] != fact["attempt_id"]
        ):
            raise ValueError("case_observation_scope_mismatch")
        if source["step_id"] not in [s["step_id"] for s in fact["steps"]]:
            raise ValueError("case_observation_step_mismatch")
        step = next(s["result"] for s in fact["steps"] if s["step_id"] == source["step_id"])
        if (
            step["implementation_ref"] != source["implementation"]
            or step["plan_checksum"] != source["plan_checksum"]
        ):
            raise ValueError("case_observation_implementation_mismatch")
        obj = source["object_ref"]
        artifacts = step["result"]["artifacts"]
        if not any(
            a["asset_code"] == source["artifact_code"]
            and a["checksum"] == source["artifact_checksum"] == obj["checksum"]
            and a["uri"] == obj["uri"]
            and entry["output"]["input_checksum"] in a["provenance"]["input_checksums"]
            for a in artifacts
        ):
            raise ValueError("case_observation_artifact_mismatch")
        data = LocalObjectStore(Path(bundle["output_root"])).read_bounded(obj["uri"], 1048576)
        if (
            len(data) != obj["byte_size"]
            or byte_checksum(data) != obj["checksum"]
            or json.loads(data) != entry["output"]
        ):
            raise ValueError("case_observation_object_mismatch")
        if entry["output"]["input_checksum"] not in {a.checksum for a in context.request.artifacts}:
            raise ValueError("case_observation_input_mismatch")
        outputs.append(entry)
    return dict(
        schema="glaucoma-case-observation-binding/1.0.0",
        node_key="MODEL_ASSESSMENT",
        run_id=report["run_id"],
        context_checksum=context.checksum,
        report_checksum=report["checksum"],
        sample=bundle["training_selection"]["filename"],
        technical_status=report["technical_status"],
        outputs=outputs,
        graph=report["graph"],
        accepted_validations=report["accepted_validations"],
        qualification="verified_historical_research_output_not_medical_support",
        historical=True,
        new_execution=False,
        current_execution_authorized=False,
        limitations=[
            "old score is not a calibrated disease probability",
            "no new SLO measurements or memory evidence imported",
            "current tool eligibility must be rechecked before any new execution",
        ],
    )


def engineering_evidence():
    """QC is justified by engineering contracts, not an invented medical claim."""
    paths = (
        ("packages/dataset_adapters/prepared.py", "_capture", ("IMAGE_QC",)),
        ("Algorithm_Tool/slo_observations/runner.py", "read_image/main", ("IMAGE_QC", "OUTPUT_QC")),
        ("packages/tool_execution/qc.py", "registered QC dispatch", ("OUTPUT_QC",)),
    )
    return dict(
        schema="glaucoma-engineering-bindings/1.0.0",
        entries=[
            dict(
                path=path,
                checksum=byte_checksum((ROOT / path).read_bytes()),
                symbols=symbols,
                nodes=list(nodes),
                qualification="technical_check_not_diagnosis",
            )
            for path, symbols, nodes in paths
        ],
        medical_weight=None,
    )


def additional_catalog_evidence():
    """Reuse already registered formulas; this does not install or authorize them."""
    from apps.slo_tool_registry import native_connection

    candidates = (
        ("DISC_GEOMETRY", "calculate_optic_disc_geometry", "local_disc_geometry_v1"),
        ("CUP_RIM_MEASURE", "calculate_vcdr", "local_typed_formula"),
        ("CUP_RIM_MEASURE", "calculate_acdr", "local_typed_formula"),
        ("CUP_RIM_MEASURE", "calculate_hcdr", "local_mask_formula"),
        ("CUP_RIM_MEASURE", "calculate_rdar", "local_mask_formula"),
        ("CUP_RIM_MEASURE", "calculate_rim_width_profile", "local_mask_formula"),
    )
    entries = []
    with native_connection() as connection:
        for node, code, implementation in candidates:
            manifest = connection.execute(
                "SELECT medgraph.get_tool_execution_manifest(%s,%s,%s)",
                (code, "1.0.0", implementation),
            ).fetchone()[0]
            state = connection.execute(
                "SELECT enabled,validation_status FROM medgraph.v_tool_execution_manifest "
                "WHERE tool_code=%s AND version=%s AND implementation_key=%s",
                (code, "1.0.0", implementation),
            ).fetchone()
            if manifest is None or state is None:
                raise ValueError("existing_algorithm_missing")
            entries.append(
                dict(
                    node=node,
                    tool_code=code,
                    version="1.0.0",
                    implementation_key=implementation,
                    manifest=manifest,
                    manifest_checksum=checksum(manifest),
                    enabled=state[0],
                    validation_status=state[1],
                    execution_authorized=False,
                    missing_resource_digests=[
                        r["resource_key"] for r in manifest["resources"] if not r.get("sha256")
                    ],
                    patient_input_ready=False,
                    input_gap="no_same_case_disc_mask"
                    if node == "DISC_GEOMETRY"
                    else "no_selected_same_case_cup_and_disc_masks",
                )
            )
    return dict(
        schema="glaucoma-existing-formula-bindings/1.0.0",
        entries=entries,
        catalog_mutated=False,
        installed=False,
        medical_weight=None,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "show", "verify", "case", "catalog"))
    parser.add_argument("--node")
    parser.add_argument("--directory", type=Path, default=DESTINATION)
    args = parser.parse_args(argv)
    from apps.knowledge_source_tools import repository

    plan = load(DEFAULT)
    if args.action == "catalog":
        result = additional_catalog_evidence()
        write_new(args.directory / "existing-algorithm-bindings.json", result)
        print(json.dumps({"existing_algorithms": len(result["entries"]), "catalog_mutated": False}))
        return 0
    if args.action == "case":
        result = existing_case_binding()
        write_new(args.directory / "case-observation.json", result)
        print(
            json.dumps(
                {
                    "node": result["node_key"],
                    "run_id": result["run_id"],
                    "technical_status": result["technical_status"],
                    "new_execution": False,
                }
            )
        )
        return 0
    if args.action == "prepare":
        if inspect_tools() != plan["tool_bindings"]:
            raise ValueError("tool_snapshot_changed")
        repo = repository("editor")
        index, chunks = index_nice(repo, plan)
        bindings = build_bindings(plan, chunks, fixed_page_evidence())
        records = revised_records(plan, bindings)
        states = repo.put_many(records, idempotency_key="glaucoma-bindings:" + bindings["checksum"])
        if not all(s.record == r for s, r in zip(states, records, strict=True)):
            raise ValueError("draft_readback_mismatch")
        bindings["new_template_ref"] = dump(
            next(r for r in records if isinstance(r, k.Template)).reference()
        )
        # Keep the builder hash independent of the exact original-model DB envelope.
        write_new(args.directory / "index.json", index)
        write_new(args.directory / "records.json", [dump(r) for r in records])
        write_new(args.directory / "bindings.json", bindings)
        write_new(args.directory / "engineering-bindings.json", engineering_evidence())
        print(
            json.dumps(
                {
                    "nodes": len(bindings["nodes"]),
                    "relationships": len(bindings["relationships"]),
                    "draft_records": len(records),
                    "chunks": len(chunks),
                    "published": False,
                }
            )
        )
        return 0
    bindings = json.loads((args.directory / "bindings.json").read_text())
    template = bindings.pop("new_template_ref")
    verify_bindings(plan, bindings)
    engineering_path = args.directory / "engineering-bindings.json"
    engineering = json.loads(engineering_path.read_text()) if engineering_path.exists() else None
    if engineering is not None and engineering != engineering_evidence():
        raise ValueError("engineering_binding_source_drift")
    if args.action == "verify":
        repo = repository("editor")
        from apps.patient_service_assembly import OBJECT_ROOTS, check_object_root

        check_object_root(OBJECT_ROOTS["knowledge"], "knowledge")
        store = LocalObjectStore(OBJECT_ROOTS["knowledge"])
        source = repo.get(ref(plan["source_ref"]))
        if source.withdrawn or source.status is k.KnowledgeStatus.DEPRECATED:
            raise ValueError("source_unavailable")
        if fixed_page_evidence() != bindings["pages"]:
            raise ValueError("page_receipt_changed")
        for c in bindings["source_chunks"]:
            s = repo.get(ref(c["ref"]))
            body = store.read_bounded(s.record.object_uri, 16384)
            if (
                s.withdrawn
                or dump(s.record.source_version_ref) != c["source_ref"]
                or dump(s.record.location) != c["location"]
                or dump(s.record.derivation.position) != c["position"]
                or s.record.content_checksum != c["text_checksum"]
                or byte_checksum(body) != c["text_checksum"]
                or body.decode() != c["text"]
            ):
                raise ValueError("stored_chunk_binding_mismatch")
        raw = json.loads((args.directory / "records.json").read_text())
        if raw != [dump(r) for r in revised_records(plan, bindings)]:
            raise ValueError("revised_records_mismatch")
        if template != dump(
            next(
                r for r in revised_records(plan, bindings) if isinstance(r, k.Template)
            ).reference()
        ):
            raise ValueError("template_binding_mismatch")
        states = [
            repo.get(k.parse_record_json(k.KnowledgeKind(r["kind"]), json.dumps(r)).reference())
            for r in raw
        ]
        if any(
            s.withdrawn or s.status is not k.KnowledgeStatus.DRAFT or dump(s.record) != r
            for s, r in zip(states, raw, strict=True)
        ):
            raise ValueError("draft_snapshot_mismatch")
        print(
            json.dumps({"exact_records": len(states), "template_ref": template, "published": False})
        )
    else:
        nodes = [n for n in bindings["nodes"] if args.node is None or n["key"] == args.node]
        if not nodes:
            raise ValueError("node_not_found")
        print(
            json.dumps(
                {
                    "template_ref": template,
                    "nodes": nodes,
                    "engineering_sources": engineering,
                    "case_observation_file": str(args.directory / "case-observation.json")
                    if (args.directory / "case-observation.json").exists()
                    else None,
                    "existing_algorithm_bindings_file": str(
                        args.directory / "existing-algorithm-bindings.json"
                    )
                    if (args.directory / "existing-algorithm-bindings.json").exists()
                    else None,
                    "relationships": [
                        e
                        for e in bindings["relationships"]
                        if args.node is None or args.node in (e["parent"], e["child"])
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
