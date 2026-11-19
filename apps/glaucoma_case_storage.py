"""Persist the observation graph as a CASE Artifact using existing Runtime tables."""

import copy
import json
from datetime import UTC, datetime

from apps.glaucoma_case_evidence import DIRECTORY, read_json
from apps.glaucoma_evidence_bindings import dump, write_new
from apps.research_single_case import credential_boundary
from db.runtime.repository.case_evidence import CaseEvidenceRepository
from db.runtime.repository.case_pipeline import identity
from packages.contracts import Artifact
from packages.object_store import LocalObjectStore, checksum


def _active_tree(snapshot):
    structure = snapshot["evidence_structure"]
    nodes = []
    for n in snapshot["nodes"]:
        if n["key"] not in structure["main_node_keys"]:
            continue
        role = structure["node_roles"][n["key"]]
        nodes.append(
            {
                "key": n["key"],
                "code": n["node_code"],
                "parent": n["parent"],
                "knowledge_ref": {
                    k: n["database"]["record"][k] for k in ("code", "version", "checksum")
                },
                "criterion_ids": role["criterion_ids"],
                "result_summary": role["result_summary"],
                "result_status": role["result_status"],
                "evidence_source_node_key": n["key"],
            }
        )
        branch = snapshot.get("branch_geometry")
        if branch and n["key"] == branch["node"]:
            nodes[-1].update(result_summary=branch["summary"], result_status="已有结果")
    return {"nodes": nodes, "edges": structure["main_edges"]}


def store_view(snapshot, *, repo=None, store=None):
    from apps.evidence_graph_explanation import explain
    from apps.evidence_graph_html import validate_snapshot
    from apps.glaucoma_evidence_plan import checksum as document_checksum
    from apps.patient_service_assembly import OBJECT_ROOTS

    validate_snapshot(snapshot)
    saved_view = copy.deepcopy(snapshot)
    saved_view.pop("case_graph_artifact", None)
    saved_view["display_explanations"] = explain(snapshot)
    saved_view["checksum"] = document_checksum(
        {k: v for k, v in saved_view.items() if k != "checksum"}
    )
    case = snapshot["case_evidence"]
    if repo is None:
        repo = CaseEvidenceRepository(credential_boundary().resolve_target("runtime"))
    current = {a.asset_code: a for a in repo.get_case_bundle(case["case_code"]).artifacts}
    expected = {a["asset_code"]: a["checksum"] for a in [case["input"], *case["artifacts"]]}
    for output in case["fairclip"]["outputs"]:
        source = output["source"]
        expected[source["artifact_code"]] = source["artifact_checksum"]
    refs = []
    for code, digest in sorted(expected.items()):
        item = current.get(code)
        if item is None or item.checksum != digest or item.scope.case_code != case["case_code"]:
            raise ValueError("case_graph_source_mismatch")
        refs.append(item)
    template = next(s["record"] for s in snapshot["records"] if s["record"]["kind"] == "template")
    reference = {k: template[k] for k in ("code", "version", "checksum")}
    knowledge_versions = [reference]
    if "source_associations" in snapshot:
        extension = snapshot["source_associations"]["extension_template"]
        knowledge_versions.append({k: extension[k] for k in ("code", "version", "checksum")})
    graph_version = "1.3.0" if "research_criteria" in snapshot else "1.2.0"
    if "model_selection" in snapshot:
        graph_version = "1.4.0"
    if snapshot.get("research_criteria", {}).get("schema") == "case-research-criteria/1.2.0":
        graph_version = "1.5.0"
    if snapshot.get("research_criteria", {}).get("schema") == "case-research-criteria/1.3.0":
        graph_version = "1.6.0"
    if "evidence_structure" in snapshot:
        graph_version = "1.7.0"
    if "case_completion" in snapshot:
        graph_version = "1.8.0"
    if "source_associations" in snapshot:
        graph_version = "1.9.0"
    if "branch_geometry" in snapshot:
        graph_version = "1.10.0"
    if snapshot.get("source_associations", {}).get("schema") == "case-source-associations/1.1.0":
        graph_version = "1.11.0"
    document = {
        "schema": "case-evidence-graph/" + graph_version,
        "view_snapshot": saved_view,
        "case_code": case["case_code"],
        "sample": case["sample"],
        "knowledge_template": reference,
        "graph_role": "patient_observation_projection",
        "input": case["input"],
        "nodes": [
            {
                "key": n["key"],
                "code": n["node_code"],
                "parent": n["parent"],
                "knowledge_ref": {
                    k: n["database"]["record"][k] for k in ("code", "version", "checksum")
                },
                "evidence": n["patient_evidence"],
            }
            for n in snapshot["nodes"]
        ],
        "relationships": snapshot["relationships"],
        "accepted_validations": [],
        "correlation_group": case["input"]["checksum"],
        "artifacts": case["artifacts"],
        "historical_model": case["fairclip"],
        "source_artifact_codes": [a.asset_code for a in refs],
        "acquisition_policy": snapshot["acquisition_policy"],
        "acquisition_requirements": snapshot["acquisition_requirements"],
        "acquisition_summary": snapshot["acquisition_summary"],
    }
    if "research_criteria" in snapshot:
        document["research_criteria"] = snapshot["research_criteria"]
    if "model_selection" in snapshot:
        document["model_selection"] = snapshot["model_selection"]
    if "case_completion" in snapshot:
        document["case_completion"] = snapshot["case_completion"]
    if "source_associations" in snapshot:
        # The complete extension is already in view_snapshot. Bind it by exact
        # digest here instead of duplicating source passages and node records.
        document["source_associations_ref"] = {
            key: snapshot["source_associations"][key] for key in ("schema", "checksum")
        }
    if "evidence_structure" in snapshot:
        structure = snapshot["evidence_structure"]
        document["evidence_structure"] = structure
        document["active_medical_tree"] = _active_tree(snapshot)
        document["original_topology_role"] = "preserved_authoring_history"
    if "branch_geometry" in snapshot:
        document["branch_geometry_ref"] = {
            key: snapshot["branch_geometry"][key] for key in ("schema", "checksum")
        }
    body = json.dumps(document, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    sha = checksum(body)
    path = DIRECTORY / ("graph-artifact-" + sha[7:] + ".json")
    if len(body) > 4_194_304:
        raise ValueError("case_graph_size_limit")
    if store is None:
        store = LocalObjectStore(OBJECT_ROOTS["visible"])
    if path.exists():
        artifact = Artifact.model_validate(read_json(path))
    else:
        uri = store.promote(store.stage(body), sha)
        artifact = Artifact.model_validate(
            {
                "schema_version": "1.0",
                "contract_type": "Artifact",
                "asset_code": identity("ASSET", case["case_code"] + sha),
                "artifact_type_code": "ARTIFACT:CASE_EVIDENCE_GRAPH",
                "uri": uri,
                "checksum": sha,
                "media_type": "application/json",
                "qc_status": "PASS",
                "dimensions": [],
                "measurements": [],
                "derived_from": sorted(a.asset_code for a in refs),
                "research_only": True,
                "scope": {"scope_type": "CASE", "case_code": case["case_code"]},
                "laterality": case["input"]["laterality"],
                "coordinate_space": "NOT_APPLICABLE",
                "unit": None,
                "provenance": {
                    "actor": "PLANNER",
                    "created_at": datetime.now(UTC).isoformat(),
                    "generator_version": "case-evidence-graph-" + graph_version,
                    "source_codes": [a.asset_code for a in refs],
                    "input_checksums": [a.checksum for a in refs],
                    "knowledge_versions": knowledge_versions,
                },
            }
        )
        write_new(path, dump(artifact))
    if artifact.checksum != sha or store.read_bounded(artifact.uri, 4_194_304) != body:
        raise ValueError("case_graph_object_mismatch")
    repo.register_observation_bundle(
        (artifact,), identity("TASK", "case-graph:" + artifact.asset_code)
    )
    current = {a.asset_code: a for a in repo.get_case_bundle(case["case_code"]).artifacts}
    if current.get(artifact.asset_code) != artifact:
        raise ValueError("case_graph_readback_mismatch")
    return dump(artifact)


def read_view(case_code, graph_code, *, repo=None, store=None):
    """Exact historical export; does not create a case, infer facts or run tools."""
    from apps.evidence_graph_html import validate_snapshot
    from apps.patient_service_assembly import OBJECT_ROOTS

    if repo is None:
        repo = CaseEvidenceRepository(credential_boundary().resolve_target("runtime"))
    if store is None:
        store = LocalObjectStore(OBJECT_ROOTS["visible"])
    current = {a.asset_code: a for a in repo.get_case_bundle(case_code).artifacts}
    artifact = current.get(graph_code)
    if (
        artifact is None
        or artifact.artifact_type_code != "ARTIFACT:CASE_EVIDENCE_GRAPH"
        or artifact.scope.case_code != case_code
    ):
        raise ValueError("case_graph_scope_or_type_mismatch")
    body = store.read_bounded(artifact.uri, 4_194_304)
    if checksum(body) != artifact.checksum:
        raise ValueError("case_graph_object_mismatch")
    document = json.loads(body)
    if document.get("schema") == "case-evidence-graph/1.18.0":
        from apps.evidence_graph_records import read_manifest

        return read_manifest(document, case_code, current, repo, store, artifact)
    if document.get("schema") == "case-evidence-graph/1.17.0":
        from apps.glaucoma_vascular_followup import read_manifest

        return read_manifest(document, case_code, current, repo, store, artifact)
    if document.get("schema") in {
        "case-evidence-graph/1.13.0",
        "case-evidence-graph/1.14.0",
        "case-evidence-graph/1.15.0",
        "case-evidence-graph/1.16.0",
        "case-evidence-graph/1.19.0",
    }:
        from apps.evidence_graph_scores import read_manifest

        return read_manifest(document, case_code, current, repo, store, artifact)
    if document.get("schema") == "case-evidence-graph/1.12.0":
        from apps.evidence_graph_details import read_manifest

        return read_manifest(document, case_code, current, repo, store, artifact)
    if document.get("schema") not in {
        "case-evidence-graph/1.2.0",
        "case-evidence-graph/1.3.0",
        "case-evidence-graph/1.4.0",
        "case-evidence-graph/1.5.0",
        "case-evidence-graph/1.6.0",
        "case-evidence-graph/1.7.0",
        "case-evidence-graph/1.8.0",
        "case-evidence-graph/1.9.0",
        "case-evidence-graph/1.10.0",
        "case-evidence-graph/1.11.0",
    }:
        raise ValueError("legacy_case_graph_has_no_standalone_view")
    value = document["view_snapshot"]
    validate_snapshot(value)
    if document.get("schema") in {
        "case-evidence-graph/1.3.0",
        "case-evidence-graph/1.4.0",
        "case-evidence-graph/1.5.0",
        "case-evidence-graph/1.6.0",
        "case-evidence-graph/1.7.0",
        "case-evidence-graph/1.8.0",
        "case-evidence-graph/1.9.0",
        "case-evidence-graph/1.10.0",
        "case-evidence-graph/1.11.0",
    } and (
        not value.get("research_criteria")
        or document.get("research_criteria") != value["research_criteria"]
    ):
        raise ValueError("criterion_result_mismatch")
    if document.get("schema") in {
        "case-evidence-graph/1.4.0",
        "case-evidence-graph/1.5.0",
        "case-evidence-graph/1.6.0",
        "case-evidence-graph/1.7.0",
        "case-evidence-graph/1.8.0",
        "case-evidence-graph/1.9.0",
        "case-evidence-graph/1.10.0",
        "case-evidence-graph/1.11.0",
    } and (
        not value.get("model_selection")
        or document.get("model_selection") != value["model_selection"]
    ):
        raise ValueError("model_selection_result_mismatch")
    if document["schema"] in {
        "case-evidence-graph/1.7.0",
        "case-evidence-graph/1.8.0",
        "case-evidence-graph/1.9.0",
        "case-evidence-graph/1.10.0",
        "case-evidence-graph/1.11.0",
    }:
        structure = value.get("evidence_structure")
        if not structure or document.get("evidence_structure") != structure:
            raise ValueError("evidence_structure_binding_mismatch")
        expected_tree = _active_tree(value)
        if document.get("active_medical_tree") != expected_tree:
            raise ValueError("evidence_structure_active_tree_mismatch")
    if document["schema"] in {
        "case-evidence-graph/1.8.0",
        "case-evidence-graph/1.9.0",
        "case-evidence-graph/1.10.0",
        "case-evidence-graph/1.11.0",
    } and (
        not value.get("case_completion")
        or document.get("case_completion") != value["case_completion"]
    ):
        raise ValueError("completion_saved_binding_mismatch")
    if document["schema"] in {
        "case-evidence-graph/1.9.0",
        "case-evidence-graph/1.10.0",
        "case-evidence-graph/1.11.0",
    } and (
        not value.get("source_associations")
        or document.get("source_associations_ref")
        != {key: value["source_associations"][key] for key in ("schema", "checksum")}
    ):
        raise ValueError("association_saved_binding_mismatch")
    if document["schema"] in {"case-evidence-graph/1.10.0", "case-evidence-graph/1.11.0"} and (
        not value.get("branch_geometry")
        or document.get("branch_geometry_ref")
        != {key: value["branch_geometry"][key] for key in ("schema", "checksum")}
    ):
        raise ValueError("branch_saved_binding_mismatch")
    if document["schema"] == "case-evidence-graph/1.11.0":
        if value["source_associations"]["schema"] != "case-source-associations/1.1.0":
            raise ValueError("tool_evidence_version_mismatch")
        expected_nodes = [
            {
                "key": n["key"],
                "code": n["node_code"],
                "parent": n["parent"],
                "knowledge_ref": {
                    k: n["database"]["record"][k] for k in ("code", "version", "checksum")
                },
                "evidence": n["patient_evidence"],
            }
            for n in value["nodes"]
        ]
        if document["nodes"] != expected_nodes or document["accepted_validations"]:
            raise ValueError("tool_evidence_node_projection_mismatch")
    if document["case_code"] != case_code or value["case_evidence"]["case_code"] != case_code:
        raise ValueError("case_graph_scope_or_type_mismatch")
    if sorted(document["source_artifact_codes"]) != sorted(artifact.derived_from):
        raise ValueError("case_graph_source_mismatch")
    for code in artifact.derived_from:
        item = current.get(code)
        if (
            item is None
            or item.scope.case_code != case_code
            or item.checksum not in artifact.provenance.input_checksums
        ):
            raise ValueError("case_graph_source_mismatch")
    # This is a self-contained historical rendering, not a tool revalidation.
    # Upstream tools own different object roots (e.g. FairCLIP score objects).
    # Refs are checked against Runtime metadata; never guess their store root.
    return value
