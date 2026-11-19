"""Standalone, non-executing authoring viewer; exact DB readback, not publication."""

import argparse
import copy
import html
import json
from datetime import UTC, datetime
from pathlib import Path

from apps.glaucoma_egs_content import DESTINATION as EGS_DIRECTORY
from apps.glaucoma_egs_content import verify_page
from apps.glaucoma_evidence_bindings import DESTINATION as BINDING_DIRECTORY
from apps.glaucoma_evidence_bindings import dump, ref, verify_bindings, write_new
from apps.glaucoma_evidence_plan import DEFAULT, ROOT, checksum, load
from db.knowledge import models as k
from packages.object_store import LocalObjectStore
from packages.object_store import checksum as byte_checksum

DIRECTORY = ROOT / "docs/experiments/glaucoma-html-v2"
PREVIOUS_DIRECTORY = ROOT / "docs/experiments/glaucoma-html-v1"
TEMPLATE = ROOT / "apps/templates/evidence_graph.html"
CASE_TEMPLATE = ROOT / "apps/templates/case_evidence_graph.html"
READABLE_TEMPLATE = ROOT / "apps/templates/readable_evidence_graph.html"
READABLE_LAYOUT = ROOT / "apps/templates/evidence_graph_layout.js"


def selected_chunks(page):
    return [
        c
        for c in page["chunks"]
        if len(c["text"]) > 100
        and any(term in c["text"].casefold() for term in page["binding"]["terms"])
    ]


def extended_records(old_records, pages, *, version="1.4.0"):
    """Append a new version of original models with exact EGS source/chunk refs."""
    records, mapped = [], {}
    for raw in old_records:
        old = k.parse_record_json(k.KnowledgeKind(raw["kind"]), json.dumps(raw))
        key = old.code.removesuffix(":REQUIREMENT").split(":")[-1]
        relevant = [p for p in pages if key in p["binding"]["nodes"]]
        if isinstance(old, k.Template):
            relevant = pages
        fields = {
            name: getattr(old, name)
            for name in type(old).model_fields
            if name not in {"kind", "checksum", "version", "supersedes"}
        }
        for name, value in tuple(fields.items()):
            if isinstance(value, k.KnowledgeReference):
                fields[name] = mapped.get(value, value)
        if relevant:
            fields["source_refs"] = tuple(
                dict.fromkeys((*old.source_refs, *(ref(p["source_ref"]) for p in relevant)))
            )
            if hasattr(old, "binding_refs"):
                fields["binding_refs"] = tuple(
                    dict.fromkeys(
                        (
                            *(old.binding_refs or ()),
                            *(ref(c["ref"]) for p in relevant for c in selected_chunks(p)),
                        )
                    )
                )
        new = k.seal(type(old), version=version, supersedes=old.reference(), **fields)
        records.append(new)
        mapped[old.reference()] = new.reference()
    return tuple(records)


def read_snapshot(repo, *, persist=False):
    from apps.patient_service_assembly import OBJECT_ROOTS, check_object_root

    check_object_root(OBJECT_ROOTS["knowledge"], "knowledge")
    store = LocalObjectStore(OBJECT_ROOTS["knowledge"])
    bindings = json.loads((BINDING_DIRECTORY / "bindings.json").read_text())
    original = {key: value for key, value in bindings.items() if key != "new_template_ref"}
    verify_bindings(load(DEFAULT), original)
    pages = json.loads((EGS_DIRECTORY / "content.json").read_text())["pages"]
    for page in pages:
        verify_page(repo, store, page)
        if not selected_chunks(page) and not page["preview_blocks_not_indexed_evidence"]:
            raise ValueError("egs_substantive_passage_missing")
    old_records = json.loads((PREVIOUS_DIRECTORY / "records.json").read_text())
    records = extended_records(old_records, pages, version="1.5.0")
    if persist:
        # The sole write is an explicit, versioned Draft append in the original DB.
        repo.put_many(
            records, idempotency_key="glaucoma-egs-bindings:" + checksum([dump(r) for r in records])
        )
        write_new(DIRECTORY / "records.json", [dump(r) for r in records])
    snapshots = []
    for record in records:
        snapshot = repo.get(record.reference())
        if snapshot.record != record:
            raise ValueError("knowledge_export_record_mismatch")
        snapshots.append(dump(snapshot))
    # Rehydrate archived passages from the object store; sidecar text is not authority.
    source_chunks = copy.deepcopy(bindings["source_chunks"])
    for chunk in source_chunks:
        snapshot = repo.get(ref(chunk["ref"]))
        source = repo.get(snapshot.record.source_version_ref)
        body = store.read_bounded(snapshot.record.object_uri, 16384)
        if (
            byte_checksum(body) != chunk["text_checksum"]
            or body.decode() != chunk["text"]
            or snapshot.withdrawn
            or source.withdrawn
        ):
            raise ValueError("knowledge_export_passage_unavailable")
    nodes = copy.deepcopy(bindings["nodes"])
    for node in nodes:
        record = next(
            s
            for s in snapshots
            if s["record"]["kind"] == "node" and s["record"]["code"] == node["node_code"]
        )
        node["database"] = record
        node["node_type"] = record["record"]["definition"]["node_type"]
        node["egs_sources"] = [p["id"] for p in pages if node["key"] in p["binding"]["nodes"]]
        node["medical_score"] = {
            "value": None,
            "reason": "weight_policy_under_discussion",
            "propagation_enabled": False,
        }
    relationships = copy.deepcopy(bindings["relationships"])
    for edge in relationships:
        # Explicitly distinguish topical enrichment from direct support of a relation.
        edge["related_egs_pages_not_automatic_support"] = [
            p["id"]
            for p in pages
            if edge["parent"] in p["binding"]["nodes"] or edge["child"] in p["binding"]["nodes"]
        ]
    case = json.loads((BINDING_DIRECTORY / "case-observation.json").read_text())
    # This optional original observation is a historical receipt, not a fresh run.
    case["display_status"] = "historical_verified_receipt_not_reexecuted_or_current_authorization"
    value = {
        "schema": "evidence-graph-html-snapshot/1.0.0",
        "exported_at": datetime.now(UTC).isoformat(),
        "title": "青光眼证据需求图 · 节点与来源浏览器",
        "view_mode": "authoring_draft",
        "clinical_patient_graph": False,
        "database_readback": True,
        "records": snapshots,
        "nodes": nodes,
        "relationships": relationships,
        "egs_pages": pages,
        "source_chunks": source_chunks,
        "historical_pages": bindings["pages"],
        "engineering": json.loads((BINDING_DIRECTORY / "engineering-bindings.json").read_text()),
        "additional_algorithms": json.loads(
            (BINDING_DIRECTORY / "existing-algorithm-bindings.json").read_text()
        ),
        "historical_case": case,
        "score_policy": {
            "status": "proposal_not_applied",
            "numeric_weights": None,
            "llm_assessment": "Cited research judgment with rubric; not disease probability",
            "existing_engine": "03 AND/OR state propagation and counts, not probability fusion",
            "separate_layers": [
                "tool_raw_score",
                "source_relevance",
                "patient_support_refutation",
                "graph_completeness_conflict",
                "optional_calibrated_disease_probability",
            ],
        },
    }
    return {**value, "checksum": checksum(value)}


def validate_snapshot(value):
    from apps.evidence_graph_details import validate_view
    from apps.glaucoma_associations import validate as validate_associations
    from apps.glaucoma_branch_measurement import validate as validate_branch
    from apps.glaucoma_case_completion import validate as validate_completion
    from apps.glaucoma_evidence_structure import validate as validate_structure
    from apps.glaucoma_model_selection import validate as validate_selection
    from apps.glaucoma_research_criteria import validate as validate_criteria

    if value["checksum"] != checksum(
        {key: item for key, item in value.items() if key != "checksum"}
    ):
        raise ValueError("export_snapshot_checksum_mismatch")
    if "current_scoring" in value:
        from apps.evidence_graph_scores import unscored
        from packages.evidence_scoring.api import validate

        base = unscored(value)
        validate_snapshot(base)
        validate(base, value["current_scoring"])
        return
    if "evidence_records" in value:
        from apps.evidence_graph_records import validate_view as validate_records

        validate_snapshot(validate_records(value))
        return
    if "vascular_followup" in value:
        from apps.glaucoma_vascular_followup import validate_view as validate_vascular

        validate_snapshot(validate_vascular(value))
        return
    if "research_scoring" in value:
        from apps.evidence_graph_scores import unscored
        from packages.evidence_scoring.api import validate

        base = unscored(value)
        validate_snapshot(base)
        validate(base, value["research_scoring"])
        return
    validate_view(value)
    nodes = value["nodes"]
    keys = {n["key"] for n in nodes}
    if len(keys) != len(nodes) or len(nodes) > 128:
        raise ValueError("export_duplicate_or_excess_nodes")
    roots = [n for n in nodes if n["parent"] is None]
    if len(roots) != 1:
        raise ValueError("export_root_required")
    parents = {n["key"]: n["parent"] for n in nodes}
    for node in nodes:
        visited, cursor = set(), node["key"]
        while cursor is not None:
            if cursor in visited or cursor not in parents:
                raise ValueError("export_cycle_or_missing_parent")
            visited.add(cursor)
            cursor = parents[cursor]
    if any(e["parent"] not in keys or e["child"] not in keys for e in value["relationships"]):
        raise ValueError("export_relation_endpoint_missing")
    criteria_view = value
    if "source_associations" in value:
        from apps.glaucoma_associations import KEYS

        validate_associations(value)
        # The unchanged v1.3 criteria own the original node set; new association
        # nodes do not become extra rule results or change historical summaries.
        criteria_view = {**value, "nodes": [n for n in nodes if n["key"] not in KEYS]}
    validate_criteria(criteria_view)
    validate_selection(value)
    validate_structure(value)
    validate_completion(value)
    validate_associations(value)
    validate_branch(value)
    if "evidence_structure" in value and "display_explanations" in value:
        from apps.evidence_graph_explanation import explain

        if value["display_explanations"] != explain(value):
            raise ValueError("evidence_structure_explanation_mismatch")


def render_html(value):
    from apps.evidence_graph_explanation import explain
    from apps.evidence_graph_presentation import build as present

    validate_snapshot(value)
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    explanation = value.get("display_explanations") or explain(value)
    explanations = json.dumps(explanation, ensure_ascii=False, allow_nan=False)
    presentation = json.dumps(present(value, explanation), ensure_ascii=False, allow_nan=False)
    if len(raw.encode()) > 12_000_000:
        raise ValueError("export_size_limit")
    # Never let remote text terminate a script element; DOM uses textContent only.
    for text, escaped in (
        ("&", "\\u0026"),
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("\u2028", "\\u2028"),
        ("\u2029", "\\u2029"),
    ):
        raw = raw.replace(text, escaped)
        explanations = explanations.replace(text, escaped)
        presentation = presentation.replace(text, escaped)
    return (
        READABLE_TEMPLATE.read_text()
        .replace("__LAYOUT__", READABLE_LAYOUT.read_text())
        .replace("__TITLE__", html.escape(value["title"]))
        .replace("__EXPLANATIONS__", explanations)
        .replace("__PRESENTATION__", presentation)
        .replace("__SNAPSHOT__", raw)
    )


def write_export(snapshot, output, *, replace=False):
    rendered = render_html(snapshot)
    if not replace and (output.exists() or output.with_suffix(".json").exists()):
        raise ValueError("export_output_exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    if replace:
        for path in (output, output.with_suffix(".json")):
            if path.exists():
                raw = path.read_bytes()
                history = path.parent / "history" / (byte_checksum(raw)[7:] + path.suffix)
                history.parent.mkdir(exist_ok=True)
                if not history.exists():
                    with history.open("xb") as backup:
                        backup.write(raw)
                elif history.read_bytes() != raw:
                    raise ValueError("export_history_conflict")
    with output.open("w" if replace else "x") as stream:
        stream.write(rendered)
    with output.with_suffix(".json").open("w" if replace else "x") as stream:
        json.dump(snapshot, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    from apps.knowledge_source_tools import repository

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--from-case-graph", help="Export an exact saved graph Artifact from Runtime"
    )
    parser.add_argument("--case-code", help="Case owning the saved graph")
    parser.add_argument(
        "--snapshot", type=Path, help="Re-render a local snapshot, without DB access"
    )
    parser.add_argument(
        "--case-results", action="store_true", help="Read verified first-case observations"
    )
    parser.add_argument(
        "--research-criteria",
        action="store_true",
        help="Add source-bound structural research comparisons to verified case results",
    )
    parser.add_argument(
        "--save-case-graph", action="store_true", help="Persist the CASE observation graph Artifact"
    )
    parser.add_argument(
        "--replace-current",
        action="store_true",
        help="Archive previous export then update current HTML/JSON",
    )
    parser.add_argument(
        "--bind-egs", action="store_true", help="Append Draft 1.5.0 source bindings once"
    )
    args = parser.parse_args()
    if args.from_case_graph or args.snapshot:
        if args.case_results or args.save_case_graph or args.bind_egs or args.research_criteria:
            parser.error("saved/local export cannot rebuild or write database records")
        if args.from_case_graph and (not args.case_code or args.snapshot):
            parser.error("--from-case-graph requires --case-code and forbids --snapshot")
        if args.snapshot and args.case_code:
            parser.error("--snapshot does not select a database case")
        if args.snapshot:
            if args.snapshot.stat().st_size > 12_000_000:
                raise ValueError("export_size_limit")
            snapshot = json.loads(args.snapshot.read_text())
        else:
            from apps.glaucoma_case_storage import read_view

            snapshot = read_view(args.case_code, args.from_case_graph)
        write_export(snapshot, args.output, replace=args.replace_current)
        print(
            json.dumps(
                {
                    "html": str(args.output),
                    "nodes": len(snapshot["nodes"]),
                    "mode": "database_saved_graph" if args.from_case_graph else "local_snapshot",
                }
            )
        )
        return
    if args.case_code:
        parser.error("--case-code requires --from-case-graph")
    if args.save_case_graph and not args.case_results:
        parser.error("--save-case-graph requires --case-results")
    if args.research_criteria and not args.case_results:
        parser.error("--research-criteria requires --case-results")
    # Fail before any DB write if the output would overwrite a previous artifact.
    if not args.replace_current and (
        args.output.exists() or args.output.with_suffix(".json").exists()
    ):
        parser.error("output already exists; choose a new filename")
    snapshot = read_snapshot(repository("editor"), persist=args.bind_egs)
    if args.case_results:
        from apps.glaucoma_case_evidence import load_verified
        from apps.glaucoma_case_view import project

        snapshot = project(snapshot, load_verified())
        if args.research_criteria:
            from apps.glaucoma_criteria_v1_3 import apply
            from apps.glaucoma_evidence_structure import apply as organize
            from apps.glaucoma_model_selection import apply as select_model

            snapshot = organize(apply(select_model(snapshot)))
        if args.save_case_graph:
            from apps.glaucoma_case_storage import store_view

            validate_snapshot(snapshot)
            snapshot["case_graph_artifact"] = store_view(snapshot)
            snapshot["checksum"] = checksum({k: v for k, v in snapshot.items() if k != "checksum"})
    write_export(snapshot, args.output, replace=args.replace_current)
    print(
        json.dumps(
            {
                "html": str(args.output),
                "nodes": len(snapshot["nodes"]),
                "relationships": len(snapshot["relationships"]),
                "database_records": len(snapshot["records"]),
                "egs_pages": len(snapshot["egs_pages"]),
                "clinical_patient_graph": False,
                "saved_graph_artifact": snapshot.get("case_graph_artifact", {}).get("asset_code"),
                "checksum": snapshot["checksum"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
