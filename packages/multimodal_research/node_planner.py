"""Standalone proposition/input planner with procedure-only Memory reuse.

This module creates an initial evidence-tree *plan*.  It never dispatches a
patient tool, calls a model, searches the web, assigns a medical score, or
copies a historical patient's observations.  Historical Memory records are
used only as procedure hints; current Knowledge definitions and current input
owners remain authoritative.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from packages.grounding.models import digest

from .requirement_catalog import disposition, read as read_catalog

VERSION = "multimodal-planner-plan/1.1.0"


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).absolute().read_bytes())
    if not isinstance(value, dict):
        raise ValueError("planner_json_object_required")
    return value


def _load_memory(path: str | Path | None) -> tuple[dict[str, Any], ...]:
    if path is None:
        return ()
    value = _read_json(path)
    rows = value.get("records", value if isinstance(value, list) else ())
    if not isinstance(rows, list):
        raise ValueError("planner_memory_records")
    result = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("planner_memory_record")
        # A caller supplied index is a read-only export, never an authorization
        # marker.  The original source owner must still be rechecked by runtime.
        if row.get("kind") not in {"tool_experience", "graph_snapshot", "recovery_experience"}:
            continue
        result.append(row)
    return tuple(result)


def _read_installed_memory(checks: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Read the installed Memory owner without creating a second store.

    This is explicitly read-only.  A missing/empty archive is a valid result;
    connection or owner errors are surfaced instead of being converted into an
    empty successful plan.
    """
    try:
        from apps.case_memory_store import PRIVATE, assembly

        if not PRIVATE.exists():
            return ()
        reader, _ = assembly("reader")
        keys = set()
        for check in checks:
            for modality in {check["modality"], *(v["modality"] for v in check.get("input_variants", ()))}:
                keys.update(reader.procedure_keys(targets=(check["id"],), modality=modality, limit=3))
        records = []
        for key in sorted(keys):
            records.append(reader.read(key).record.model_dump(mode="json"))
        return tuple(records)
    except Exception as error:
        raise ValueError("planner_installed_memory_read_failed") from error


def _load_qualitative(path: str | Path | None, input_checksum: str) -> dict[str, Any]:
    if path is None:
        return {"present": False, "components": {}}
    value = _read_json(path)
    bound = value.get("input_checksum") or value.get("packet_checksum")
    if bound is not None and bound != input_checksum:
        raise ValueError("planner_qualitative_input_binding")
    rows = value.get("components", ())
    if isinstance(rows, dict):
        components = rows
    elif isinstance(rows, list):
        components = {
            str(row.get("component_id")): row
            for row in rows
            if isinstance(row, dict) and row.get("component_id")
        }
    else:
        components = {}
    return {"present": True, "components": components, "checksum": digest(value)}


def _memory_candidates(records: tuple[dict[str, Any], ...], check: dict[str, Any], component) -> list[dict[str, Any]]:
    """Match procedure hints without using case values, scores, or dataset names."""
    modality = check["modality"]
    representation = set(check["input_capabilities"]["representations"])
    target = set(check.get("goals", ()))
    candidates = []
    for record in records:
        detail = record.get("detail", {})
        if record.get("kind") not in {"tool_experience", "graph_snapshot"}:
            continue
        if detail.get("target") not in {check["id"], *target}:
            continue
        if modality not in str(detail.get("modality", "")).split("+"):
            continue
        steps = detail.get("steps", [])
        if not isinstance(steps, list):
            continue
        if component is not None and component.get("representation") not in representation:
            continue
        candidates.append(record)
    return candidates


def _qualitative_scope(qualitative: dict[str, Any], component_id: str) -> bool:
    row = qualitative.get("components", {}).get(component_id, {})
    if not isinstance(row, dict):
        return False
    # Only explicit saved scope/structure qualifies here.  Modality or a file
    # name alone never proves that the anatomy is visible.
    for key in ("visual_scope", "anatomical_region", "anatomy", "content"):
        value = row.get(key)
        if value not in (None, "", "unknown", "not_assessed", "not_structured"):
            return True
    return row.get("status") in {"confirmed", "completed", "visual_analysis_completed"}


def _component_view(capability) -> dict[str, Any]:
    return {
        "component_id": capability.component_id,
        "component_checksum": capability.component_checksum,
        "representation": capability.representation,
        "modality": capability.modality,
        "concept": capability.concept,
        "input_status": capability.input_status,
        "unit": capability.unit,
        "unit_basis": capability.unit_basis,
        "laterality": capability.laterality,
        "visit": capability.visit,
        "acquisition_fraction": capability.acquisition_fraction,
        "abilities": [a.name for a in capability.abilities if a.status != "unavailable"],
    }


def _compatible_components(check: dict[str, Any], capabilities) -> tuple[list[Any], list[dict[str, Any]]]:
    needs = check["input_capabilities"]
    accepted, excluded = [], []
    modalities = {check["modality"], *(v["modality"] for v in check.get("input_variants", ())) }
    # The catalog's paired_fundus question is fulfilled by two explicitly
    # bound color_fundus components.  This is a typed scope bridge, not a
    # dataset-name heuristic or a claim that one eye can substitute for two.
    if needs.get("scope") == "paired_eyes" and check["modality"] == "paired_fundus":
        modalities.add("color_fundus")
    for component in capabilities.components:
        if component.modality not in modalities or component.input_status == "withheld":
            continue
        reason = None
        if component.representation not in needs["representations"]:
            reason = "representation_mismatch"
        elif needs.get("concepts") and component.concept not in needs["concepts"]:
            reason = "concept_mismatch"
        if reason:
            excluded.append({"component": component.component_id, "reason": reason})
        else:
            accepted.append(component)
    return accepted, excluded


def _node_id(prefix: str, value: str) -> str:
    return prefix + ":" + value.replace(" ", "_")


def build_plan(*, manifest, capabilities, task, catalog, qualitative=None,
               memory_records=(), input_path=None, qualitative_path=None,
               memory_source="none"):
    """Build a deterministic initial tree from owner-validated projections."""
    if capabilities.input_checksum != manifest.checksum:
        raise ValueError("planner_capability_input_binding")
    qualitative = qualitative or {"present": False, "components": {}}
    target = task.target_code
    checks = [c.model_dump(mode="json") for c in catalog.checks if target in c.goals]
    groups = {g.id: g.model_dump(mode="json") for g in catalog.nodes}
    nodes: list[dict[str, Any]] = [{
        "id": "root:" + target,
        "type": "proposition",
        "target": target,
        "status": "selected",
        "source": "task_registry",
        "dependencies": [],
    }]
    edges: list[dict[str, str]] = []
    decisions: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    pending_conditions_index: list[dict[str, Any]] = []
    selected = 0
    conditional = 0
    rejected = 0
    group_ids: set[str] = set()

    def add_group(group_id: str):
        if group_id in group_ids:
            return
        group = groups[group_id]
        parent = group["parent"]
        if parent != "root" and parent in groups:
            add_group(parent)
            parent_id = _node_id("group", parent)
        else:
            parent_id = "root:" + target
        nodes.append({"id": _node_id("group", group_id), "type": "group",
                      "title": group["title"], "status": "selected",
                      "source": "requirement_catalog", "dependencies": []})
        edges.append({"from": parent_id, "to": _node_id("group", group_id), "relation": "AND"})
        group_ids.add(group_id)

    for check in checks:
        add_group(check["parent"])
        matched, excluded = _compatible_components(check, capabilities)
        scope = check["input_capabilities"]["scope"]
        if scope == "paired_eyes":
            eyes = {c.laterality for c in matched}
            if not {"LEFT", "RIGHT"} <= eyes:
                matched = matched[:]
                pair_reason = "paired_eye_input_incomplete"
            else:
                pair_reason = None
        else:
            pair_reason = None
        pending_conditions: list[str] = []
        gap_details: dict[str, Any] = {}
        if not matched:
            status, reason = "missing_input", "no_compatible_input_representation"
            rejected += 1
            gap_details = {"excluded": excluded}
            components = []
        else:
            components = matched
            if pair_reason:
                status, reason = "conditional", pair_reason
                conditional += 1
            else:
                # Planning answers “can this method be scheduled with the
                # supplied representation?”.  It must not answer the later
                # clinical questions “is the anatomy visible?”, “is quality
                # sufficient?”, or “is the source unit verified?”.  Those are
                # execution/review conditions and are retained explicitly so
                # the tool and reviewer can resolve them after dispatch.
                if scope == "visible_region_pending" and not any(
                    _qualitative_scope(qualitative, c.component_id) for c in components
                ):
                    pending_conditions.append("qualitative_scope_pending")
                if any(c.input_status == "partial" for c in components):
                    pending_conditions.append("partial_input")
                if scope == "source_measurement" and any(c.unit is None for c in components):
                    pending_conditions.append("source_unit_and_meaning_required")
                status, reason = (
                    "selected",
                    "input_compatible_pending_execution_checks"
                    if pending_conditions else "input_and_scope_compatible",
                )
                selected += 1
        node_id = _node_id("check", check["id"])
        bindings = [_component_view(c) for c in components]
        memory = []
        for component in components[:1]:
            memory = _memory_candidates(memory_records, check, _component_view(component))
            if memory:
                break
        method_source = "memory_procedure_hint" if memory else "current_owner_baseline"
        node = {
            "id": node_id,
            "type": "examination",
            "check_id": check["id"],
            "parent": check["parent"],
            "title": check["title"],
            "method": check["method"],
            "modality": check["modality"],
            "status": status,
            "decision_reason": reason,
            "pending_conditions": pending_conditions,
            "bindings": bindings,
            "method_source": method_source,
            "memory_candidates": [r.get("key") for r in memory[:3]],
            "execution_authorized": False,
            "clinical_evidence": False,
            "dependencies": [],
            "source": {"catalog": check["source"], "quote": check["quote"]},
        }
        nodes.append(node)
        edges.append({"from": _node_id("group", check["parent"]), "to": node_id, "relation": "AND"})
        # A Memory hint contributes procedure steps only.  Patient artifact
        # codes, old values, observations and scores are deliberately omitted.
        if memory:
            previous = None
            for index, step in enumerate(memory[0].get("detail", {}).get("steps", ())):
                if not isinstance(step, dict):
                    continue
                step_id = _node_id("method", check["id"] + ":" + str(index))
                nodes.append({
                    "id": step_id,
                    "type": "method",
                    "operation": step.get("operation"),
                    "tool": step.get("tool"),
                    "version": step.get("version"),
                    "input_roles": step.get("input_roles", ()),
                    "output_roles": step.get("output_roles", ()),
                    "status": "selected" if status == "selected" else "conditional",
                    "source_memory_key": memory[0].get("key"),
                    "execution_authorized": False,
                })
                if previous is not None:
                    edges.append({"from": previous, "to": step_id, "relation": "AND"})
                previous = step_id
            if previous is not None:
                edges.append({"from": previous, "to": node_id, "relation": "AND"})
        decisions.append({
            "node_id": node_id,
            "check_id": check["id"],
            "decision": "reuse_and_rebind" if memory and status in {"selected", "conditional"}
            else status,
            "reason": reason,
            "memory_keys": [r.get("key") for r in memory[:3]],
            "bindings": [c["component_id"] for c in bindings],
            "excluded_components": excluded,
        })
        if status != "selected":
            gaps.append({"check": check["id"], "state": status, "reason": reason,
                         **gap_details})
        if pending_conditions:
            # Follow-up checks are kept in their own ledger.  They are not
            # gaps: a downstream executor must see them, but must not mistake
            # them for missing input and skip the selected method.
            pending_conditions_index.append({
                "check": check["id"],
                "conditions": pending_conditions,
            })

    plan = {
        "schema_version": VERSION,
        "query": {
            "input_checksum": manifest.checksum,
            "task_checksum": task.checksum,
            "target": target,
            "input_path": str(input_path) if input_path else None,
            "qualitative_path": str(qualitative_path) if qualitative_path else None,
            "memory_source": memory_source,
        },
        "tree": {"root": "root:" + target, "nodes": nodes, "edges": edges},
        "reuse_decisions": decisions,
        "gaps": gaps,
        "pending_conditions": pending_conditions_index,
        "counts": {
            "catalog_checks": len(checks),
            "selected": selected,
            "conditional": conditional,
            "rejected_or_missing": rejected,
            "memory_hints": sum(bool(d["memory_keys"]) for d in decisions),
        },
        "qualification": "initial_plan_only_no_patient_execution_or_medical_score",
        "execution_authorized": False,
        "api_requests": 0,
        "model_runs": 0,
        "database_writes": 0,
    }
    plan["checksum"] = digest(plan)
    return plan


def run(*, input_path, task_id, output, registry, catalog_profile,
        qualitative_path=None, memory_index=None, installed_memory=False,
        scope_input=None):
    from packages.dataset_adapters.multimodal.capabilities import from_preparation
    from packages.dataset_adapters.multimodal.preparation import verify
    from packages.dataset_adapters.multimodal.tasks import load_intent

    if scope_input:
        from .scope import read as read_scope

        scope = read_scope(scope_input)
        preparations = [row["input"] for row in scope["inputs"]]
        manifests = [verify(path) for path in preparations]
        if {m.dataset for m in manifests} != {scope["dataset"]}:
            raise ValueError("planner_scope_current_input_dataset")
        all_components = []
        for row, path in zip(scope["inputs"], preparations):
            eye = row["eye"]
            for component in from_preparation(path).components:
                # Each single-eye adapter uses local IDs such as
                # ``visit:0:fundus``.  Scope composition qualifies them by eye
                # so OD and OS cannot overwrite one another in a plan.
                all_components.append(
                    component.model_copy(update={"component_id": eye + ":" + component.component_id})
                )
        manifest = SimpleNamespace(checksum=scope["checksum"], dataset=scope["dataset"])
        capabilities = SimpleNamespace(input_checksum=scope["checksum"], components=tuple(all_components))
        input_path = scope_input
    else:
        if not input_path:
            raise ValueError("planner_input_required")
        manifest = verify(input_path)
        capabilities = from_preparation(input_path)
    task, _ = load_intent(task_id, registry)
    profile = _read_json(catalog_profile)
    reference = profile.get("requirement_catalog")
    if not isinstance(reference, dict):
        raise ValueError("planner_catalog_reference")
    catalog = read_catalog(reference)
    qualitative = _load_qualitative(qualitative_path, manifest.checksum)
    checks = [c.model_dump(mode="json") for c in catalog.checks if task.target_code in c.goals]
    records = _load_memory(memory_index)
    if installed_memory:
        records = tuple(dict(row) for row in (*records, *_read_installed_memory(checks)))
    plan = build_plan(
        manifest=manifest, capabilities=capabilities, task=task, catalog=catalog,
        qualitative=qualitative, memory_records=records, input_path=input_path,
        qualitative_path=qualitative_path,
        memory_source=("installed_owner" if installed_memory else "exported_index" if memory_index else "none"),
    )
    out = Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError("planner_output_must_be_new")
    out.mkdir(parents=True, mode=0o700)
    with (out / "plan.json").open("x") as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    with (out / "reuse-decisions.json").open("x") as stream:
        json.dump({"schema_version": VERSION, "decisions": plan["reuse_decisions"],
                   "checksum": digest(plan["reuse_decisions"])}, stream,
                  ensure_ascii=False, indent=2)
        stream.write("\n")
    rows = []
    for node in plan["tree"]["nodes"]:
        label = node.get("check_id") or node.get("title") or node.get("operation") or node["id"]
        rows.append(
            "<li><code>" + escape(str(node["id"])) + "</code> "
            + escape(str(label)) + " <b>" + escape(str(node.get("status", "")))
            + "</b></li>"
        )
    html = (
        "<!doctype html><meta charset='utf-8'><title>Initial Planner Tree</title>"
        "<h1>Initial Planner Tree</h1><p>" + escape(plan["query"]["target"])
        + "</p><ul>" + "".join(rows) + "</ul>"
        "<p>Execution authorized: false; no API/model/database operations.</p>"
    )
    (out / "plan.html").write_text(html, encoding="utf-8")
    return {"output": str(out), "checksum": plan["checksum"], **plan["counts"],
            "execution_authorized": False, "api_requests": 0, "database_writes": 0}
