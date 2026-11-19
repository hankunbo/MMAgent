"""Versioned projection of saved evidence, not a new clinical graph or authorization.

No disease IDs, model names, current Memory/Knowledge lookup, or score formula belongs here.
Every derived record carries a JSON pointer into the supplied historical report.
"""

import copy
import json
from collections import deque

from packages.grounding.models import digest

from .evidence_tree import project as legacy_project
from .evidence_view import TYPES, disposition
from .presentation_contracts import (
    EvidenceSections,
    PresentationPacket,
    PresentationStructure,
)
from .presentation_contracts import (
    check as validate_envelope,
)

VERSION = "evidence-presentation/1.0.0"
STRUCTURE = "case-presentation-structure/1.0.0"
MAX_NODES = 10000
MAX_EDGES = 50000
RESTRICTED = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "credentials",
    "access_token",
    "refresh_token",
    "private_config",
    "hidden_labels",
    "ground_truth",
    "raw_note",
    "clinical_note",
    "environment",
    "env",
    "argv",
}


def pointer_part(value):
    return str(value).replace("~", "~0").replace("/", "~1")


def resolve(value, pointer):
    if pointer == "":
        return value
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("presentation_invalid_pointer")
    try:
        for part in pointer[1:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            value = value[int(part)] if isinstance(value, list) else value[part]
        return value
    except (KeyError, IndexError, ValueError, TypeError) as error:
        raise ValueError("presentation_unresolved_pointer:" + pointer) from error


def sanitize(value, pointer="", redacted=None, depth=0):
    """Do not expose credential/hidden-evaluation fields in unknown extensions or logs."""
    if depth > 80:
        raise ValueError("presentation_input_depth_limit")
    redacted = [] if redacted is None else redacted
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            path = pointer + "/" + pointer_part(key)
            if key.lower() in RESTRICTED or key.lower().endswith(("_api_key", "_password")):
                redacted.append(path)
                result[key] = "[restricted: not exported]"
            else:
                result[key] = sanitize(item, path, redacted, depth + 1)
        return result
    if isinstance(value, list):
        return [
            sanitize(v, pointer + "/" + str(i), redacted, depth + 1) for i, v in enumerate(value)
        ]
    if isinstance(value, float) and not float("-inf") < value < float("inf"):
        raise ValueError("presentation_nonfinite_number")
    return copy.deepcopy(value)


def origin(report, pointer, kind="saved_record_projection"):
    return dict(pointer=pointer, checksum=digest(resolve(report, pointer)), kind=kind)


def unique(rows, key, label):
    values = [r[key] for r in rows]
    if len(values) != len(set(values)):
        raise ValueError("presentation_duplicate_" + label)
    return {r[key]: r for r in rows}


def forest(nodes, edges):
    """Deterministic display forest; preserve shared/cyclic original relations separately."""
    by_id = {n["id"]: n for n in nodes}
    if len(by_id) != len(nodes) or len(nodes) > MAX_NODES or len(edges) > MAX_EDGES:
        raise ValueError("presentation_node_identity_or_size")
    adjacency = {k: [] for k in by_id}
    indegree = dict.fromkeys(by_id, 0)
    for e in edges:
        if e["source"] not in by_id or e["target"] not in by_id:
            raise ValueError("presentation_edge_endpoint")
        if e["relation"] == "contains":
            adjacency[e["source"]].append(e["target"])
            indegree[e["target"]] += 1
    queue = deque(k for k in by_id if indegree[k] == 0)
    visited = []
    while queue:
        key = queue.popleft()
        visited.append(key)
        for child in adjacency[key]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    cyclic = set(by_id) - set(visited)
    parents, warnings = {}, []
    if cyclic:
        warnings.append(dict(code="contains_cycle_or_downstream", nodes=sorted(cyclic)))
    for e in edges:
        if e["relation"] != "contains" or e["source"] in cyclic or e["target"] in cyclic:
            continue
        parents.setdefault(e["target"], e["source"])
    # All nodes remain reachable. Alternative parents remain in the relation table.
    for n in nodes:
        if n["id"] != "root":
            parents.setdefault(n["id"], "root")
    parents.pop("root", None)
    return parents, warnings


def project(report, *, source_checksum):
    if not isinstance(report, dict):
        raise ValueError("presentation_report_object")
    # The original adapter performs exact score and duplicate-assessment checks.
    old = legacy_project(report, input_checksum=source_checksum)
    redacted = []
    safe = sanitize(report, redacted=redacted)
    material = (safe.get("original_review") or {}).get("material", {})
    records = (safe.get("original_review") or {}).get("records", {})
    tasks_list = records.get("review_tasks", {}).get("tasks", [])
    tasks = unique(tasks_list, "assessment_id", "task")
    observations = {}
    for o in material.get("observations", []):
        code = o["reference"]["code"]
        if code in observations:
            raise ValueError("presentation_duplicate_observation")
        observations[code] = o
    facts = {n["id"]: n["payload"] for n in old["nodes"]}
    requirements = {r["id"]: r for r in safe.get("requirements", [])}
    rows = {r["id"]: r for r in old["scores"].get("reviews", [])}
    processes = unique(safe.get("process_evaluations", []), "id", "process")
    inventory = {r["id"]: r for r in safe.get("requirement_inventory", {}).get("rows", [])}
    checks, nodes, edges, coverage, diagnostics = {}, [], [], [], []
    added = set()

    def node(key, kind, title, pointer, **extras):
        if key in added:
            return
        if len(nodes) >= MAX_NODES:
            raise ValueError("presentation_node_limit")
        added.add(key)
        nodes.append(dict(id=key, type=kind, title=title, origin=origin(report, pointer), **extras))

    def edge(a, b, relation, pointer, kind="saved_reference"):
        if len(edges) >= MAX_EDGES:
            raise ValueError("presentation_edge_limit")
        value = dict(source=a, target=b, relation=relation, origin=origin(report, pointer, kind))
        if value not in edges:
            edges.append(value)

    def cover(pointer, destination):
        coverage.append(dict(pointer=pointer, destination=destination, status="accessible"))

    title = safe.get("title") or "病例目标与证据"
    node("root", "goal", title, "", state="病例总览", goal_summary=material.get("goal_summary"))
    if safe.get("observation_closure"):
        nodes[-1]["observation_closure"] = safe["observation_closure"]
        cover("/observation_closure", "root")
    req_indexes = {r["id"]: i for i, r in enumerate(report.get("requirements", []))}
    review_base = "/current_scoring/reviews" if "current_scoring" in report else "/scores/reviews"
    review_indexes = {r["id"]: i for i, r in enumerate(old["scores"].get("reviews", []))}
    task_indexes = {r["assessment_id"]: i for i, r in enumerate(tasks_list)}
    task_base = "/original_review/records/review_tasks/tasks"
    groups = {}
    for identifier in dict.fromkeys([*requirements, *rows]):
        req, row, task = (
            requirements.get(identifier, {}),
            rows.get(identifier),
            tasks.get(identifier, {}),
        )
        definition = task.get("definition") or {}
        key = "check:" + identifier
        pointer = (
            "/requirements/" + str(req_indexes[identifier])
            if identifier in req_indexes
            else review_base + "/" + str(review_indexes[identifier])
        )
        label = (
            req.get("title")
            or definition.get("title")
            or inventory.get(identifier, {}).get("title")
        )
        label = (
            label
            or req.get("examination")
            or definition.get("examination")
            or (row or {}).get("title")
            or identifier
        )
        obs = sanitize(facts.get(key + ":fact"))
        if isinstance(obs, dict) and obs.get("code") in observations:
            obs = observations[obs["code"]]
        status = dict(
            requirement=req.get("status"),
            acquisition=(obs or {}).get("status") if isinstance(obs, dict) else None,
            process=processes.get(identifier),
            catalog_definition=inventory.get(identifier, {}).get("definition_installed"),
            current_definition=definition.get("interpretation_status"),
            review_present=row is not None,
            direction=disposition(row),
        )
        # Explicit new semantics only; legacy interpretation/examination are not split by guessing.
        semantics = task.get("presentation_sections") or req.get("presentation_sections") or {}
        if semantics and semantics.get("schema_version") != "evidence-sections/1.0.0":
            raise ValueError("presentation_sections_version")
        if semantics:
            validate_envelope(EvidenceSections, semantics)
        check = dict(
            id=identifier,
            node_id=key,
            title=label,
            status=status,
            sections=dict(
                sources=sanitize(facts.get(key, {}).get("sources", [])),
                rationale=semantics.get("rationale"),
                acquisition=semantics.get("acquisition"),
                reference=semantics.get("reference"),
                legacy_definition=definition,
                observation=obs,
                score=sanitize(row),
                process=processes.get(identifier),
                review=sanitize(old["reader"]["interpretations"].get(identifier)),
                **(
                    {"direction_assessment": sanitize(req.get("direction_assessment"))}
                    if "direction_assessment" in req
                    else {}
                ),
                modalities=req.get("modality", []),
                method=task.get("method"),
                dependencies=(row or {}).get("dependency_paths", []),
            ),
            requirement=req,
            task=task,
            catalog=inventory.get(identifier),
            observation_code=(task.get("observation") or {}).get("code"),
        )
        checks[identifier] = check
        node(key, "requirement", label, pointer, check_id=identifier, state=status["direction"])
        cover(pointer, key)
        if identifier in task_indexes:
            cover(task_base + "/" + str(task_indexes[identifier]), key)
        if row is not None:
            rp = review_base + "/" + str(review_indexes[identifier])
            cover(rp, key)
            for path in row.get("dependency_paths", []):
                for ref in path:
                    rid = "resource:" + ref
                    node(rid, "execution", ref, rp + "/dependency_paths", state="已保存资源引用")
                    edge(rid, key, "depends_on", rp + "/dependency_paths")
            group = row.get("correlation_group")
            if group:
                gid = "group:" + group
                node(
                    gid,
                    "goal",
                    "同源证据组 · " + group,
                    rp + "/correlation_group",
                    state="原分组投影",
                )
                edge(
                    "root",
                    gid,
                    "contains",
                    rp + "/correlation_group",
                    "display_group_from_saved_attribute",
                )
                edge(
                    gid,
                    key,
                    "contains",
                    rp + "/correlation_group",
                    "display_group_from_saved_attribute",
                )
                groups[key] = gid
        if key not in groups:
            edge("root", key, "contains", pointer, "ungrouped_display_only")
        if status["catalog_definition"] is False and row is not None:
            diagnostics.append(dict(code="catalog_and_current_review_are_distinct", node=key))

    for key, result in (safe.get("native_tools") or {}).get("results", {}).items():
        path = "/native_tools/results/" + pointer_part(key)
        node("tool:" + key, "execution", key, path, state="本轮工具记录", payload=result)
        cover(path, "tool:" + key)

    # Owner-saved arbitrary organization overrides only navigation, never observation/scoring.
    structure = safe.get("presentation_structure")
    if structure:
        if structure.get("schema_version") != STRUCTURE:
            raise ValueError("presentation_structure_version")
        validate_envelope(PresentationStructure, structure)
        unique(structure.get("nodes", []), "id", "structure_node")
        edges = [e for e in edges if e["relation"] != "contains"]
        nodes = [n for n in nodes if not n["id"].startswith("group:")]
        added = {n["id"] for n in nodes}
        for i, n in enumerate(structure.get("nodes", [])):
            if n["id"] in added or not isinstance(n.get("type"), str):
                raise ValueError("presentation_structure_node_identity")
            kind = n["type"] if n["type"] in TYPES else "context"
            if kind != n["type"]:
                diagnostics.append(
                    dict(code="unknown_node_type", node=n["id"], original_type=n["type"])
                )
            node(
                n["id"], kind, n["title"], "/presentation_structure/nodes/" + str(i), state="原结构"
            )

        for i, e in enumerate(structure.get("edges", [])):
            edge(
                e["source"],
                e["target"],
                e["relation"],
                "/presentation_structure/edges/" + str(i),
                "owner_saved_relation",
            )
    elif "nodes" in safe:
        # Read the historical parent field, not a new disease profile or a title heuristic.
        historical = safe["nodes"]
        keys = [n.get("key") for n in historical]
        if all(isinstance(k, str) for k in keys) and len(set(keys)) == len(keys):
            for i, n in enumerate(historical):
                key = "historical:" + n["key"]
                node(key, "goal", n.get("title", n["key"]), "/nodes/" + str(i), state="历史原树")
            for i, n in enumerate(historical):
                key, parent = "historical:" + n["key"], n.get("parent")
                if parent in keys:
                    edge(
                        "historical:" + parent,
                        key,
                        "contains",
                        "/nodes/" + str(i) + "/parent",
                        "historical_saved_parent",
                    )
                elif parent is not None:
                    diagnostics.append(
                        dict(code="historical_missing_parent", node=key, parent=parent)
                    )
        diagnostics.append(dict(code="historical_tree_and_review_ids_not_guessed_merged"))
    else:
        diagnostics.append(dict(code="organization_projected_from_saved_groups_not_clinical_edges"))

    # Resource nodes are accessible via dependencies/appendix, not extra independent checks.
    nav_nodes = [
        n
        for n in nodes
        if n["type"] in {"goal", "requirement"}
        or n["origin"]["pointer"].startswith("/presentation_structure/nodes/")
    ]
    nav_ids = {n["id"] for n in nav_nodes}
    # Validate ALL endpoints including non-navigation relationships.
    if any(e["source"] not in added or e["target"] not in added for e in edges):
        raise ValueError("presentation_edge_endpoint")
    parents, warnings = forest(
        nav_nodes, [e for e in edges if e["source"] in nav_ids and e["target"] in nav_ids]
    )
    diagnostics.extend(warnings)
    for n in nav_nodes:
        n["parent"] = parents.get(n["id"])
    for field, value in safe.items():
        path = "/" + pointer_part(field)
        cover(path, "appendix:fields:" + field)
        if isinstance(value, list):
            for i in range(len(value)):
                cover(path + "/" + str(i), "appendix:fields:" + field)
    for path in redacted:
        coverage.append(dict(pointer=path, destination="appendix:coverage", status="restricted"))
    memory = (safe.get("native_tools") or {}).get("memory") or material.get("memory")
    destinations = {n["id"] for n in nodes} | {"appendix:coverage"}
    destinations.update("appendix:fields:" + k for k in safe)
    if any(c["destination"] not in destinations for c in coverage):
        raise ValueError("presentation_coverage_destination")
    packet = dict(
        schema_version=VERSION,
        source_checksum=source_checksum,
        report_content_checksum=digest(report),
        source_version=old["source_version"],
        sample=old["sample"],
        case_code=safe.get("result", {}).get("case_code"),
        input_checksum=(safe.get("native_tools") or {}).get("image_checksum"),
        scope="historical_read_only_not_current_authorization",
        types=TYPES,
        nodes=nodes,
        edges=edges,
        navigation=nav_nodes,
        checks=checks,
        scores=sanitize(old["scores"]),
        scores_checksum=digest(sanitize(old["scores"])),
        counts=old["counts"],
        confidence=old["confidence"],
        removals=sanitize(old["removals"]),
        memory=dict(
            saved=memory,
            current_authorization=False,
            semantics="saved_candidates_or_use_record_not_new_retrieval",
        ),
        appendices=safe,
        diagnostics=diagnostics,
        coverage=dict(complete=True, entries=coverage, restricted=redacted),
    )
    return packet


def validate_packet(packet):
    validate_envelope(PresentationPacket, packet)
    if packet.get("schema_version") != VERSION:
        raise ValueError("presentation_packet_version")
    if packet.get("scope") != "historical_read_only_not_current_authorization":
        raise ValueError("presentation_packet_scope")
    if digest(packet["scores"]) != packet["scores_checksum"]:
        raise ValueError("presentation_packet_score_binding")
    ids = {n["id"] for n in packet["nodes"]}
    if "root" not in ids or len(ids) != len(packet["nodes"]):
        raise ValueError("presentation_packet_node_identity")
    forest(packet["nodes"], packet["edges"])
    nav = packet["navigation"]
    nav_ids = {n["id"] for n in nav}
    if not nav_ids <= ids or len(nav_ids) != len(nav) or "root" not in nav_ids:
        raise ValueError("presentation_packet_navigation")
    expected, _ = forest(
        nav, [e for e in packet["edges"] if e["source"] in nav_ids and e["target"] in nav_ids]
    )
    if any(n.get("parent") != expected.get(n["id"]) for n in nav):
        raise ValueError("presentation_packet_navigation_binding")
    rows = {r["id"]: r for r in packet["scores"].get("reviews", [])}
    if packet["confidence"] != packet["scores"].get("root", {}).get("summary", {}).get(
        "confidence", {}
    ).get("value"):
        raise ValueError("presentation_packet_confidence_binding")
    expected_counts = dict(
        reviewed=len(rows),
        numeric=sum(r.get("judgment", {}).get("value") is not None for r in rows.values()),
        nonzero_contributors=sum(
            bool(r.get("support") or r.get("refutation")) for r in rows.values()
        ),
        neutral=sum(r.get("judgment", {}).get("value") == 0 for r in rows.values()),
        unassessed=sum(r.get("judgment", {}).get("value") is None for r in rows.values()),
    )
    if any(packet["counts"][k] != v for k, v in expected_counts.items()):
        raise ValueError("presentation_packet_counts_binding")
    for key, check in packet["checks"].items():
        if check["id"] != key or check["node_id"] not in ids:
            raise ValueError("presentation_packet_check_binding")
        if check["sections"]["score"] != rows.get(key):
            raise ValueError("presentation_packet_check_score")
    destinations = (
        ids | {"appendix:coverage"} | {"appendix:fields:" + k for k in packet["appendices"]}
    )
    if not packet["coverage"]["complete"] or any(
        c["destination"] not in destinations for c in packet["coverage"]["entries"]
    ):
        raise ValueError("presentation_packet_coverage")
    # Serialization also rejects non-finite values; packets confer no authenticity authority.
    json.dumps(packet, allow_nan=False)
    return packet
