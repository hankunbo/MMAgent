"""Persist/query typed observations and interpretations in the existing Runtime."""

import argparse
import copy
import json
from datetime import UTC, datetime
from pathlib import Path

from apps.evidence_graph_details import _read_json, _reference
from apps.glaucoma_evidence_plan import checksum as snapshot_checksum
from db.runtime.repository.case_pipeline import identity
from packages.contracts import Artifact
from packages.evidence_details.models import body, resolve
from packages.evidence_details.records import build, validate
from packages.object_store import checksum

VERSION = "case-evidence-graph/1.18.0"
BASE_VERSION = "case-evidence-graph/1.17.0"
RECORD_TYPE = "ARTIFACT:CASE_OBSERVATION_INTERPRETATION"


def attach(base, records):
    validate(base, records)
    value = copy.deepcopy(base)
    value["evidence_records"] = records
    value["checksum"] = snapshot_checksum({k: v for k, v in value.items() if k != "checksum"})
    return value


def validate_view(value):
    base = {k: v for k, v in value.items() if k not in {"evidence_records", "checksum"}}
    base["checksum"] = snapshot_checksum(base)
    validate(base, value["evidence_records"])
    return base


def read_manifest(document, case_code, current, repo, store, artifact):
    from apps.glaucoma_case_storage import read_view

    if (
        set(document) != {"schema", "case_code", "base", "records"}
        or document["case_code"] != case_code
    ):
        raise ValueError("record_manifest_scope")
    selected = {}
    for key, kind in [("base", "ARTIFACT:CASE_EVIDENCE_GRAPH"), ("records", RECORD_TYPE)]:
        r = document[key]
        a = current.get(r["code"])
        if (
            a is None
            or a.scope.case_code != case_code
            or a.artifact_type_code != kind
            or a.checksum != r["checksum"]
        ):
            raise ValueError("record_manifest_binding")
        selected[key] = a
    if (
        set(artifact.derived_from) != {a.asset_code for a in selected.values()}
        or set(selected["records"].derived_from) != {selected["base"].asset_code}
        or _read_json(selected["base"], store).get("schema") != BASE_VERSION
    ):
        raise ValueError("record_manifest_derivation")
    base = read_view(case_code, selected["base"].asset_code, repo=repo, store=store)
    return attach(base, _read_json(selected["records"], store))


def register(case_code, base_code, *, repo, store):
    """Only the persisted parent is accepted; callers cannot submit scores or facts."""
    from apps.glaucoma_case_storage import read_view

    current = {a.asset_code: a for a in repo.get_case_bundle(case_code).artifacts}
    original = current.get(base_code)
    if (
        original is None
        or original.artifact_type_code != "ARTIFACT:CASE_EVIDENCE_GRAPH"
        or original.scope.case_code != case_code
        or _read_json(original, store).get("schema") != BASE_VERSION
    ):
        raise ValueError("record_persisted_base_required")
    base = read_view(case_code, base_code, repo=repo, store=store)
    records = build(base)
    key = identity(
        "TASK", "observation-interpretation:" + case_code + base_code + records["checksum"]
    )
    prior = repo.read_observation_bundle(key)
    if prior is not None:
        if len(prior) != 2 or prior[0].artifact_type_code != RECORD_TYPE:
            raise ValueError("record_request_conflict")
        value = read_view(case_code, prior[1].asset_code, repo=repo, store=store)
        if value != attach(base, records):
            raise ValueError("record_retry_mismatch")
        return {
            "records": prior[0].model_dump(mode="json"),
            "graph": prior[1].model_dump(mode="json"),
            "request": key,
        }
    created = []
    attempted = False

    def artifact(value, kind, parents):
        raw = body(value)
        if len(raw) > 4_194_304:
            raise ValueError("record_object_bound")
        sha = checksum(raw)
        code = identity("ASSET", case_code + kind + sha)
        if code in current:
            # A record under another task must not silently adopt a different receipt.
            raise ValueError("record_unexpected_existing_artifact")
        stage = store.stage(raw)
        try:
            uri = store.promote(stage, sha)
        except BaseException:
            store.discard(stage)
            raise
        created.append(uri)
        value = original.model_dump(mode="json")
        value.update(
            asset_code=code,
            artifact_type_code=kind,
            uri=uri,
            checksum=sha,
            derived_from=sorted(a.asset_code for a in parents),
            dimensions=[],
            measurements=[],
            research_only=True,
        )
        value["provenance"].update(
            generator_version="observation-interpretation-normalization-1.0.0",
            created_at=datetime.now(UTC).isoformat(),
            source_codes=[a.asset_code for a in parents],
            input_checksums=[a.checksum for a in parents],
        )
        return Artifact.model_validate(value)

    try:
        record = artifact(records, RECORD_TYPE, [original])
        graph = artifact(
            {
                "schema": VERSION,
                "case_code": case_code,
                "base": _reference(original),
                "records": _reference(record),
            },
            "ARTIFACT:CASE_EVIDENCE_GRAPH",
            [original, record],
        )
        attempted = True
        repo.register_observation_bundle((record, graph), key)
    except ValueError:
        for uri in created:
            store.discard(uri)
        raise
    except BaseException:
        if not attempted:
            for uri in created:
                store.discard(uri)
        raise
    if read_view(case_code, graph.asset_code, repo=repo, store=store) != attach(base, records):
        raise ValueError("record_readback_mismatch")
    return {
        "records": record.model_dump(mode="json"),
        "graph": graph.model_dump(mode="json"),
        "request": key,
    }


def inspect(view, *, node=None, assessment=None, resolve_bindings=False):
    base = validate_view(view)
    records = view["evidence_records"]
    if node is not None and node not in records["node_index"]:
        raise ValueError("record_unknown_node")
    selected = [
        j
        for j in records["interpretations"]
        if (node is None or j["node"] == node)
        and (assessment is None or j["assessment_id"] == assessment)
    ]
    if assessment is not None and not selected:
        raise ValueError("record_unknown_assessment")
    ids = {j["observation_id"] for j in selected}
    result = {
        "schema_version": records["schema_version"],
        "case_code": records["case_code"],
        "task_scope": records["task_scope"],
        "scoring_status": records["scoring_status"],
        "observations": [o for o in records["observations"] if o["id"] in ids],
        "interpretations": selected,
        "inventory": records["inventory"],
    }
    if resolve_bindings:
        pointers = set()
        for row in [*result["observations"], *selected]:
            for name, value in row.items():
                if name.endswith("_bindings"):
                    pointers.update(r["pointer"] for r in value)
                elif name == "legacy_score_binding":
                    pointers.add(value["pointer"])
        result["resolved_bindings"] = {p: resolve(base, p) for p in sorted(pointers)}
    return result


def main():
    from apps.glaucoma_case_storage import read_view
    from apps.patient_service_assembly import OBJECT_ROOTS, check_object_root
    from apps.research_single_case import credential_boundary
    from db.runtime.repository.case_evidence import CaseEvidenceRepository
    from packages.object_store import LocalObjectStore

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["register", "inspect"])
    p.add_argument("--case-code", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--node")
    p.add_argument("--assessment")
    p.add_argument("--resolve-bindings", action="store_true")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.action == "register" and (
        a.node or a.assessment or a.resolve_bindings or a.output is None or a.output.exists()
    ):
        p.error("register requires a new receipt file and no query filters")
    if a.action == "inspect" and a.output:
        p.error("inspect is read-only and prints the selected records")
    check_object_root(OBJECT_ROOTS["visible"], "visible")
    repo = CaseEvidenceRepository(credential_boundary().resolve_target("runtime"))
    store = LocalObjectStore(OBJECT_ROOTS["visible"])
    if a.action == "register":
        result = register(a.case_code, a.graph, repo=repo, store=store)
        with a.output.open("x") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
        print(
            json.dumps(
                {"graph": result["graph"]["asset_code"], "records": result["records"]["asset_code"]}
            )
        )
    else:
        print(
            json.dumps(
                inspect(
                    read_view(a.case_code, a.graph, repo=repo, store=store),
                    node=a.node,
                    assessment=a.assessment,
                    resolve_bindings=a.resolve_bindings,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
