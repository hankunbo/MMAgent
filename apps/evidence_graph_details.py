"""Append/read judgment traces through existing Runtime Artifact/Audit ports."""

import argparse
import copy
import json
from pathlib import Path

from apps.glaucoma_evidence_plan import checksum as snapshot_checksum
from db.runtime.repository.case_pipeline import identity
from packages.contracts import Artifact
from packages.evidence_details.models import FACETS, body
from packages.evidence_details.projection import build, validate
from packages.object_store import checksum

VERSION = "case-evidence-graph/1.12.0"
PROFILE_TYPE = "ARTIFACT:EVIDENCE_DETAIL_PROFILES"
TRACE_TYPE = "ARTIFACT:CASE_JUDGMENT_DETAILS"


def base_view(value):
    base = {k: v for k, v in value.items() if k not in {"judgment_details", "checksum"}}
    return {**base, "checksum": snapshot_checksum(base)}


def attach(base, profiles, ledger):
    validate(base, profiles, ledger)
    value = copy.deepcopy(base)
    value["judgment_details"] = {"profiles": profiles, "ledger": ledger, "facet_labels": FACETS}
    value["checksum"] = snapshot_checksum({k: v for k, v in value.items() if k != "checksum"})
    return value


def validate_view(value):
    if "judgment_details" not in value:
        return
    detail = value["judgment_details"]
    if set(detail) != {"profiles", "ledger", "facet_labels"} or detail["facet_labels"] != FACETS:
        raise ValueError("detail_view_schema_mismatch")
    validate(base_view(value), detail["profiles"], detail["ledger"])


def _reference(a):
    return {"code": a.asset_code, "checksum": a.checksum}


def _read_json(a, store):
    raw = store.read_bounded(a.uri, 4_194_304)
    if checksum(raw) != a.checksum:
        raise ValueError("detail_object_checksum_mismatch")
    return json.loads(raw)


def read_manifest(document, case_code, current, repo, store, artifact):
    from apps.glaucoma_case_storage import read_view

    if (
        set(document) != {"schema", "case_code", "base", "profiles", "trace", "input_checksum"}
        or document["case_code"] != case_code
    ):
        raise ValueError("detail_manifest_scope_mismatch")
    selected = {}
    for key, expected in [
        ("base", "ARTIFACT:CASE_EVIDENCE_GRAPH"),
        ("profiles", PROFILE_TYPE),
        ("trace", TRACE_TYPE),
    ]:
        ref = document[key]
        a = current.get(ref["code"])
        if (
            a is None
            or a.scope.case_code != case_code
            or a.artifact_type_code != expected
            or a.checksum != ref["checksum"]
        ):
            raise ValueError("detail_attachment_scope_or_checksum_mismatch")
        selected[key] = a
    if set(artifact.derived_from) != {a.asset_code for a in selected.values()}:
        raise ValueError("detail_manifest_derivation_mismatch")
    # Only one extension hop. No recursion/cycles or reinterpretation of old bytes.
    raw_base = _read_json(selected["base"], store)
    if raw_base.get("schema") == VERSION or "judgment_details" in raw_base.get("view_snapshot", {}):
        raise ValueError("detail_recursive_base_forbidden")
    base = read_view(case_code, selected["base"].asset_code, repo=repo, store=store)
    if base["case_evidence"]["input"]["checksum"] != document["input_checksum"]:
        raise ValueError("detail_manifest_input_mismatch")
    for key in ("profiles", "trace"):
        expected_sources = {selected["base"].asset_code}
        if key == "trace":
            expected_sources.add(selected["profiles"].asset_code)
        if set(selected[key].derived_from) != expected_sources:
            raise ValueError("detail_attachment_derivation_mismatch")
    return attach(
        base, _read_json(selected["profiles"], store), _read_json(selected["trace"], store)
    )


def register(case_code, base_code, *, repo, store, request_key=None):
    """Inputs are rebuilt from persisted base; no caller-provided judgment ledger."""
    from apps.evidence_detail_profiles import collect
    from apps.glaucoma_case_storage import read_view

    base = read_view(case_code, base_code, repo=repo, store=store)
    if "judgment_details" in base:
        raise ValueError("detail_use_original_base")
    key = identity("TASK", "judgment-details:" + (request_key or case_code + base_code + VERSION))
    previous = repo.read_observation_bundle(key)
    if previous is not None:
        if len(previous) != 3 or any(a.scope.case_code != case_code for a in previous):
            raise ValueError("detail_request_scope_conflict")
        profile, trace, graph = previous
        manifest = _read_json(graph, store)
        if manifest.get("schema") != VERSION or manifest.get("base", {}).get("code") != base_code:
            raise ValueError("detail_request_base_conflict")
        saved = read_view(case_code, graph.asset_code, repo=repo, store=store)
        return {
            "graph": graph.model_dump(mode="json"),
            "profiles": profile.model_dump(mode="json"),
            "trace": trace.model_dump(mode="json"),
            "coverage": saved["judgment_details"]["ledger"]["coverage"],
        }
    profiles = collect(base)
    ledger = build(base, profiles)
    validate(base, profiles, ledger)
    current = {a.asset_code: a for a in repo.get_case_bundle(case_code).artifacts}
    original = current[base_code]
    created, bundle = [], []
    write_attempted = False

    def artifact(value, kind, sources):
        data = body(value)
        if len(data) > 4_194_304:
            raise ValueError("detail_object_size_limit")
        sha = checksum(data)
        code = identity("ASSET", case_code + kind + sha)
        existing = current.get(code)
        if existing:
            if (
                existing.checksum != sha
                or existing.artifact_type_code != kind
                or store.read_bounded(existing.uri, 4_194_304) != data
            ):
                raise ValueError("detail_existing_artifact_conflict")
            return existing
        stage = store.stage(data)
        try:
            uri = store.promote(stage, sha)
        except BaseException:
            store.discard(stage)
            raise
        created.append(uri)
        raw = original.model_dump(mode="json")
        raw.update(
            asset_code=code,
            artifact_type_code=kind,
            uri=uri,
            checksum=sha,
            derived_from=sorted(a.asset_code for a in sources),
            measurements=[],
            dimensions=[],
        )
        raw["provenance"].update(
            generator_version="judgment-binding-1.0.0",
            created_at=profiles["assembly"]["recorded_at"],
            source_codes=[a.asset_code for a in sources],
            input_checksums=[a.checksum for a in sources],
        )
        return Artifact.model_validate(raw)

    try:
        profile = artifact(profiles, PROFILE_TYPE, [original])
        trace = artifact(ledger, TRACE_TYPE, [original, profile])
        document = {
            "schema": VERSION,
            "case_code": case_code,
            "base": _reference(original),
            "profiles": _reference(profile),
            "trace": _reference(trace),
            "input_checksum": ledger["input_checksum"],
        }
        graph = artifact(document, "ARTIFACT:CASE_EVIDENCE_GRAPH", [original, profile, trace])
        bundle = [profile, trace, graph]
        write_attempted = True
        repo.register_observation_bundle(tuple(bundle), key)
    except ValueError:
        # Repository ValueErrors leave its transaction context rolled back.
        for uri in created:
            store.discard(uri)
        raise
    except BaseException:
        if not write_attempted:
            for uri in created:
                store.discard(uri)
        # A transport failure after commit is ambiguous: retain only our objects
        # for exact retry/readback, never delete potentially committed evidence.
        raise
    saved = read_view(case_code, graph.asset_code, repo=repo, store=store)
    if saved != attach(base, profiles, ledger):
        raise ValueError("detail_readback_mismatch")
    return {
        "graph": graph.model_dump(mode="json"),
        "profiles": profile.model_dump(mode="json"),
        "trace": trace.model_dump(mode="json"),
        "coverage": ledger["coverage"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["register", "inspect"])
    parser.add_argument("--case-code", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--node")
    parser.add_argument("--tool")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--request-key", help="new explicit key only when refreshing the saved profile snapshot"
    )
    args = parser.parse_args()
    from apps.glaucoma_case_storage import read_view
    from apps.patient_service_assembly import OBJECT_ROOTS, check_object_root
    from apps.research_single_case import credential_boundary
    from db.runtime.repository.case_evidence import CaseEvidenceRepository
    from packages.object_store import LocalObjectStore

    check_object_root(OBJECT_ROOTS["visible"], "visible")
    repo = CaseEvidenceRepository(credential_boundary().resolve_target("runtime"))
    store = LocalObjectStore(OBJECT_ROOTS["visible"])
    view = read_view(args.case_code, args.graph, repo=repo, store=store)
    if args.action == "register":
        if args.output is None or args.output.exists():
            parser.error("register requires a new --output receipt path")
        result = register(
            args.case_code, args.graph, repo=repo, store=store, request_key=args.request_key
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(
            json.dumps(
                {"graph": result["graph"]["asset_code"], "coverage": result["coverage"]},
                ensure_ascii=False,
            )
        )
        return
    details = view["judgment_details"]
    if args.node:
        ids = details["ledger"]["node_index"][args.node]
        result = [a for a in details["ledger"]["assessments"] if a["id"] in ids]
    elif args.tool:
        result = {
            "tools": [t for t in details["profiles"]["tools"] if t["code"] == args.tool],
            "evaluations": [
                e for e in details["profiles"]["evaluations"] if args.tool in e["tool_codes"]
            ],
        }
    else:
        result = details["ledger"]["coverage"]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
