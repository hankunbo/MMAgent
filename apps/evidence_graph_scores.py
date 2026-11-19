"""Persist/export the first research score review through Runtime's existing ports."""

import argparse
import copy
import json
from datetime import UTC, datetime
from pathlib import Path

from apps.evidence_graph_details import _read_json, _reference
from apps.glaucoma_evidence_plan import checksum as snapshot_checksum
from db.runtime.repository.case_pipeline import identity
from packages.contracts import Artifact
from packages.evidence_details.models import body
from packages.evidence_scoring.api import build, validate
from packages.object_store import checksum

VERSION = "case-evidence-graph/1.13.0"
VERSIONS = {
    "1.0.0": VERSION,
    "2.0.0": "case-evidence-graph/1.14.0",
    "3.0.0": "case-evidence-graph/1.15.0",
    "4.0.0": "case-evidence-graph/1.16.0",
    "5.0.0": "case-evidence-graph/1.19.0",
}
SCORE_TYPE = "ARTIFACT:CASE_RESEARCH_SCORES"


def unscored(value):
    field = "current_scoring" if "current_scoring" in value else "research_scoring"
    base = {k: v for k, v in value.items() if k not in {field, "checksum"}}
    return {**base, "checksum": snapshot_checksum(base)}


def attach(base, scores):
    validate(base, scores)
    value = copy.deepcopy(base)
    field = (
        "current_scoring"
        if scores["schema_version"] == "case-research-scoring/5.0.0"
        else "research_scoring"
    )
    value[field] = scores
    value["checksum"] = snapshot_checksum({k: v for k, v in value.items() if k != "checksum"})
    return value


def read_manifest(document, case_code, current, repo, store, artifact):
    from apps.glaucoma_case_storage import read_view

    if (
        set(document) != {"schema", "case_code", "base", "scores", "input_checksum"}
        or document["case_code"] != case_code
    ):
        raise ValueError("research_score_manifest_scope_mismatch")
    selected = {}
    for key, kind in [("base", "ARTIFACT:CASE_EVIDENCE_GRAPH"), ("scores", SCORE_TYPE)]:
        ref = document[key]
        a = current.get(ref["code"])
        if (
            a is None
            or a.scope.case_code != case_code
            or a.artifact_type_code != kind
            or a.checksum != ref["checksum"]
        ):
            raise ValueError("research_score_attachment_mismatch")
        selected[key] = a
    if set(artifact.derived_from) != {a.asset_code for a in selected.values()}:
        raise ValueError("research_score_manifest_derivation_mismatch")
    if set(selected["scores"].derived_from) != {selected["base"].asset_code}:
        raise ValueError("research_score_derivation_mismatch")
    parent_version = (
        "case-evidence-graph/1.18.0"
        if document["schema"] == VERSIONS["5.0.0"]
        else "case-evidence-graph/1.12.0"
    )
    if _read_json(selected["base"], store).get("schema") != parent_version:
        raise ValueError("research_score_requires_v112_base")
    base = read_view(case_code, selected["base"].asset_code, repo=repo, store=store)
    if document["input_checksum"] != base["case_evidence"]["input"]["checksum"]:
        raise ValueError("research_score_input_mismatch")
    scores = _read_json(selected["scores"], store)
    score_version = scores.get("schema_version", "").removeprefix("case-research-scoring/")
    if score_version not in VERSIONS or document["schema"] != VERSIONS[score_version]:
        raise ValueError("research_score_manifest_version_mismatch")
    return attach(base, scores)


def register(case_code, base_code, *, repo, store, request_key=None, score_version="1.0.0"):
    from apps.glaucoma_case_storage import read_view

    current = {a.asset_code: a for a in repo.get_case_bundle(case_code).artifacts}
    original = current.get(base_code)
    if original is None or original.artifact_type_code != "ARTIFACT:CASE_EVIDENCE_GRAPH":
        raise ValueError("research_score_base_missing")
    parent_version = (
        "case-evidence-graph/1.18.0" if score_version == "5.0.0" else "case-evidence-graph/1.12.0"
    )
    if _read_json(original, store).get("schema") != parent_version:
        raise ValueError("research_score_requires_v112_base")
    base = read_view(case_code, base_code, repo=repo, store=store)
    scores = build(base, score_version)
    version = VERSIONS[score_version]
    key = identity(
        "TASK", "research-scores:" + (request_key or case_code + base_code + scores["checksum"])
    )
    previous = repo.read_observation_bundle(key)
    if previous is not None:
        if len(previous) != 2 or any(a.scope.case_code != case_code for a in previous):
            raise ValueError("research_score_request_scope_conflict")
        saved_score, graph = previous
        manifest = _read_json(graph, store)
        if manifest.get("schema") != version or manifest.get("base") != _reference(original):
            raise ValueError("research_score_request_base_conflict")
        value = read_view(case_code, graph.asset_code, repo=repo, store=store)
        field = "current_scoring" if score_version == "5.0.0" else "research_scoring"
        if value[field] != scores:
            raise ValueError("research_score_request_content_conflict")
        return _receipt(saved_score, graph, scores)
    created = []
    attempted = False

    def artifact(value, kind, sources):
        data = body(value)
        if len(data) > 4_194_304:
            raise ValueError("research_score_size_limit")
        sha = checksum(data)
        code = identity("ASSET", case_code + kind + sha)
        existing = current.get(code)
        if existing:
            if (
                existing.artifact_type_code != kind
                or existing.checksum != sha
                or set(existing.derived_from) != {a.asset_code for a in sources}
                or store.read_bounded(existing.uri, 4_194_304) != data
            ):
                raise ValueError("research_score_existing_conflict")
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
            generator_version=scores["policy_version"].replace("/", "-"),
            created_at=datetime.now(UTC).isoformat(),
            source_codes=[a.asset_code for a in sources],
            input_checksums=[a.checksum for a in sources],
        )
        return Artifact.model_validate(raw)

    try:
        score_artifact = artifact(scores, SCORE_TYPE, [original])
        manifest = {
            "schema": version,
            "case_code": case_code,
            "base": _reference(original),
            "scores": _reference(score_artifact),
            "input_checksum": scores["input_checksum"],
        }
        graph = artifact(manifest, "ARTIFACT:CASE_EVIDENCE_GRAPH", [original, score_artifact])
        attempted = True
        repo.register_observation_bundle((score_artifact, graph), key)
    except ValueError:
        for uri in created:
            store.discard(uri)
        raise
    except BaseException:
        if not attempted:
            for uri in created:
                store.discard(uri)
        # Keep own objects if a commit acknowledgement may have been lost.
        raise
    if read_view(case_code, graph.asset_code, repo=repo, store=store) != attach(base, scores):
        raise ValueError("research_score_readback_mismatch")
    return _receipt(score_artifact, graph, scores)


def _receipt(score_artifact, graph, scores):
    return {
        "graph": graph.model_dump(mode="json"),
        "scores": score_artifact.model_dump(mode="json"),
        "root": scores["nodes"].get("ROOT"),
        "review_count": len(scores["reviews"]),
        "removal_count": len(scores["removals"]),
        "policy_version": scores["policy_version"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["register", "inspect"])
    parser.add_argument("--case-code", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--node")
    parser.add_argument("--score-version", choices=sorted(VERSIONS), default="4.0.0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from apps.glaucoma_case_storage import read_view
    from apps.patient_service_assembly import OBJECT_ROOTS, check_object_root
    from apps.research_single_case import credential_boundary
    from db.runtime.repository.case_evidence import CaseEvidenceRepository
    from packages.object_store import LocalObjectStore

    check_object_root(OBJECT_ROOTS["visible"], "visible")
    repo = CaseEvidenceRepository(credential_boundary().resolve_target("runtime"))
    store = LocalObjectStore(OBJECT_ROOTS["visible"])
    if args.action == "register":
        if args.output is None or args.output.exists():
            parser.error("register requires a new --output receipt")
        result = register(
            args.case_code, args.graph, repo=repo, store=store, score_version=args.score_version
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(
            json.dumps(
                {"graph": result["graph"]["asset_code"], "reviews": result["review_count"]},
                ensure_ascii=False,
            )
        )
    else:
        view = read_view(args.case_code, args.graph, repo=repo, store=store)
        scores = view.get("current_scoring", view.get("research_scoring"))
        print(
            json.dumps(
                scores["nodes"][args.node] if args.node else scores, ensure_ascii=False, indent=2
            )
        )


if __name__ == "__main__":
    main()
