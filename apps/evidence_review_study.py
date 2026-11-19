"""Freeze/replay an offline review study, without database or model execution."""

import argparse
import hashlib
import json
from pathlib import Path

from apps.evidence_graph_scores import unscored
from apps.glaucoma_evidence_plan import checksum as snapshot_checksum
from packages.evidence_details.models import body, digest
from packages.evidence_scoring import current
from packages.evidence_scoring.review_protocol import ReviewRound
from packages.evidence_scoring.review_study import run

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/research/evidence_review_protocol_v1.json"
MAX_BYTES = 32 * 1024 * 1024


def byte_checksum(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def read_json(path):
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("study_input_too_large")
    value = json.loads(raw)
    body(value)  # Reject NaN/Infinity anywhere, including unconsumed metadata.
    return raw, value


def code_inventory():
    paths = {
        *ROOT.joinpath("packages/evidence_scoring").glob("*.py"),
        *ROOT.joinpath("packages/evidence_details").glob("*.py"),
        ROOT / "apps/evidence_review_study.py",
        ROOT / "apps/evidence_graph_records.py",
        ROOT / "apps/evidence_graph_scores.py",
        ROOT / "apps/glaucoma_evidence_plan.py",
        ROOT / "configs/prompts/evidence_review_v1.md",
        ROOT / "pyproject.toml",
        PROTOCOL,
    }
    return {str(p.relative_to(ROOT)): byte_checksum(p.read_bytes()) for p in sorted(paths)}


def assemble(snapshot):
    raw, view = read_json(snapshot)
    if view.get("checksum") != snapshot_checksum(
        {k: v for k, v in view.items() if k != "checksum"}
    ):
        raise ValueError("study_snapshot_checksum_mismatch")
    base = unscored(view)
    score = current.validate(base, view["current_scoring"])
    _, protocol = read_json(PROTOCOL)
    if (score["schema_version"], score["policy_version"]) != (
        protocol["production_score_version"],
        protocol["production_policy_version"],
    ):
        raise ValueError("study_policy_version_mismatch")
    experiment = run(base)
    manifest = {
        "schema_version": "case-review-study/1.0.0",
        "coordinator_only": True,
        "snapshot_file": "baseline.json",
        "snapshot_file_checksum": byte_checksum(raw),
        "snapshot_checksum": view["checksum"],
        "case_code": score["case_code"],
        "input_checksum": score["input_checksum"],
        "score_checksum": score["checksum"],
        "policy_checksum": score["policy_checksum"],
        "protocol_checksum": digest(protocol),
        "implementation": code_inventory(),
        "assessment_ids": sorted(r["id"] for r in score["reviews"]),
        "mechanism_result_checksum": digest(experiment),
        "review_materials": {"status": "not_released", "checksum": None},
        "independent_rounds": {r: "not_started" for r in protocol["rounds"]},
        "production_modified": False,
    }
    manifest["checksum"] = digest(manifest)
    return manifest, experiment


def freeze(snapshot, output):
    snapshot, output = snapshot.resolve(), output.resolve()
    if not output.is_relative_to(Path("/opt/data/test")):
        raise ValueError("study_output_outside_project_storage")
    if output.exists():
        raise ValueError("study_output_already_exists")
    manifest, experiment = assemble(snapshot)
    # New directory only. Never empty, adopt or repair an old evidence directory.
    output.mkdir(parents=False, exist_ok=False)
    # Retain the exact source bytes so a later HTML/JSON export cannot replace
    # the study's inputs. This is coordinator-only, never a blinded packet.
    raw, _ = read_json(snapshot)
    if byte_checksum(raw) != manifest["snapshot_file_checksum"]:
        raise ValueError("study_snapshot_changed_during_freeze")
    with (output / "baseline.json").open("xb") as stream:
        stream.write(raw)
    files = {
        "manifest.json": manifest,
        "mechanisms.json": experiment,
        "review-round.schema.json": ReviewRound.model_json_schema(),
        "protocol.json": read_json(PROTOCOL)[1],
        "review-plan.json": {
            "status": "not_started",
            "material_checksum": None,
            "assessment_ids": manifest["assessment_ids"],
            "rounds": ["A", "B", "C"],
            "instruction": "Release a separately curated blinded packet before any round. "
            "Do not send manifest.json, mechanisms.json or the old graph to reviewers.",
        },
    }
    for name, value in files.items():
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return experiment["passed"]


def verify(output):
    _, saved = read_json(output / "manifest.json")
    if saved["checksum"] != digest({k: v for k, v in saved.items() if k != "checksum"}):
        raise ValueError("study_manifest_mismatch")
    if saved["implementation"] != code_inventory():
        raise ValueError("study_implementation_drift")
    if saved["snapshot_file"] != "baseline.json":
        raise ValueError("study_snapshot_path_mismatch")
    manifest, experiment = assemble(output / "baseline.json")
    if manifest != saved or read_json(output / "mechanisms.json")[1] != experiment:
        raise ValueError("study_replay_mismatch")
    if read_json(output / "protocol.json")[1] != read_json(PROTOCOL)[1]:
        raise ValueError("study_protocol_drift")
    if read_json(output / "review-round.schema.json")[1] != ReviewRound.model_json_schema():
        raise ValueError("study_review_schema_drift")
    return experiment["passed"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    new = sub.add_parser("freeze", help="Validate current v5 snapshot and run offline checks")
    new.add_argument("--snapshot", type=Path, required=True)
    new.add_argument("--output", type=Path, required=True)
    replay = sub.add_parser("verify", help="Read-only exact replay; never writes old evidence")
    replay.add_argument("--study", type=Path, required=True)
    args = parser.parse_args(argv)
    passed = freeze(args.snapshot, args.output) if args.command == "freeze" else verify(args.study)
    print(json.dumps({"mechanism_checks_passed": passed, "new_medical_reviews": 0}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
