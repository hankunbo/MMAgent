"""Final incremental proof and terminal historical spending for the integrated run."""

import json
from pathlib import Path

from packages.grounding.models import digest
from packages.object_store import checksum

PACKAGE = Path("docs/acceptance/12-completion-source-selection-v1")
BASELINE = Path("docs/acceptance/12-completion-integrated-v1/implementation-freeze.json")
BASELINE_SHA = "sha256:56624d0a48111baa4204e64ee832bed78af68d91ad77d4bac14c97365afe1ada"


def read(path):
    return json.loads(Path(path).read_text())


def check_acceptance():
    from apps.vectorengine_round import implementation_files

    before = read(PACKAGE / "implementation-before.json")
    current = read(PACKAGE / "implementation-freeze.json")
    proof = read(PACKAGE / "verification.json")
    delta = {
        p: dict(before=before.get(p), after=current.get(p))
        for p in sorted(before.keys() | current.keys())
        if before.get(p) != current.get(p)
    }
    if (
        digest(before) != BASELINE_SHA
        or before != read(BASELINE)
        or set(current) != implementation_files()
        or any(checksum(Path(p).read_bytes()) != sha for p, sha in current.items())
        or read(PACKAGE / "delta.json") != delta
        or proof.get("offline_passed") is not True
        or proof.get("implementation_checksum") != digest(current)
    ):
        raise ValueError("integrated_implementation_or_coverage")
    evidence = proof["validation_evidence"]
    if not evidence or any(checksum(Path(p).read_bytes()) != sha for p, sha in evidence.items()):
        raise ValueError("integrated_validation_evidence")
    for name in ("cpu-final", "schema-final", "static-final"):
        result = read(PACKAGE / name / "exit.json")
        if (
            result["exit_code"] != 0
            or result["before"] != current
            or result["after"] != current
            or (name == "pg-final" and not result["owned_cleanup_verified"])
        ):
            raise ValueError("integrated_validation_failed")
    for name in (
        "verification.json",
        "implementation-before.json",
        "implementation-freeze.json",
        "delta.json",
    ):
        path = PACKAGE / name
        evidence[str(path)] = checksum(path.read_bytes())
    return current, evidence
