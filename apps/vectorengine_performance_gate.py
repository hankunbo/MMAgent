"""Final incremental proof and terminal historical spending for the integrated run."""

import json
from pathlib import Path

from packages.grounding.models import digest
from packages.object_store import checksum

PACKAGE = Path("docs/acceptance/12-completion-performance-v1")
BASELINE = Path("docs/acceptance/12-completion-source-selection-v1/implementation-freeze.json")
BASELINE_SHA = "sha256:398a5ccc03cbfd3b968c6e4c7bfd0e3e87b986d405bcfc309c505c621f0a809d"


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
        or read(PACKAGE / "delta.json") != delta
        or proof.get("offline_passed") is not True
        or proof.get("implementation_checksum") != digest(current)
    ):
        raise ValueError("integrated_implementation_or_coverage")
    evidence = proof["validation_evidence"]
    if not evidence or any(checksum(Path(p).read_bytes()) != sha for p, sha in evidence.items()):
        raise ValueError("integrated_validation_evidence")
    for name in ("cpu-final", "pg-final", "schema-final", "static-final"):
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
    # The final gate-dispatch-only increment reuses the exact, unchanged PG owner.
    updated = read(PACKAGE / "implementation-freeze-r02.json")
    increment = read(PACKAGE / "gate-increment.json")
    changed = {p for p in current.keys() | updated.keys() if current.get(p) != updated.get(p)}
    if (
        changed
        != {
            "apps/vectorengine_batch_prepare.py",
            "apps/vectorengine_performance_gate.py",
            "tests/agent_provider/test_current_source_batch.py",
        }
        or increment["before_checksum"] != digest(current)
        or increment["after_checksum"] != digest(updated)
        or set(updated) != implementation_files()
        or any(checksum(Path(p).read_bytes()) != sha for p, sha in updated.items())
    ):
        raise ValueError("performance_gate_increment")
    for name in ("cpu-gate-r02", "static-gate-r02"):
        result = read(PACKAGE / name / "exit.json")
        if result["exit_code"] != 0 or result["before"] != updated or result["after"] != updated:
            raise ValueError("performance_gate_increment_validation")
        if result["log_checksum"] != checksum((PACKAGE / name / "output.log").read_bytes()):
            raise ValueError("performance_gate_increment_log")
        for file in ("exit.json", "command.json", "output.log"):
            path = PACKAGE / name / file
            evidence[str(path)] = checksum(path.read_bytes())
    for name in ("implementation-freeze-r02.json", "gate-increment.json"):
        path = PACKAGE / name
        evidence[str(path)] = checksum(path.read_bytes())
    return updated, evidence


def check_public_baseline(public):
    """Exact public proof under the explicit CPU/PG-covered read-performance delta."""
    check_acceptance()
    base = Path("docs/acceptance/12-completion-source-selection-v1")
    before = read(BASELINE)
    gate = read(public.gate)
    proof = read(base / "public-readback.json")
    capture = read(base / "execute-public/exit.json")
    if (
        gate["round_checksum"] != digest(public)
        or gate["implementation"] != before
        or not proof["qualified"]
        or proof["requests"] != 2
        or proof["run_id"] != public.run_id
        or capture["exit_code"] != 0
        or capture["before"] != before
        or capture["after"] != before
        or capture["log_checksum"] != checksum((base / "execute-public/output.log").read_bytes())
    ):
        raise ValueError("performance_public_binding")
    for p, sha in {**gate["materials"], **gate["evidence"]}.items():
        if checksum(Path(p).read_bytes()) != sha:
            raise ValueError("performance_public_evidence")
    # Request, parsing, prompt, budget and response semantics must be byte-identical.
    protected = {
        p: sha
        for p, sha in before.items()
        if p.startswith(
            (
                "packages/llm_provider/",
                "packages/evidence_details/",
                "agents/provider_agents/",
                "configs/prompts/",
                "configs/agents/",
            )
        )
        or p == public.provider_config
        or p == public.material_catalog
    }
    if any(checksum(Path(p).read_bytes()) != sha for p, sha in protected.items()):
        raise ValueError("performance_public_wire_changed")
    return gate
