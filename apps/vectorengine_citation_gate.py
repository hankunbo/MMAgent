"""One explicit citation repair evidence chain; no gate or runtime creation here."""

import json
from pathlib import Path

from packages.grounding.models import digest
from packages.object_store import checksum

PACKAGE = Path("docs/acceptance/12-completion-b-citation-contract-v1")
BASELINE = Path("docs/acceptance/12-completion-a-batched-review-v1")
BASELINE_CHECKSUM = "sha256:d651302c13a8e85b09e495e6cf44c6f7b22f7f8c0a06bc3800359f6574ed3527"
COVERED = {
    "apps/vectorengine_batch_prepare.py",
    "apps/vectorengine_citation_gate.py",
    "agents/provider_agents/batched_review.py",
    "packages/llm_provider/transport.py",
    "packages/llm_provider/review_citations.py",
    "configs/prompts/agents/review-batch-1.1.0.txt",
    "configs/agents/vectorengine_batch_citations.json",
    "configs/research/vectorengine_batch_citations_public.json",
    "configs/research/vectorengine_batch_citations_case.json",
    "tests/agent_provider/test_review_citation_contract.py",
    "tests/agent_provider/test_citation_gate.py",
}


def difference(before, after):
    return {
        p: dict(before=before.get(p), after=after.get(p))
        for p in sorted(set(before) | set(after))
        if before.get(p) != after.get(p)
    }


def verify_delta(baseline, current, delta, verification):
    expected = difference(baseline, current)
    if (
        digest(baseline) != BASELINE_CHECKSUM
        or delta != expected
        or set(expected) != COVERED
        or verification.get("schema_version") != "review-citation-repair-verification/1.0.0"
        or verification.get("offline_passed") is not True
        or verification.get("persistent_semantics_changed") is not False
        or verification.get("baseline_checksum") != digest(baseline)
        or verification.get("implementation_checksum") != digest(current)
        or verification.get("delta_checksum") != digest(delta)
        or verification.get("covered_changes") != sorted(expected)
    ):
        raise ValueError("citation_repair_uncovered_delta")


def check_acceptance():
    from apps.vectorengine_batch_prepare import baseline_evidence
    from apps.vectorengine_round import implementation_files

    def read(path):
        return json.loads(path.read_text())

    baseline = read(BASELINE / "implementation-freeze.json")
    current = read(PACKAGE / "implementation-freeze.json")
    proof = read(PACKAGE / "verification.json")
    verify_delta(baseline, current, read(PACKAGE / "delta.json"), proof)
    if set(current) != implementation_files() or any(
        checksum(Path(p).read_bytes()) != sha for p, sha in current.items()
    ):
        raise ValueError("citation_implementation_drift")
    evidence = baseline_evidence(baseline)
    inherited = read(Path("docs/acceptance/12-completion-b-public-review-v1/precheck.json"))[
        "evidence_references"
    ]
    if any(inherited.get(p) != sha for p, sha in evidence.items()):
        raise ValueError("citation_original_A_evidence_drift")
    if proof.get("inherited_evidence") != evidence:
        raise ValueError("citation_inherited_evidence_binding")
    checks = {}
    for name in ("cpu-final", "schemas", "static-final"):
        directory = PACKAGE / "synthetic" / name
        command = read(directory / "command.json")
        python = "/opt/data/test/envs/MMAgent1/bin/python"
        command_valid = (
            command == ["make", "env-check", "lock-check", "format-check", "lint", "schema-check"]
            if name == "static-final"
            else command == [python, "-m", "packages.llm_provider.schema_tools"]
            if name == "schemas"
            else command[:3] == [python, "-m", "pytest"]
            and {
                "tests/agent_provider/test_review_citation_contract.py",
                "tests/agent_provider/test_citation_gate.py",
            }.issubset(command)
        )
        result = read(directory / "exit.json")
        if (
            not command_valid
            or result.get("exit_code") != 0
            or result.get("implementation_checksum") != digest(current)
        ):
            raise ValueError("citation_validation_required:" + name)
        if name == "cpu-final" and (
            not result.get("passed") or not result.get("owned_cleanup_verified")
        ):
            raise ValueError("citation_cpu_cleanup_required")
        for file in ("command.json", "output.log", "exit.json"):
            path = directory / file
            checks[str(path)] = checksum(path.read_bytes())
    if proof.get("validation_evidence") != checks:
        raise ValueError("citation_validation_evidence_drift")
    evidence.update(checks)
    for name in ("implementation-freeze.json", "delta.json", "verification.json"):
        path = PACKAGE / name
        evidence[str(path)] = checksum(path.read_bytes())
    return current, evidence
