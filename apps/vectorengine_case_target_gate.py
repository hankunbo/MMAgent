"""Explicit C-only repair delta over successful B; never rewrites old gates."""

import json
from pathlib import Path

from packages.grounding.models import digest
from packages.object_store import checksum

PACKAGE = Path("docs/acceptance/12-completion-c-target-repair-v1")
BASELINE = Path("docs/acceptance/12-completion-b-citations-public-v1")
BASELINE_CHECKSUM = "sha256:79a52663138d39d9b4df6082324090febc197c224c1c05890817237f1baee0b8"
COVERED = {
    ".local/vectorengine/run.py",
    "apps/vectorengine_case_preflight.py",
    "apps/vectorengine_case_materials.py",
    "apps/vectorengine_round.py",
    "apps/vectorengine_single_case.py",
    "apps/vectorengine_live_assembly.py",
    "apps/vectorengine_batch_prepare.py",
    "apps/vectorengine_case_target_gate.py",
    "apps/vectorengine_public_diagnostic.py",
    "packages/graph_engine/research_candidates.py",
    "packages/knowledge_claims/research_candidates.py",
    "configs/research/vectorengine_batch_citations_case_target_r01.json",
    "tests/agent_provider/test_case_graph_target.py",
    "tests/agent_provider/test_case_target_gate.py",
    "tests/agent_provider/test_round_preparation.py",
    "tests/agent_provider/test_batch_preparation.py",
    "tests/agent_provider/test_round3_assembly.py",
    "tests/agent_provider/test_live_terminal_guard.py",
    "tests/agent_provider/test_bounded_live.py",
}


def read(path):
    return json.loads(path.read_text())


def verify_delta(before, current, delta, proof):
    expected = {
        p: dict(before=before.get(p), after=current.get(p))
        for p in sorted(set(before) | set(current))
        if before.get(p) != current.get(p)
    }
    if (
        digest(before) != BASELINE_CHECKSUM
        or delta != expected
        or set(expected) != COVERED
        or proof.get("schema_version") != "case-target-repair-verification/1.0.0"
        or proof.get("offline_passed") is not True
        or proof.get("persistent_semantics_changed") is not False
        or proof.get("baseline_checksum") != digest(before)
        or proof.get("implementation_checksum") != digest(current)
        or proof.get("delta_checksum") != digest(delta)
    ):
        raise ValueError("case_target_uncovered_delta")


def check_acceptance():
    from apps.vectorengine_round import implementation_files

    before = read(BASELINE / "implementation-before.json")
    current = read(PACKAGE / "implementation-freeze.json")
    proof = read(PACKAGE / "verification.json")
    verify_delta(before, current, read(PACKAGE / "delta.json"), proof)
    evidence = {}
    for name in ("cpu-final", "schemas", "static-final"):
        result = read(PACKAGE / (name + "-exit.json"))
        command = read(PACKAGE / (name + "-command.json"))
        expected = {
            "schemas": [
                "/opt/data/test/envs/MMAgent1/bin/python",
                "-m",
                "packages.llm_provider.schema_tools",
            ],
            "static-final": [
                "make",
                "env-check",
                "lock-check",
                "format-check",
                "lint",
                "schema-check",
            ],
        }
        command_ok = (
            command == expected[name]
            if name in expected
            else command[:3] == ["/opt/data/test/envs/MMAgent1/bin/python", "-m", "pytest"]
            and {
                "tests/agent_provider/test_case_graph_target.py",
                "tests/agent_provider/test_case_target_gate.py",
            }
            <= set(command)
        )
        if (
            not command_ok
            or result.get("exit_code") != 0
            or result.get("implementation") != current
        ):
            raise ValueError("case_target_validation_required:" + name)
        if name == "cpu-final" and not result.get("owned_cleanup_verified"):
            raise ValueError("case_target_cleanup_required")
        for suffix in ("-command.json", "-exit.json", ".log"):
            p = PACKAGE / (name + suffix)
            evidence[str(p)] = checksum(p.read_bytes())
    if proof.get("validation_evidence") != evidence:
        raise ValueError("case_target_validation_drift")
    # Bind saved real public evidence, prior stopped-case evidence and previous validation.
    inherited = proof["inherited_evidence"]
    if not inherited or any(checksum(Path(p).read_bytes()) != sha for p, sha in inherited.items()):
        raise ValueError("case_target_inherited_evidence_drift")
    evidence.update(inherited)
    for name in ("implementation-freeze.json", "delta.json", "verification.json"):
        p = PACKAGE / name
        evidence[str(p)] = checksum(p.read_bytes())
    # The original repair freeze stays immutable. Only this exact, tested increment
    # changes its zero-request interpretation and the corresponding regression tests.
    current, evidence = check_zero_request_increment(current, evidence)
    if set(current) != implementation_files() or any(
        checksum(Path(p).read_bytes()) != sha for p, sha in current.items()
    ):
        raise ValueError("case_target_implementation_drift")
    return current, evidence


def check_zero_request_increment(before, evidence):
    package = Path("docs/acceptance/12-completion-c-zero-request-r02")
    current = read(package / "implementation-freeze.json")
    delta = read(package / "delta.json")
    proof = read(package / "verification.json")
    actual = {
        p: dict(before=before.get(p), after=current.get(p))
        for p in sorted(set(before) | set(current))
        if before.get(p) != current.get(p)
    }
    if (
        set(actual)
        != {
            "apps/vectorengine_case_target_gate.py",
            "tests/agent_provider/test_case_target_gate.py",
        }
        or delta != actual
        or proof.get("schema_version") != "case-zero-request-repair/1.0.0"
        or proof.get("offline_passed") is not True
        or proof.get("persistent_semantics_changed") is not False
        or proof.get("baseline_checksum") != digest(before)
        or proof.get("implementation_checksum") != digest(current)
        or proof.get("delta_checksum") != digest(delta)
    ):
        raise ValueError("case_target_zero_request_delta")
    expected = {
        "cpu-final": [
            "/opt/data/test/envs/MMAgent1/bin/python",
            "-m",
            "pytest",
            "tests/agent_provider/test_case_target_gate.py",
            "tests/agent_provider/test_case_graph_target.py",
            "-q",
        ],
        "static-final": [
            "make",
            "env-check",
            "lock-check",
            "format-check",
            "lint",
            "schema-check",
        ],
    }
    validation = {}
    for name, command in expected.items():
        result = read(package / (name + "-exit.json"))
        if (
            read(package / (name + "-command.json")) != command
            or result.get("exit_code") != 0
            or result.get("implementation") != current
        ):
            raise ValueError("case_target_zero_request_validation")
        for suffix in ("-command.json", "-exit.json", ".log"):
            path = package / (name + suffix)
            validation[str(path)] = checksum(path.read_bytes())
    if proof.get("validation_evidence") != validation:
        raise ValueError("case_target_zero_request_evidence")
    evidence = dict(evidence, **validation)
    for name in ("implementation-freeze.json", "delta.json", "verification.json"):
        path = package / name
        evidence[str(path)] = checksum(path.read_bytes())
    return current, evidence


def check_public_baseline(public):
    """Only v4 C may reuse this exact old public gate under the verified repair delta."""
    _, evidence = check_acceptance()
    gate = read(Path(public.gate))
    before = read(BASELINE / "implementation-before.json")
    if (
        str(Path(public.gate)) not in evidence
        or gate.get("implementation") != before
        or gate.get("round_checksum") != digest(public)
        or gate.get("schema_version") != "bounded-live-round-gate/1.0.0"
        or gate.get("synthetic_regression_passed") is not True
    ):
        raise ValueError("case_target_public_gate_binding")
    for p, sha in {**gate["materials"], **gate["evidence"]}.items():
        if checksum(Path(p).read_bytes()) != sha:
            raise ValueError("case_target_public_evidence_drift")
    return gate


def check_prior_case(documents):
    """Current original owners, not the exported report, prove zero spent case requests."""
    from packages.research_runtime.agent_records import RuntimeAgentJournal

    state = documents.sessions.read(documents.run_id)
    if state is None or state.phase != "stopped" or not state.cancelled or state.pending:
        raise ValueError("case_target_prior_not_cancelled")
    journal = RuntimeAgentJournal(documents).read()
    if not journal:
        return
    from apps.vectorengine_round import load_round
    from packages.grounding.models import validate
    from packages.llm_provider.live_allocation import StageAllocation

    # Only exact metadata is zero-use: any reservation, HTTP attempt/confirmation,
    # response, unknown extension or extra field remains fail-closed. Never create
    # a RequestBudget here: this current-state check must not append journal entries.
    if len(journal) != 4 or journal[3] != {"phase": "live_cancelled"}:
        raise ValueError("case_target_prior_requests_present")
    prior = load_round("configs/research/vectorengine_batch_citations_case.json")
    public = load_round(prior.public_config)
    allocation = validate(StageAllocation, journal[2])
    if (
        documents.run_id != state.binding.run_id
        or documents.run_id != prior.run_id
        or allocation.stage != "case"
        or allocation.session_id != prior.session_id
        or allocation.public_run_id != public.run_id
        or allocation.public_used != 2
        or allocation.configuration_checksums != (prior.provider_checksum,)
    ):
        raise ValueError("case_target_prior_metadata_binding")
    expected_budget = dict(
        schema_version="agent-budget/1.0.0",
        binding=dict(
            run_id=documents.run_id, deadline=state.binding.deadline.isoformat(), max_requests=40
        ),
    )
    expected_limits = dict(
        schema_version="bounded-live-limits/1.0.0",
        config_checksum=digest(allocation),
        max_steps=20,
        max_no_progress=2,
        max_searches=8,
        max_concurrency=1,
        max_retries=1,
        seconds=2700,
        deadline=state.binding.deadline.isoformat(),
    )
    if journal[0] != expected_budget or journal[1] != expected_limits:
        raise ValueError("case_target_prior_metadata_binding")
