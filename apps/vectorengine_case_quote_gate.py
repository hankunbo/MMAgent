"""Exact incremental gate and current-owner carry-forward, not a retry of a stopped run."""

import json
from pathlib import Path

from packages.grounding.models import digest
from packages.object_store import checksum

PACKAGE = Path("docs/acceptance/12-completion-c-quote-integration-v1")
BASELINE = Path("docs/acceptance/12-completion-cde-target-live-20261001")
BASELINE_CHECKSUM = "sha256:fb42277366c435d58ebff925e1d9fa2d1a14f404d6182f4eb1240de30e93b8d9"
COVERED = {
    "agents/expert_agent/reporting.py",
    "agents/provider_agents/service.py",
    "apps/vectorengine_batch_prepare.py",
    "apps/vectorengine_case_quote_gate.py",
    "apps/vectorengine_live_assembly.py",
    "apps/vectorengine_public_diagnostic.py",
    "apps/vectorengine_round.py",
    "configs/research/vectorengine_batch_citations_case_quote_r01.json",
    "packages/evidence_details/acceptance.py",
    "packages/evidence_details/source_quotes.py",
    "packages/evidence_details/successor_owner.py",
    "packages/knowledge_claims/research_candidates.py",
    "packages/knowledge_claims/research_quote.py",
    "packages/llm_provider/bounded_live.py",
    "packages/llm_provider/live_allocation.py",
    "packages/llm_provider/schema_tools.py",
    "packages/llm_provider/schemas/StageAllocationV2.schema.json",
    "tests/agent_provider/test_research_quote_alignment.py",
    "tests/agent_provider/test_source_quote_continuation.py",
    "tests/agent_provider/test_source_quote_integration.py",
    "tests/agent_provider/test_source_quote_postgres.py",
    "tests/expert_agent/test_batch_assembly.py",
}


def read(path):
    return json.loads(path.read_text())


def verify_delta(before, current, delta):
    expected = {
        p: dict(before=before.get(p), after=current.get(p))
        for p in sorted(set(before) | set(current))
        if before.get(p) != current.get(p)
    }
    if digest(before) != BASELINE_CHECKSUM or set(expected) != COVERED or delta != expected:
        raise ValueError("case_quote_uncovered_delta")


def check_acceptance():
    from apps.vectorengine_round import implementation_files

    before = read(BASELINE / "implementation-before.json")
    current = read(PACKAGE / "implementation-freeze.json")
    proof = read(PACKAGE / "verification.json")
    delta = {
        p: dict(before=before.get(p), after=current.get(p))
        for p in sorted(set(before) | set(current))
        if before.get(p) != current.get(p)
    }
    verify_delta(before, current, read(PACKAGE / "delta.json"))
    if (
        digest(before) != BASELINE_CHECKSUM
        or proof.get("schema_version") != "source-quote-integration-verification/1.0.0"
        or proof.get("offline_passed") is not True
        or proof.get("implementation_checksum") != digest(current)
        or proof.get("baseline_checksum") != digest(before)
        or proof.get("delta_checksum") != digest(delta)
        or read(PACKAGE / "delta.json") != delta
        or set(current) != implementation_files()
        or any(checksum(Path(p).read_bytes()) != sha for p, sha in current.items())
    ):
        raise ValueError("case_quote_implementation_or_delta")
    evidence = {}
    required_commands = {
        "cpu-final": ["/opt/data/test/envs/MMAgent1/bin/python", "-m", "pytest"],
        "pg-final": [
            "/opt/data/test/envs/MMAgent1/bin/python",
            "-m",
            "pytest",
            "tests/agent_provider/test_source_quote_postgres.py",
            "-q",
            "-s",
        ],
        "schemas": [
            "/opt/data/test/envs/MMAgent1/bin/python",
            "-m",
            "packages.llm_provider.schema_tools",
        ],
        "static-final": ["make", "env-check", "lock-check", "format-check", "lint", "schema-check"],
    }
    for name, command in required_commands.items():
        actual_command = read(PACKAGE / (name + "-command.json"))
        result = read(PACKAGE / (name + "-exit.json"))
        if name == "cpu-final":
            valid_command = actual_command[:3] == command and {
                "tests/agent_provider/test_source_quote_integration.py",
                "tests/agent_provider/test_source_quote_continuation.py",
                "tests/agent_provider/test_research_quote_alignment.py",
                "tests/agent_provider/test_acceptance.py",
            } <= set(actual_command)
        else:
            valid_command = actual_command == command
        if (
            not valid_command
            or result.get("exit_code") != 0
            or result.get("implementation") != current
            or result.get("implementation_after") != current
            or result.get("log_checksum") != checksum((PACKAGE / (name + ".log")).read_bytes())
            or (name == "pg-final" and result.get("owned_cleanup_verified") is not True)
        ):
            raise ValueError("case_quote_validation:" + name)
        for suffix in ("-command.json", "-exit.json", ".log"):
            path = PACKAGE / (name + suffix)
            evidence[str(path)] = checksum(path.read_bytes())
    if evidence != proof.get("validation_evidence"):
        raise ValueError("case_quote_validation_evidence")
    inherited = proof["inherited_evidence"]
    if not inherited or any(checksum(Path(p).read_bytes()) != sha for p, sha in inherited.items()):
        raise ValueError("case_quote_inherited_evidence")
    evidence.update(inherited)
    for name in ("verification.json", "delta.json", "implementation-freeze.json"):
        path = PACKAGE / name
        evidence[str(path)] = checksum(path.read_bytes())
    return current, evidence


def check_public_baseline(public):
    _, evidence = check_acceptance()
    gate = read(Path(public.gate))
    if (
        str(Path(public.gate)) not in evidence
        or gate.get("round_checksum") != digest(public)
        or gate.get("synthetic_regression_passed") is not True
    ):
        raise ValueError("case_quote_public_binding")
    for p, sha in {**gate["materials"], **gate["evidence"]}.items():
        if checksum(Path(p).read_bytes()) != sha:
            raise ValueError("case_quote_public_evidence")
    return gate


def prior_case_usage(documents, prior):
    """Read original terminal Control/journal; unknown spending is never treated as zero."""
    from packages.llm_provider.live_allocation import PriorCaseUsage
    from packages.llm_provider.provider import RequestBudget
    from packages.research_runtime.agent_records import RuntimeAgentJournal

    state = documents.sessions.read(prior.run_id)
    journal = RuntimeAgentJournal(documents)
    events = journal.read()
    if (
        state is None
        or state.phase != "stopped"
        or state.pending
        or documents.run_id != prior.run_id
        or state.binding.run_id != prior.run_id
        or not events
    ):
        raise ValueError("case_quote_prior_not_terminal")
    budget = RequestBudget(
        run_id=prior.run_id, deadline=state.binding.deadline, max_requests=40, journal=journal
    )
    records = budget.records()
    counts = budget.transport_counts()
    if (
        budget.used != 1
        or len(records) != 1
        or records[0].simulated
        or records[0].role != "source"
        or records[0].error != "scope"
        or records[0].scope.run_id != prior.run_id
        or records[0].config_checksum != prior.provider_checksum
        or counts["dispatch_unknown"]
        or counts["remote_response_unknown"]
        or counts["dispatched_observed"] != 1
        or counts["remote_response_confirmed"] != 1
        or documents.sessions.read(prior.run_id) != state
        or journal.read() != events
    ):
        raise ValueError("case_quote_prior_accounting")
    return PriorCaseUsage(
        run_id=prior.run_id,
        configuration_checksum=digest(prior),
        control_checksum=digest(state),
        journal_checksum=digest(events),
        used=budget.used,
    )
