"""Final incremental proof and terminal historical spending for the integrated run."""

import json
from pathlib import Path

from packages.grounding.models import digest
from packages.object_store import checksum

PACKAGE = Path("docs/acceptance/12-completion-integrated-v1")
BASELINE = Path("docs/acceptance/12-completion-review-shape-v1/implementation-freeze.json")
BASELINE_SHA = "sha256:710a50da786dfb14c97e169fe536335335312f7da5b924eb76f146e5b96f666b"


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
    return current, evidence


def historical_usage(round_config, evidence, target):
    from apps.vectorengine_public_diagnostic import original_documents
    from apps.vectorengine_round import load_round
    from packages.llm_provider.live_allocation import HistoricalRunUsage
    from packages.llm_provider.provider import RequestBudget
    from packages.research_runtime.agent_records import RuntimeAgentJournal

    result = []
    for path in round_config.prior_round_configs:
        old = load_round(path)
        docs = original_documents(evidence, target, old.run_id)
        state = docs.sessions.read(old.run_id)
        journal = RuntimeAgentJournal(docs)
        events = journal.read()
        if state is None or state.phase != "stopped" or state.pending or not events:
            raise ValueError("historical_run_not_terminal")
        budget = RequestBudget(
            run_id=old.run_id, deadline=state.binding.deadline, max_requests=40, journal=journal
        )
        counts = budget.transport_counts()
        if (
            not budget.used
            or len(budget.records()) != budget.used
            or counts["dispatch_unknown"]
            or counts["remote_response_unknown"]
            or any(
                r.simulated
                or r.scope.run_id != old.run_id
                or r.config_checksum != old.provider_checksum
                for r in budget.records()
            )
            or docs.sessions.read(old.run_id) != state
            or journal.read() != events
        ):
            raise ValueError("historical_usage_unknown_or_changed")
        result.append(
            HistoricalRunUsage(
                stage=old.stage,
                run_id=old.run_id,
                configuration_checksum=digest(old),
                control_checksum=digest(state),
                journal_checksum=digest(events),
                used=budget.used,
            )
        )
    return tuple(result)
