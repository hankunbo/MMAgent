"""Demonstrate the current cross-owner gap, not a success publication or PG proof.

These pass when the existing boundary is reproduced. The SQL research constraint
is independently reviewed and frozen; no synthetic medical success is injected.
"""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from db.memory.publication import PublicationRepository
from packages.grounding.models import digest
from packages.memory_writer.projection import project
from packages.research_runtime.single_case import ResearchSingleCaseService
from tests.memory.test_writer_projection import material, source
from tests.memory_review.support import binding


def reject_as_success(snapshot):
    record = project(snapshot, source(snapshot), request_key="10e-gap-probe")[-1]
    bound = binding(record).model_copy(update={"group": "success"})
    reads = []

    def read(key):
        reads.append(key)
        return SimpleNamespace(record=record, checksum=digest(record))

    # Pure _target consumer contract. No constructor/PG authorization claim.
    port = SimpleNamespace(read=read)
    with pytest.raises(ValueError, match="^memory_group_evidence_missing$"):
        PublicationRepository._target(port, bound)
    assert reads == [record.key]
    return record


def test_completed_original_planner_is_not_runtime_success():
    snapshot = material()  # Original Planner controls completion; no invented Runtime outcome.
    record = reject_as_success(snapshot)
    assert record.detail.terminal.scope == "planner_control"
    assert record.detail.terminal.runtime_status == "CREATED"
    assert record.detail.terminal.control_phase == "stopped"


def test_original_successful_technical_recovery_is_still_insufficient():
    class RuntimePort:
        status = "EXECUTING"

        def run_status(self, run_id):
            return self.status

        def record_lifecycle(self, old, *, phase, reason, target_status, report):
            if target_status:
                self.status = target_status
            return dict(old, phase=phase, technical_reason=reason, report=report)

    runtime = RuntimePort()
    ledger = SimpleNamespace(
        checksum=digest("empty synthetic ledger"), technical=(), entries=(), synthetic=True
    )
    scores = SimpleNamespace(
        save=lambda *args: ledger,
        read=lambda *args: ledger,
        policy_checksum=digest("synthetic policy"),
        objects=None,
    )
    executor = SimpleNamespace(
        recover=lambda *args: object(),
        recover_commit=lambda *args: SimpleNamespace(status="SUCCEEDED"),
    )
    state = dict(
        run_id="RUN:OID_PREFLIGHT",
        phase="EXECUTING",
        binding=dict(
            execution_key="existing-attempt",
            score_key="existing-score",
            plan_checksum=digest("plan"),
        ),
        deadline=(datetime.now(UTC) + timedelta(minutes=2)).isoformat(),
    )
    driver = ResearchSingleCaseService(
        runtime=runtime,
        research=None,
        choice=None,
        bind_plan=None,
        config_checksum=digest("config"),
        budget_seconds=60,
    )
    # The original service consumes a fixed SUCCEEDED recovery reply. No executor launches.
    report = driver._drive(state, None, executor, scores, None, True)
    assert report["run_status"] == "COMPLETED_INSUFFICIENT_EVIDENCE"
    assert report["reason"] == "research_output_without_medical_graph"
    snapshot = material()
    snapshot["control"] = None
    snapshot["runtime"].update(
        runtime_status=report["run_status"],
        lifecycle_phase="FINISHED",
        report=report,
    )
    record = reject_as_success(snapshot)
    assert record.detail.terminal.scope == "runtime_lifecycle"
    assert record.detail.terminal.runtime_status == "COMPLETED_INSUFFICIENT_EVIDENCE"
