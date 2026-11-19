"""Real PG and harmless processes: inconsistent facts never become confirmations."""

import json
from copy import deepcopy

import pytest
from psycopg.types.json import Jsonb

from db.runtime.repository.tool_execution import RuntimeToolExecutions
from packages.contracts import ToolExecutionStatus as Status
from packages.grounding.models import digest, validate
from packages.tool_execution.models import StepResult, WorkflowResult
from tests.database.knowledge_support_04a import sql
from tests.tool_execution.test_execution import closure as closure
from tests.tool_execution.test_execution import database as database
from tests.tool_execution.test_execution import host_request as host_request
from tests.tool_execution.test_execution import integrated as integrated
from tests.tool_execution.test_execution import make_ready as make_ready
from tests.tool_execution.test_execution import mapping_ready as mapping_ready


class Pause(RuntimeError):
    pass


class BeforeConfirmation(RuntimeToolExecutions):
    hold = True

    def confirm(self, workflow):
        self.workflow = workflow
        if self.hold:
            raise Pause("before confirmation")
        return super().confirm(workflow)


class BeforeTerminal(RuntimeToolExecutions):
    hold = True

    def advance(self, attempt, step_id, expected, target, **kwargs):
        if target == Status.SUCCEEDED and self.hold:
            self.record = kwargs["record"]
            raise Pause("before terminal commit")
        return super().advance(attempt, step_id, expected, target, **kwargs)


def counts(database):
    return sql(
        database.admin,
        "SELECT "
        "(SELECT count(*) FROM runtime.artifact_metadata),"
        "(SELECT count(*) FROM runtime.artifact_derivation),"
        "(SELECT count(*) FROM runtime.tool_step_execution WHERE result IS NOT NULL),"
        "(SELECT count(*) FROM audit.event)",
    )[0]


def process_count(p):
    return len(list((p.root / "attempts").glob("*/*/process.json")))


def test_real_nonzero_cannot_confirm_cancelled_or_partial_and_recovers_once(make_ready, database):
    p = make_ready("nonzero", repo_type=BeforeConfirmation)
    key = "outcome-nonzero"
    with pytest.raises(Pause):
        p.executor.execute(p.plan, key)
    original = p.repo.workflow
    assert original.status == "FAILED" and original.steps[0].exit_code == 7
    before = counts(database)
    for status in ("CANCELLED", "PARTIAL_FAILED"):
        data = original.model_dump(mode="json")
        data["status"] = status
        for forged in (
            data,
            original.model_copy(update={"status": status}),
            WorkflowResult.model_construct(**data),
        ):
            with pytest.raises(ValueError):
                RuntimeToolExecutions.confirm(p.repo, forged)
            assert counts(database) == before
            stored = p.executor.recover(p.plan.document.run_id, key)
            assert stored["status"] == "ACTIVE" and stored["result"] is None
            assert stored["steps"][0]["result"] == original.steps[0].model_dump(mode="json")
    # A syntactically coherent rehashed decision is still not the persisted decision.
    data = original.model_dump(mode="json")
    data["decision"]["scope_checksum"] = "sha256:" + "e" * 64
    for step in data["steps"]:
        step["binding_checksum"] = digest(
            dict(
                mapping=step["logical_to_physical"],
                copies=step["copied_checksums"],
                decision=data["decision"],
            )
        )
    forged = validate(WorkflowResult, data)
    with pytest.raises(ValueError, match="decision_mismatch"):
        RuntimeToolExecutions.confirm(p.repo, forged)
    assert counts(database) == before
    # Reject a corrupt workflow journal before confirmation, preserving real failed step.
    path = p.root / "attempts" / original.attempt_id / "workflow.json"
    wire = path.read_text()
    damaged = json.loads(wire)
    damaged["status"] = "CANCELLED"
    path.write_text(json.dumps(damaged))
    p.repo.hold = False
    with pytest.raises(ValueError):
        p.executor.recover_commit(p.plan, key)
    assert counts(database) == before
    path.write_text(wire)
    result = p.executor.recover_commit(p.plan, key)
    assert result == original and process_count(p) == 1
    assert p.executor.execute(p.plan, key) == original
    assert p.executor.recover_commit(p.plan, key) == original
    assert counts(database) == (*before[:3], before[3] + 1)


def test_invalid_uncommitted_step_leaves_zero_artifact_derivation_terminal_audit(
    make_ready, database
):
    p = make_ready(repo_type=BeforeTerminal)
    key = "outcome-uncommitted"
    with pytest.raises(Pause):
        p.executor.execute(p.plan, key)
    attempt = p.executor.recover(p.plan.document.run_id, key)
    record = p.repo.record
    assert attempt["steps"][0]["status"] == "RUNNING" and attempt["steps"][0]["result"] is None
    before = counts(database)
    for change in ({"exit_code": 7}, {"error_code": "process_nonzero"}):
        data = record.model_dump(mode="json")
        data.update(change)
        for forged in (data, record.model_copy(update=change), StepResult.model_construct(**data)):
            with pytest.raises(ValueError):
                RuntimeToolExecutions.advance(
                    p.repo, attempt, record.step_id, Status.RUNNING, Status.SUCCEEDED, record=forged
                )
            assert counts(database) == before
    # Valid data with the wrong transition history must also fail before SQL.
    with pytest.raises(ValueError, match="history"):
        RuntimeToolExecutions.advance(
            p.repo, attempt, record.step_id, Status.RUNNING, Status.FAILED, record=record
        )
    path = p.root / "attempts" / record.attempt_id / record.step_id / "result.json"
    wire = path.read_text()
    bad = json.loads(wire)
    bad["exit_code"] = 7
    path.write_text(json.dumps(bad))
    p.repo.hold = False
    with pytest.raises(ValueError):
        p.executor.recover_commit(p.plan, key)
    assert counts(database) == before
    path.write_text(wire)
    result = p.executor.recover_commit(p.plan, key)
    assert result.status == "SUCCEEDED" and process_count(p) == 1
    assert p.executor.recover_commit(p.plan, key) == result


def test_corrupted_pg_readback_and_idempotent_cache_are_rejected(make_ready, database):
    p = make_ready()
    key = "outcome-corruption"
    workflow = p.executor.execute(p.plan, key)
    attempt = p.executor.recover(p.plan.document.run_id, key)
    before = counts(database)
    original = workflow.model_dump(mode="json")
    step = original["steps"][0]
    mutations = [
        (
            "runtime.tool_attempt",
            "result",
            {**original, "status": "CANCELLED"},
            "attempt_id",
            workflow.attempt_id,
        ),
        (
            "runtime.tool_attempt",
            "result",
            {**original, "status": "PARTIAL_FAILED"},
            "attempt_id",
            workflow.attempt_id,
        ),
        (
            "runtime.tool_step_execution",
            "result",
            {**step, "exit_code": 7},
            "attempt_id",
            workflow.attempt_id,
        ),
        (
            "runtime.tool_attempt",
            "decision",
            {**attempt["decision"], "plan_checksum": "sha256:" + "f" * 64},
            "attempt_id",
            workflow.attempt_id,
        ),
    ]
    for table, column, damaged, where, identity in mutations:
        old = deepcopy(
            attempt["decision"]
            if column == "decision"
            else step
            if table.endswith("tool_step_execution")
            else original
        )
        query = f"UPDATE {table} SET {column}=%s WHERE {where}=%s"
        # Explicit corruption of this test's own rows simulates damaged historical storage.
        sql(database.admin, query, (Jsonb(damaged), identity))
        try:
            for read in (
                lambda: p.repo.recover(p.plan.document.run_id, key),
                lambda: p.executor.recover(p.plan.document.run_id, key),
                lambda: p.executor.execute(p.plan, key),
                lambda: p.executor.recover_commit(p.plan, key),
                lambda: RuntimeToolExecutions.confirm(p.repo, workflow),
            ):
                with pytest.raises(ValueError):
                    read()
                assert counts(database) == before
        finally:
            sql(database.admin, query, (Jsonb(old), identity))
    assert p.executor.execute(p.plan, key) == workflow and process_count(p) == 1
    # Simulate an otherwise valid legacy decision in this test's own stored facts.
    # 1.0 had no order field; reads remain byte-for-byte JSON-value compatible, and
    # service idempotency additionally checks order against its controlled plan.
    legacy = deepcopy(original)
    legacy["decision"]["schema_version"] = "execution-decision/1.0.0"
    del legacy["decision"]["planned_step_ids"]
    legacy_step = legacy["steps"][0]
    legacy_step["binding_checksum"] = digest(
        dict(
            mapping=legacy_step["logical_to_physical"],
            copies=legacy_step["copied_checksums"],
            decision=legacy["decision"],
        )
    )
    sql(
        database.admin,
        "UPDATE runtime.tool_step_execution SET result=%s WHERE attempt_id=%s",
        (Jsonb(legacy_step), workflow.attempt_id),
    )
    sql(
        database.admin,
        "UPDATE runtime.tool_attempt SET decision=%s,result=%s WHERE attempt_id=%s",
        (Jsonb(legacy["decision"]), Jsonb(legacy), workflow.attempt_id),
    )
    assert p.repo.recover(p.plan.document.run_id, key)["result"] == legacy
    assert p.executor.execute(p.plan, key).model_dump(mode="json") == legacy
    assert p.executor.recover_commit(p.plan, key).model_dump(mode="json") == legacy
    assert counts(database) == before and process_count(p) == 1
