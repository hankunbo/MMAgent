"""Real admission/compiler/child/QC/result recovery, test ledger and Fake GPU only."""

import json
from pathlib import Path

import pytest

from packages.grounding.models import digest
from packages.object_store import checksum
from packages.tool_execution.trusted_service import TrustedLocalGPUResearchService
from packages.tool_workflow import ExecutionPlan
from tests.tool_admission.support import reseal
from tests.tool_execution.test_trusted_local import setup as setup
from tests.tool_execution.trusted_support import prepared


def test_registered_research_through_qc_result_and_idempotent_recovery(setup):
    f = prepared(setup)
    result = f.service.execute_bundle(f.plan.to_bundle_json(), "same-request")
    assert result.state == "recorded", result
    assert result.workflow.status == "SUCCEEDED", result.model_dump_json(indent=2)
    assert result.workflow.steps[0].result.qc_status.value == "PASS"
    assert f.service.execute(f.plan, "same-request") == result
    assert f.service.recover_commit(f.plan, "same-request") == result.workflow
    assert f.service.recover(f.plan.document.run_id, "same-request")["status"] == "CONFIRMED"
    assert len([x for x in f.repo.transitions if x[1] == "RUNNING"]) == 1
    assert not result.decisive_evidence
    step = result.workflow.steps[0]
    base = Path(f.service._runner.workspace) / result.workflow.attempt_id / step.step_id
    assert checksum((base / "stdout.log").read_bytes()) == step.stdout_checksum
    assert checksum((base / "stderr.log").read_bytes()) == step.stderr_checksum


def test_current_revocation_rejects_before_process(setup):
    f = prepared(setup)
    f.catalogs.disabled = (f.request.candidate.implementation_ref,)
    result = f.service.execute(f.plan, "revoked")
    assert result.state == "technical_rejection"
    assert not f.repo.attempts


@pytest.mark.parametrize("change", ["argv", "parameters", "purpose", "scope", "backend"])
def test_resealed_caller_plan_never_authorizes(setup, change):
    f = prepared(setup)
    doc = f.plan.document.model_dump(mode="json")
    if change == "argv":
        doc["steps"][0]["argv"].append("--unreviewed")
    elif change == "parameters":
        doc["steps"][0]["effective_parameters"]["evil"] = "/etc/passwd"
    elif change == "purpose":
        doc["purpose"] = "decisive_research"
    elif change == "scope":
        doc["scope"]["patient_hash"] = digest("wrong")
    else:
        doc["backend"] = "arbitrary_executor"
    doc.pop("checksum")
    doc["checksum"] = digest(doc)
    forged = ExecutionPlan(json.dumps(doc), f.plan._context_json)
    assert f.service.execute(forged, "forged").state == "technical_rejection"
    assert not f.repo.attempts


def test_resealed_registered_request_still_requires_owner_match(setup):
    f = prepared(setup)
    forged = reseal(f.request, purpose="debug")
    with pytest.raises((ValueError, KeyError)):
        f.f.service.admit_research(forged)


def test_unknown_commit_returns_envelope_no_execution_retry(setup):
    f = prepared(setup)
    actual = f.repo.confirm

    def lost_reply(workflow):
        actual(workflow)
        raise OSError("simulated transport loss")

    f.repo.confirm = lost_reply
    first = f.service.execute(f.plan, "lost")
    assert first.state == "uncertain" and first.workflow is None
    assert first.retry_execution is False
    second = f.service.execute(f.plan, "lost")
    assert second.state == "recorded" and second.workflow.status == "SUCCEEDED"
    assert len([x for x in f.repo.transitions if x[1] == "RUNNING"]) == 1


def test_no_caller_backend_or_approved_argument(setup):
    f = prepared(setup)
    with pytest.raises(TypeError):
        f.service.execute(f.plan, "bad", approved=True)
    with pytest.raises(TypeError):
        TrustedLocalGPUResearchService(backend="anything")


def test_unknown_active_attempt_is_not_replayed(setup):
    f = prepared(setup)
    actual = f.repo.advance

    def lost_start(attempt, step_id, expected, target, **kwargs):
        if target.value == "RUNNING":
            actual(attempt, step_id, expected, target, **kwargs)
            raise RuntimeError("simulated unknown commit of running state")
        return actual(attempt, step_id, expected, target, **kwargs)

    f.repo.advance = lost_start
    first = f.service.execute(f.plan, "uncertain-start")
    assert first.state == "uncertain"
    second = f.service.execute(f.plan, "uncertain-start")
    assert second.state == "uncertain"
    assert len([x for x in f.repo.transitions if x[1] == "RUNNING"]) == 1
    assert f.service.recover(f.plan.document.run_id, "uncertain-start")["result"] is None


def test_versioned_schemas_match_and_old_schemas_stay_readable():
    from pathlib import Path

    from packages.tool_execution import schema_tools, trusted_schema_tools
    from packages.tool_workflow import schema_tools as workflow_schemas

    for module, relative in (
        (schema_tools, "packages/tool_execution/schemas"),
        (trusted_schema_tools, "packages/tool_execution/schemas/trusted-local-v1"),
        (workflow_schemas, "packages/tool_workflow/schemas"),
    ):
        root = Path(__file__).resolve().parents[2] / relative
        for name, content in module.schemas().items():
            assert (root / name).read_text() == content
