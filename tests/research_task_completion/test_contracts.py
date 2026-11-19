"""Small semantic contracts; these are not persistent success or PG evidence."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from db.memory.publication import PublicationRepository
from packages.evidence_details.current_rejection import CurrentResearchRejected
from packages.grounding.models import digest, seal, validate
from packages.memory_review.models import Binding, Query
from packages.memory_review.task_models import TaskBinding, TaskQuery, qualification
from packages.research_runtime.task_completion import ResearchTaskCompletion
from packages.research_runtime.task_models import TaskCompletion, TaskDefinition
from packages.retrieval.memory.service import mismatch
from tests.memory_review.support import binding, conditions


def proof(run="RUN:A", case="CASE:A"):
    d = seal(
        TaskDefinition,
        goal="GOAL:SYNTHETIC",
        profile=digest("task-profile"),
        run_id=run,
        case_code=case,
        context_checksum=digest("context"),
        request_checksum=digest("request"),
        scope_checksum=digest("scope"),
        plan_checksum=digest("plan"),
        config_checksum=digest("config"),
        execution_policy=digest("execution-policy"),
        score_policy=digest("score-policy"),
        requirements={"outputs": [], "scores": []},
        validity_seconds=3600,
    )
    now = datetime.now(UTC)
    return seal(
        TaskCompletion,
        definition=d,
        lifecycle_checksum=digest("lifecycle"),
        report_checksum=digest("report"),
        workflow_checksum=digest("workflow"),
        score_checksum=digest("score"),
        output_checksums=(digest("output"),),
        valid_from=now,
        expires_at=now + timedelta(seconds=3600),
    )


def task_binding(proofs=None):
    proofs = proofs or (proof(),)
    old = binding().model_dump(exclude={"schema_version", "group"})
    return TaskBinding(**old, task_profile=proofs[0].definition.profile, proofs=proofs)


def test_explicit_dispatch_old_query_never_authorizes_task_success():
    b = task_binding()
    old = Query(
        **conditions().model_dump(), run_id="RUN:Q", case_code="CASE:Q", test_context="test"
    )
    new = TaskQuery(**old.model_dump(exclude={"schema_version"}), task_profile=b.task_profile)
    assert qualification(Binding, b) == b and qualification(Query, new) == new
    assert mismatch(b, old) == "memory_success_semantics"
    assert mismatch(b, new) is None
    medical = binding().model_copy(update={"group": "success"})
    assert mismatch(medical, new) == "memory_success_semantics"
    assert mismatch(binding(), new) is None  # C may still opt into old recovery.
    with pytest.raises(ValueError):
        validate(Binding, b)
    with pytest.raises(ValueError):
        validate(Query, new)
    assert (
        mismatch(b, new.model_copy(update={"task_profile": digest("wrong")}))
        == "memory_task_profile"
    )


@pytest.mark.parametrize("field,value", [("medical_success", True), ("checksum", digest("forged"))])
def test_constructed_or_mutated_proof_is_revalidated(field, value):
    b = task_binding()
    bad = b.proofs[0].model_copy(update={field: value})
    with pytest.raises(ValueError):
        qualification(Binding, b.model_copy(update={"proofs": (bad,)}))


def test_proof_scope_task_goal_and_duplicate_runs():
    p = proof()
    for changes in (
        {"proofs": (p, p)},
        {"task_profile": digest("other")},
        {"conditions": conditions().model_copy(update={"goal": "GOAL:OTHER"})},
    ):
        with pytest.raises(ValueError):
            qualification(Binding, task_binding().model_copy(update=changes))


def test_direct_repository_rebuilds_complete_support_set_and_current_proof():
    b = task_binding()
    source = SimpleNamespace(run_id="RUN:A")
    record = SimpleNamespace(sources=(source,))
    scope = SimpleNamespace(patient_hash=digest("patient"), case_code="CASE:A")
    repo = object.__new__(PublicationRepository)
    repo.clock = lambda: b.proofs[0].valid_from
    repo.contexts = SimpleNamespace(
        read_research_run=lambda run: SimpleNamespace(request=SimpleNamespace(scope=scope))
    )
    repo.sources = SimpleNamespace(read_task_completion=lambda run: (b.proofs[0], source))
    repo._task_target(b, record)
    with pytest.raises(CurrentResearchRejected, match="support_set"):
        repo._task_target(b, SimpleNamespace(sources=(source, SimpleNamespace(run_id="RUN:B"))))
    repo.sources.read_task_completion = lambda run: (proof(), source)
    with pytest.raises(CurrentResearchRejected, match="proof_changed"):
        repo._task_target(b, record)
    repo.sources.read_task_completion = lambda run: (_ for _ in ()).throw(OSError("storage down"))
    with pytest.raises(OSError, match="storage down"):
        repo._task_target(b, record)


def test_no_after_the_fact_proof_and_error_classification(monkeypatch):
    driver = SimpleNamespace(runtime=SimpleNamespace(lifecycle=lambda rid: {"binding": {}}))
    owner = ResearchTaskCompletion(driver, goal="GOAL:SYNTHETIC")
    with pytest.raises(CurrentResearchRejected, match="not_bound"):
        owner.read("RUN:A")
    for failure in (OSError("storage"), ValueError("unknown_read_failure")):
        with monkeypatch.context() as m:
            m.setattr(owner, "_read", lambda rid, failure=failure: (_ for _ in ()).throw(failure))
            with pytest.raises(type(failure), match=str(failure)):
                owner.read("RUN:A")
    with monkeypatch.context() as m:
        m.setattr(
            owner,
            "_read",
            lambda rid: (_ for _ in ()).throw(ValueError("research_input_fact_mismatch")),
        )
        with pytest.raises(CurrentResearchRejected, match="source_changed"):
            owner.read("RUN:A")


def test_task_review_prompt_is_reference_projection_not_task_body():
    from packages.memory_review.models import ReviewInput
    from packages.memory_review.task_models import TaskReviewInput

    b = task_binding()
    payload = TaskReviewInput.project(scope={}, binding=b, sources=(digest("source"),))
    assert payload.binding_checksum == digest(b)
    assert payload.task_proofs == (b.proofs[0].checksum,)
    assert "requirements" not in payload.model_dump_json()
    assert "score_policy" not in payload.model_dump_json()
    assert qualification(ReviewInput, payload) == payload


@pytest.mark.parametrize("field", ("plan_checksum", "request_checksum", "scope_checksum"))
def test_resealed_wrong_plan_input_or_scope_cannot_self_authorize(field):
    b = task_binding()
    original = b.proofs[0]
    definition = seal(
        TaskDefinition,
        **{**original.definition.model_dump(exclude={"checksum"}), field: digest("caller-forgery")},
    )
    forged = seal(
        TaskCompletion,
        **{**original.model_dump(exclude={"checksum", "definition"}), "definition": definition},
    )
    binding_with_forgery = b.model_copy(update={"proofs": (forged,)})
    source = SimpleNamespace(run_id=original.definition.run_id)
    repo = object.__new__(PublicationRepository)
    repo.clock = lambda: original.valid_from
    scope = SimpleNamespace(patient_hash=digest("patient"), case_code=original.definition.case_code)
    repo.contexts = SimpleNamespace(
        read_research_run=lambda run: SimpleNamespace(request=SimpleNamespace(scope=scope))
    )
    repo.sources = SimpleNamespace(read_task_completion=lambda run: (original, source))
    with pytest.raises(CurrentResearchRejected, match="proof_changed"):
        repo._task_target(
            qualification(Binding, binding_with_forgery), SimpleNamespace(sources=(source,))
        )


def test_opted_in_execution_source_rechecks_task_at_writer_boundary(monkeypatch):
    from apps.memory_execution_sources import ExecutionSources

    p = proof()
    runtime = dict(
        synthetic=True,
        lifecycle_phase="FINISHED",
        runtime_status="COMPLETED_INSUFFICIENT_EVIDENCE",
        lifecycle_checksum=p.lifecycle_checksum,
    )
    task_owner = SimpleNamespace(read_task_completion=lambda run: p)
    source = ExecutionSources(
        runtime=SimpleNamespace(read_memory_execution=lambda run: runtime),
        run_id=p.definition.run_id,
        applicability=("USE:SYNTHETIC",),
        task_completion=task_owner,
    )
    assert source.snapshot()["runtime"] == runtime
    with monkeypatch.context() as m:
        m.setattr(
            task_owner,
            "read_task_completion",
            lambda run: (_ for _ in ()).throw(
                CurrentResearchRejected("task_completion_output_drift")
            ),
        )
        with pytest.raises(CurrentResearchRejected, match="output_drift"):
            source.snapshot()
