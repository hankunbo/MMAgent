from types import SimpleNamespace

import pytest

from packages.grounding.models import digest, validate
from packages.tool_gap_queue import GapQueue, QueuePolicy
from tests.tool_gap_queue.support import setup


def test_real_admission_environment_is_queueable_without_fake_onboarding(tmp_path):
    h = setup(tmp_path)
    a = h.service.prepare_research(h.context.run_id, key="environment")
    assert a.queueable and a.findings[0].kind == "environment_unavailable"
    assert a.material["result"]["capability_gaps"] == []
    assert a.missing == (
        "selected_candidate",
        "evaluation_dataset",
        "threshold_policy",
        "onboarding_job",
    )
    assert h.service.recheck(a) == a


@pytest.mark.parametrize("missing,blocked", [(True, False), (False, True), (True, True)])
def test_missing_inputs_and_safety_dominate_environment(tmp_path, missing, blocked):
    h = setup(tmp_path, missing=missing, blocked=blocked)
    a = h.service.prepare_research(h.context.run_id, key="blocked")
    assert not a.queueable
    assert a.material["result"]["capability_gaps"] == []


@pytest.mark.parametrize("change", ["catalog", "objects", "input"])
def test_owner_technical_failure_never_becomes_empty_or_registered(tmp_path, monkeypatch, change):
    h = setup(tmp_path)
    owner, method = {
        "catalog": (h.f.service.catalogs, "read_snapshot"),
        "objects": (h.f.objects, "read_bounded"),
        "input": (h.f.owners, "read_input"),
    }[change]

    def fail(*args, **kwargs):
        raise OSError("synthetic_owner_offline")

    monkeypatch.setattr(owner, method, fail)
    with pytest.raises(OSError, match="synthetic_owner_offline"):
        h.service.prepare_research(h.context.run_id, key="technical")


def test_revoke_purpose_version_and_forged_classification_fail_recheck(tmp_path):
    h = setup(tmp_path)
    a = h.service.prepare_research(h.context.run_id, key="same")
    for changes in (
        {"purpose": "debug"},
        {"requirement": {"code": "RESEARCH:OTHER"}},
        {"findings": ()},
        {"catalog": {"version": "9.9.9"}},
    ):
        with pytest.raises(ValueError, match="gap_owner_material_changed"):
            h.service.recheck(a.model_copy(update=changes))
    h.f.service.catalogs.disable(h.request.candidate.implementation_ref)
    with pytest.raises(ValueError, match="gap_owner_material_changed"):
        h.service.recheck(a)


def test_policy_is_explicit_and_cannot_allow_missing_data(tmp_path):
    with pytest.raises(ValueError):
        QueuePolicy(allowed=("patient_data_missing",))
    h = setup(tmp_path, policy=QueuePolicy(purposes=("debug",)))
    assert not h.service.prepare_research(h.context.run_id, key="purpose").queueable


def test_05_reconstructs_unknown_producer_and_patient_data(tmp_path):
    from tests.grounding.query_support import updated
    from tests.tool_admission.support import fixture

    f = fixture(tmp_path)
    service = GapQueue(
        repository=None,
        grounding_owner=f.grounding,
        grounding_request_reader=lambda run_id, ref: f.locks.q,
    )
    for patient_types in ([], [i.constraint.artifact_ref for i in f.mapping.candidates[0].inputs]):
        request = updated(f.q, artifacts=[], patient_data_types=patient_types)
        f.locks.q = request
        a = service.prepare_grounding(request, key="data")
        assert not a.queueable
        assert not a.material["result"]["suggestions"]
        assert {r.kind for r in a.findings} & {"patient_data_missing", "unresolved_producer"}


def test_memory_is_separate_and_optional(tmp_path):
    h = setup(tmp_path)
    a = h.service.prepare_research(h.context.run_id, key="memory")
    assert h.service.memory_diagnostics(a.run_id)["status"] == "not_requested"

    def failed(run_id):
        raise OSError("memory_offline")

    h.service.memory = SimpleNamespace(retrieve=failed)
    assert h.service.memory_diagnostics(a.run_id)["status"] == "technical_failure"
    assert h.service.recheck(a) == a


def test_09_intent_is_not_queue_authority_and_unknown_not_rerun():
    from packages.validation_recovery.models import ToolRequirement
    from packages.validation_recovery.requirements import requirement_key
    from tests.validation_recovery.support import fixture

    h = fixture(category="capability_gap")
    before = h.planner.read(h.run_id)
    h.service.resolve(h.run_id, key="11a")
    incident = h.service._read("11a:incident")[0]
    intent = validate(ToolRequirement, incident["requirements"][0])
    service = GapQueue(repository=None, planner=h.planner, documents=h.f.documents)
    a = service.prepare_control(
        h.run_id, gap_checksum=digest(intent.gap), key="intent", intent_key=requirement_key(intent)
    )
    assert a.owner == "09" and not a.queueable
    assert a.intent == intent.model_dump(mode="json")
    assert a.control["binding"] == before.binding.model_dump(mode="json")


def test_05_environment_with_and_without_raw_inputs(tmp_path):
    from tests.grounding.query_support import updated
    from tests.tool_admission.support import fixture

    f = fixture(tmp_path)
    service = GapQueue(
        repository=None,
        grounding_owner=f.grounding,
        grounding_request_reader=lambda run_id, ref: f.locks.q,
    )
    environment = [
        dict(e.model_dump(mode="json"), availability="unavailable") for e in f.q.environment
    ]
    request = updated(f.q, environment=environment)
    f.locks.q = request
    a = service.prepare_grounding(request, key="resolved")
    assert a.queueable
    missing = updated(
        request,
        artifacts=[],
        patient_data_types=[i.constraint.artifact_ref for i in f.mapping.candidates[0].inputs],
    )
    f.locks.q = missing
    b = service.prepare_grounding(missing, key="raw-missing")
    assert not b.queueable
    assert "patient_data_missing" in {f.kind for f in b.findings}


@pytest.mark.parametrize("mode", ["unpublished", "expired", "revoked", "technical"])
def test_memory_original_retrieval_filters_are_not_extension_authority(tmp_path, mode):
    from packages.retrieval.memory import MemoryRetrieval
    from tests.memory_review.support import QueryOwner

    h = setup(tmp_path)
    a = h.service.prepare_research(h.context.run_id, key="optional")

    class Repo:
        contexts = h.service.contexts

        def publication_keys(self):
            return () if mode == "unpublished" else ("synthetic",)

        def read_publication(self, key):
            if mode == "technical":
                raise OSError("offline")
            raise ValueError("memory_qualification_" + mode)

    owner = QueryOwner(a.run_id, a.case_code)
    h.service.memory = MemoryRetrieval(repository=Repo(), query_reader=owner)
    result = h.service.memory_diagnostics(a.run_id)
    assert result["status"] == ("technical_failure" if mode == "technical" else "ok")
    assert not result["extension_authority"]
    assert not any(result["receipt"]["groups"].values())
    assert h.service.recheck(a) == a


def test_different_requirement_versions_purposes_capabilities_do_not_merge(tmp_path):
    from db.runtime.repository.tool_gap_queue import GapQueueRepository
    from packages.grounding.models import seal
    from packages.research_runtime.models import ResearchRunContext
    from tests.tool_admission.support import reseal

    h = setup(tmp_path)
    a = h.service.prepare_research(h.context.run_id, key="identity")
    keys = {GapQueueRepository.task_id(a)}
    for field, value in (("purpose", "debug"), ("version", "1.0.1"), ("code", "RESEARCH:OTHER")):
        req = reseal(h.request, **{field: value})
        h.f.owners.requests[(req.code, req.version, req.checksum)] = req
        context = seal(
            ResearchRunContext, request=req, synthetic=True, scope_checksum=digest(req.scope)
        )
        h.service.contexts = SimpleNamespace(read_research_run=lambda rid, c=context: c)
        b = h.service.prepare_research(context.run_id, key="identity")
        keys.add(GapQueueRepository.task_id(b))
    assert len(keys) == 4
    # An unregistered capability substitution must fail owner registration, not deduplicate.
    candidate = h.request.candidate.model_copy(
        update={
            "capability_ref": h.request.candidate.capability_ref.model_copy(
                update={"code": "CAPABILITY:OTHER"}
            )
        }
    )
    forged = reseal(h.request, candidate=candidate)
    h.service.contexts = SimpleNamespace(
        read_research_run=lambda rid: seal(
            ResearchRunContext, request=forged, synthetic=True, scope_checksum=digest(forged.scope)
        )
    )
    with pytest.raises(KeyError):
        h.service.prepare_research(h.context.run_id, key="identity")


def test_control_pending_cancel_and_version_do_not_create_extension():
    from packages.planner_control.models import Command, Pending
    from tests.validation_recovery.support import fixture

    h = fixture(category="capability_gap")
    service = GapQueue(repository=None, planner=h.planner, documents=h.f.documents)
    state = h.planner.read(h.run_id)
    gap = h.failure.gap
    pending = Pending(
        command=Command(key="original-execute", action="execute", expected_revision=state.revision),
        request_checksum=digest("command"),
        execution_key="original-execution",
    )
    h.runtime.state = state.model_copy(update={"pending": pending})
    a = service.prepare_control(h.run_id, gap_checksum=digest(gap), key="unknown")
    assert not a.queueable and a.findings[0].kind == "execution_unknown"
    assert h.planner.read(h.run_id).pending == pending
    h.runtime.state = state.model_copy(update={"cancelled": True})
    assert not service.prepare_control(
        h.run_id, gap_checksum=digest(gap), key="cancelled"
    ).queueable


def test_original_06_recover_unknown_is_diagnostic_no_relaunch(tmp_path):
    from packages.grounding.models import seal
    from packages.research_runtime.models import ResearchRunContext
    from packages.tool_execution import ToolExecutionService
    from packages.tool_workflow import ResearchContext, ToolWorkflowCompiler
    from tests.tool_admission.support import reseal
    from tests.tool_admission.test_research import research
    from tests.tool_workflow.support import choose, commands, prepared

    f = prepared(tmp_path)
    req = research(f.f, tmp_path)
    reader, invocations = commands(f.f, tmp_path, f.config, catalog_ref=req.lock.catalog_ref)
    invocation = next(
        i
        for i in invocations
        if i.implementation_ref == req.candidate.implementation_ref
        and f.f.owners.manifests[
            (req.lock.catalog_ref, req.candidate.tool_ref, i.implementation_ref)
        ].checksum
        == i.manifest_checksum
    )
    req = reseal(req, invocation=invocation)
    f.f.owners.requests[(req.code, req.version, req.checksum)] = req
    admitted = f.f.service.admit_research(req)
    compiler = ToolWorkflowCompiler(f.f.service, reader, f.config)
    plan = compiler.compile(ResearchContext(request=req, admission=admitted), choose(admitted))
    context = seal(
        ResearchRunContext, request=req, synthetic=True, scope_checksum=digest(req.scope)
    )
    executor = object.__new__(ToolExecutionService)
    # Original read method, a Fake empty execution ledger; no confirmed result is fabricated.
    executor._repo = SimpleNamespace(recover=lambda run_id, key: None)
    service = GapQueue(
        repository=None,
        admission_owner=f.f.service,
        contexts=SimpleNamespace(read_research_run=lambda run_id: context),
        execution_reader=lambda run_id, key: (plan, executor),
    )
    a = service.prepare_execution(context.run_id, execution_key="unconfirmed", key="unknown")
    assert not a.queueable and a.findings[0].kind == "execution_unknown"
    assert a.material["workflow"] is None and service.recheck(a) == a


def test_05_availability_cannot_be_supplied_by_caller(tmp_path):
    from tests.grounding.query_support import updated
    from tests.tool_admission.support import fixture

    f = fixture(tmp_path)
    service = GapQueue(repository=None, grounding_owner=f.grounding)
    with pytest.raises(ValueError, match="gap_grounding_request_owner_required"):
        service.prepare_grounding(f.q, key="unowned")
    service.grounding_request_reader = lambda run_id, ref: f.q
    with pytest.raises(ValueError, match="gap_grounding_request_changed"):
        service.prepare_grounding(updated(f.q, artifacts=[]), key="forged-input")


def test_input_source_change_invalidates_consumption(tmp_path):
    from packages.grounding.synthetic import ref

    h = setup(tmp_path)
    a = h.service.prepare_research(h.context.run_id, key="source")
    code, record = next(iter(h.f.owners.records.items()))
    h.f.owners.records[code] = record.model_copy(
        update={"source_ref": ref("source_version", "SOURCE:CHANGED")}
    )
    with pytest.raises(ValueError, match="gap_owner_material_changed"):
        h.service.recheck(a)


@pytest.mark.parametrize("changed", ["manifest", "probe"])
def test_source_binding_failures_never_become_extension(tmp_path, changed):
    from tests.tool_admission.support import reseal

    h = setup(tmp_path)
    if changed == "manifest":
        key = (
            h.request.lock.catalog_ref,
            h.request.candidate.tool_ref,
            h.request.candidate.implementation_ref,
        )
        manifest = h.f.owners.manifests[key]
        h.f.owners.manifests[key] = reseal(manifest, config_checksum=digest("changed-source"))
    else:
        for key, probe in list(h.f.owners.probes.items()):
            h.f.owners.probes[key] = reseal(probe, runtime_checksum=digest("wrong-runtime"))
    a = h.service.prepare_research(h.context.run_id, key="source-binding")
    assert not a.queueable
    assert "source_safety_blocked" in {f.kind for f in a.findings}
