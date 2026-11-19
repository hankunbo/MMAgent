"""CPU-only method review/admission/compiler probes; no PostgreSQL or execution.

The in-memory input/request owner is explicitly synthetic here. Original dataset,
component projection, parent control, and execution owners are exercised by the
separate PostgreSQL integration test, not claimed by this preparation test.
"""

from types import SimpleNamespace

import pytest

from packages.grounding.compatibility import artifact_compatibility
from packages.grounding.models import digest, seal, validate
from packages.object_store import checksum
from packages.research_runtime.registration import single_choice
from packages.tool_admission import ToolAdmissionService
from packages.tool_admission.method_metadata import metadata_requirements, validate_review_metadata
from packages.tool_admission.models import AdmissionPolicy, InputRecord, ResearchRequest
from packages.tool_workflow import ResearchContext, ToolWorkflowCompiler
from tests.dataset_adapters.test_multimodal import jpeg
from tests.grounding.query_support import NOW
from tests.research_runtime.reviewed_component_tool_support import deployment
from tests.tool_admission.support import reseal


def installed(root, **metadata_requirements):
    """Synthetic current input facts with original review/admission/compiler owners."""
    ready = deployment(root, **metadata_requirements)
    base, policy = ready.base, ready.component_policy
    old = next(iter(base.owners.records.values()))
    raw = jpeg()
    uri = base.objects.promote(base.objects.stage(raw), checksum(raw))
    common = dict(
        checksum=checksum(raw),
        qc_status="UNASSESSED",
        laterality="UNKNOWN",
        coordinate_space="PIXEL_2D",
    )
    artifact = old.artifact.model_dump(mode="json")
    artifact.update(
        **common,
        artifact_type_code=policy.input_binding.contract_type_code,
        uri=uri,
        media_type="image/jpeg",
        unit=None,
    )
    metadata = old.metadata.model_dump(mode="json")
    metadata.update(
        **common,
        artifact_ref=policy.input_binding.artifact_ref,
        modality=policy.modality,
        unit=policy.input_binding.unit_term,
        scope_terms=policy.scope_terms,
        acquired_at=None,
    )
    record = validate(
        InputRecord,
        dict(
            artifact=artifact,
            metadata=metadata,
            type_binding=policy.input_binding,
            byte_size=len(raw),
            content_kind="image",
            source_ref=policy.input_binding.source_ref,
        ),
    )
    base.owners.records[record.artifact.asset_code] = record
    request = seal(
        ResearchRequest,
        code="RESEARCH:SYNTHETIC:REVIEWED_COMPONENT_CPU",
        version="1.0.0",
        purpose="shadow",
        compatibility_policy=ready.compatibility,
        lock=dict(run_id=base.q.lock.run_id, catalog_ref=ready.catalog.reference()),
        scope=base.q.scope.model_copy(update={"laterality": "UNKNOWN"}),
        window_start=NOW,
        window_end=NOW,
        snapshot_at=NOW,
        satisfied_terms=policy.scope_terms,
        artifacts=(record.metadata,),
        candidate=ready.candidate,
        invocation=ready.invocation,
        inputs=[
            dict(port=ready.candidate.inputs[0].port, asset_codes=(record.artifact.asset_code,))
        ],
    )
    base.owners.requests[(request.code, request.version, request.checksum)] = request
    admission = ToolAdmissionService(
        policy=AdmissionPolicy(max_total_bytes=268435456),
        grounding=base.grounding,
        catalogs=ready.catalogs,
        manifests=ready.catalogs,
        profiles=ready.owners,
        probes=ready.owners,
        inputs=base.owners,
        research=base.owners,
        objects=base.objects,
        resources=ready.resources,
        clock=lambda: NOW,
    )
    compiler = ToolWorkflowCompiler(admission, ready.allowlists, ready.config)
    return SimpleNamespace(
        ready=ready,
        base=base,
        request=request,
        record=record,
        admission=admission,
        compiler=compiler,
    )


def test_reviewed_method_preserves_unknown_facts_and_compiles_exact_jpeg(tmp_path):
    a = installed(tmp_path)
    d, request = a.ready, a.request
    assert d.catalogs.read_snapshot(d.catalog.reference()) == d.catalog
    assert (
        d.catalogs.read_manifest(
            d.catalog.reference(), d.candidate.tool_ref, d.candidate.implementation_ref
        )
        == d.manifest
    )
    admitted = a.admission.admit_research(request)
    assert admitted.status == "preparation_only"
    assert admitted.retained_roots and not admitted.rejected
    plan = a.compiler.compile(
        ResearchContext(request=request, admission=admitted), single_choice(admitted)
    )
    a.compiler.validate_plan(plan)
    assert len(plan.document.steps) == 1
    assert request.scope.laterality.value == "UNKNOWN"
    assert request.artifacts[0].laterality.value == "UNKNOWN"
    assert request.artifacts[0].acquired_at is None
    assert request.artifacts[0].unit.state == "not_applicable"
    assert a.record.artifact.media_type == "image/jpeg"
    assert a.base.objects.read_bounded(a.record.artifact.uri, 4194304) == jpeg()
    assert (
        a.base.owners.read_input(request.scope, request.lock.run_id, a.record.artifact.asset_code)
        == a.record
    )
    # These flags describe the original native/review projection path. Fixture
    # provenance and all consumed bytes remain synthetic; no medical model ran.
    assert d.catalog.synthetic is False
    assert plan.document.steps[0].deployment.allowlist.synthetic_only is False


@pytest.fixture(scope="module")
def method(tmp_path_factory):
    return installed(tmp_path_factory.mktemp("method-metadata"))


def manifest_with(a, **updates):
    metadata = reseal(a.ready.metadata, **updates)
    return reseal(
        a.ready.manifest,
        preprocessing_policy={
            **a.ready.manifest.preprocessing_policy,
            "metadata_requirements": metadata.model_dump(mode="json"),
        },
    )


@pytest.mark.parametrize(
    "eye,time",
    [
        ("required", "required"),
        ("required", "not_consumed"),
        ("not_consumed", "required"),
        ("not_consumed", "not_consumed"),
    ],
)
def test_independent_eye_time_flags(method, eye, time):
    a = method
    manifest = manifest_with(a, laterality=eye, acquisition_time=time)
    flags = metadata_requirements(manifest, a.ready.candidate, "shadow", a.ready.catalogs.tool)
    reasons = artifact_compatibility(
        a.record.metadata,
        a.ready.candidate.inputs[0].constraint,
        a.request,
        laterality_required=flags.laterality_required,
        acquisition_time_required=flags.acquisition_time_required,
    )
    assert ("required_condition_unknown" in reasons) == (eye == "required")
    assert ("time_unknown" in reasons) == (time == "required")
    if eye == time == "not_consumed":
        assert reasons == ()


@pytest.mark.parametrize(
    "field",
    [
        "execution_declaration_checksum",
        "input_constraints_checksum",
        "output_constraints_checksum",
        "parameter_schema_checksum",
        "preprocessing_checksum",
        "postprocessing_checksum",
    ],
)
def test_precise_method_binding_drift_rejected(method, field):
    a = method
    manifest = manifest_with(a, **{field: digest("changed method")})
    with pytest.raises(ValueError, match="research_method_metadata_binding"):
        metadata_requirements(manifest, a.ready.candidate, "shadow", a.ready.catalogs.tool)


@pytest.mark.parametrize("which", ["input", "output"])
def test_unknown_not_qualified_for_eye_specific_method(method, which):
    from packages.contracts import Laterality

    a = method
    review = a.ready.review
    field = which + "_constraints"
    constraints = tuple(
        c.model_copy(update=dict(laterality=Laterality.LEFT)) for c in getattr(review, field)
    )
    metadata = reseal(a.ready.metadata, **{field + "_checksum": digest(constraints)})
    review = reseal(
        review,
        **{
            field: constraints,
            "preprocessing_policy": {
                **review.preprocessing_policy,
                "metadata_requirements": metadata.model_dump(mode="json"),
            },
        },
    )
    with pytest.raises(ValueError, match="research_method_metadata_binding"):
        validate_review_metadata(review)


def test_missing_unit_and_known_eye_conflict_still_fail(method):
    from packages.contracts import Laterality

    a = method
    constraint = a.ready.candidate.inputs[0].constraint
    flags = dict(laterality_required=False, acquisition_time_required=False)
    unknown_unit = a.record.metadata.model_copy(
        update=dict(unit=constraint.unit.model_copy(update=dict(state="unknown", code=None)))
    )
    assert "unit_mismatch" in artifact_compatibility(unknown_unit, constraint, a.request, **flags)
    known = a.record.metadata.model_copy(update=dict(laterality=Laterality.LEFT))
    request = a.request.model_copy(
        update=dict(scope=a.request.scope.model_copy(update=dict(laterality=Laterality.RIGHT)))
    )
    assert "laterality_mismatch" in artifact_compatibility(known, constraint, request, **flags)


def test_formal_default_purpose_and_candidate_scope(method):
    a = method
    strict = metadata_requirements(a.ready.manifest, a.ready.candidate, None, a.ready.catalogs.tool)
    assert strict.laterality_required and strict.acquisition_time_required
    with pytest.raises(ValueError, match="research_method_metadata_binding"):
        metadata_requirements(
            a.ready.manifest, a.ready.candidate, "clinical", a.ready.catalogs.tool
        )
    with pytest.raises(ValueError, match="research_method_metadata_binding"):
        metadata_requirements(a.ready.manifest, a.ready.candidate, "shadow")
    candidate = a.ready.candidate.model_copy(
        update=dict(
            output=a.ready.candidate.output.model_copy(
                update=dict(scope_terms=("CONCEPT:UNREVIEWED",))
            )
        )
    )
    with pytest.raises(ValueError, match="research_method_metadata_binding"):
        metadata_requirements(a.ready.manifest, candidate, "shadow", a.ready.catalogs.tool)


def test_reviewed_file_drift_revokes_current_read(method):
    a = method
    before = a.ready.audit.read_bytes()
    try:
        a.ready.audit.write_bytes(before + b"changed")
        with pytest.raises(ValueError, match="trusted_resource_size_drift"):
            a.ready.catalogs.read_snapshot(a.ready.catalog.reference())
    finally:
        a.ready.audit.write_bytes(before)
    assert a.ready.catalogs.read_snapshot(a.ready.catalog.reference()) == a.ready.catalog


def test_new_declaration_version_cannot_fall_back_to_legacy(method):
    a = method
    pre = dict(a.ready.manifest.preprocessing_policy)
    pre["metadata_requirements"] = {**pre["metadata_requirements"], "schema_version": "unknown/99"}
    manifest = reseal(a.ready.manifest, preprocessing_policy=pre)
    with pytest.raises(ValueError, match="research_metadata_declaration_binding"):
        metadata_requirements(manifest, a.ready.candidate, "shadow", a.ready.catalogs.tool)


def test_metadata_types_import_in_both_orders():
    import subprocess
    import sys

    for first in ("packages.tool_admission", "packages.tool_execution"):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    f"import {first}; "
                    "from packages.grounding.review_files import ReviewedFile as A; "
                    "from packages.tool_execution.trusted_models import ReviewedFile as B; "
                    "assert A is B"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert result.returncode == 0, result.stderr


def test_parent_prepare_requirements_exact_json_roundtrip(method):
    import json

    from packages.grounding.models import canonical
    from packages.research_runtime.component_steps import ComponentPlannerSteps
    from tests.research_runtime.test_component_steps import state

    a = method
    admitted = a.admission.admit_research(a.request)
    plan = a.compiler.compile(
        ResearchContext(request=a.request, admission=admitted), single_choice(admitted)
    )
    control, _ = state()
    owner = object.__new__(ComponentPlannerSteps)
    owner.requests = SimpleNamespace(registration_binding={"test": "synthetic"}, request_key="k")
    owner.work_deadline = control.binding.deadline
    context = SimpleNamespace(
        request=a.request, run_id=plan.document.run_id, checksum=digest("ctx")
    )
    requirements = dict(
        policy=digest("policy"),
        outputs=[
            dict(
                port=plan.document.steps[0].outputs[0].port,
                constraint=plan.document.steps[0].outputs[0].constraint.model_dump(mode="json"),
                qc=plan.document.steps[0].outputs[0].qc_policy_ref.model_dump(mode="json"),
                policy=digest("qc policy"),
            )
        ],
    )
    receipt = owner._receipt(control, plan, context, requirements)
    assert receipt == json.loads(canonical(receipt))
    assert canonical(receipt["execution_requirements"]) == canonical(requirements)
