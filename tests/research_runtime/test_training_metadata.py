"""Synthetic Training bytes through real 07A/registration/06/H/Runtime ports.

Only external native catalog/target and GPU observation are simulated. No model
forward, real patient, input reader substitution or production installation.
"""

import copy
import json
import os
from pathlib import Path

import pytest

from db import manage
from packages.contracts import Laterality
from packages.grounding.compatibility import artifact_compatibility, constraint_compatibility
from packages.grounding.models import canonical, digest
from packages.object_store import checksum
from packages.research_runtime import deployment
from packages.tool_admission.research_metadata import FAIRCLIP_METADATA, limited_metadata
from tests.dataset_adapters.support import fair
from tests.grounding.test_database import database as database
from tests.integrations.medgraph.test_research import pin
from tests.integrations.medgraph.test_research import reviewed as reviewed
from tests.research_runtime.test_deployment_factory import installed_shape
from tests.tool_admission.support import reseal
from tests.tool_execution.test_trusted_local import setup as setup


def test_metadata_is_bound_by_actual_reviewed_catalog(reviewed):
    from packages.fairclip_research.core import CONFIG, CONFIG_CHECKSUM
    from packages.integrations.medgraph.research import ReviewedCatalog

    f = reviewed
    original = canonical(f.reader.read_snapshot(f.catalog.reference()))
    before = ReviewedCatalog(f.reader, pin(f.record), f.policy)
    review = reseal(
        f.review,
        preprocessing_policy={
            "version": "1.0.0",
            "policy_code": "POLICY:TEST:METADATA",
            "fixed_preprocess": CONFIG["preprocess"],
            "metadata_requirements": FAIRCLIP_METADATA,
        },
        postprocessing_policy={"classification_checksum": CONFIG_CHECKSUM},
    )
    f.record.write_bytes(canonical(review))
    after = ReviewedCatalog(f.reader, pin(f.record), f.policy)
    assert after.manifest.preprocessing_policy["metadata_requirements"] == FAIRCLIP_METADATA
    assert after.manifest.review_ref.checksum == review.checksum
    assert after.tool.implementation_ref != before.tool.implementation_ref
    assert after.catalog.reference() != before.catalog.reference()
    assert canonical(f.reader.read_snapshot(f.catalog.reference())) == original
    with pytest.raises(ValueError):
        before.read_snapshot(before.catalog.reference())
    f.record.write_bytes(canonical(reseal(review, purposes=["debug"])))
    with pytest.raises(ValueError, match="research_metadata_review_purpose_mismatch"):
        ReviewedCatalog(f.reader, pin(f.record), f.policy)


@pytest.mark.parametrize(
    "change",
    [
        "version",
        "use",
        "time_dependent",
        "eye_dependent",
        "classification",
        "review",
        "implementation",
        "purpose",
    ],
)
def test_reviewed_metadata_exact_binding(setup, monkeypatch, change):
    f = installed_shape(setup, monkeypatch, training_metadata=True)
    manifest, candidate, purpose = f.manifest, f.candidate, "shadow"
    pre = copy.deepcopy(manifest.preprocessing_policy)
    declaration = pre["metadata_requirements"]
    if change in {"version", "use"}:
        declaration[change] = "unapproved"
    elif change == "time_dependent":
        declaration["acquired_at"] = "required"
    elif change == "eye_dependent":
        candidate = candidate.model_copy(
            update={"output": candidate.output.model_copy(update={"laterality": Laterality.LEFT})}
        )
    elif change == "classification":
        declaration["classification_checksum"] = digest("other model configuration")
    elif change == "review":
        manifest = reseal(manifest, policy_checksum=digest("another review"))
    elif change == "implementation":
        candidate = candidate.model_copy(
            update={
                "implementation_ref": candidate.implementation_ref.model_copy(
                    update={"version": "999.0.0"}
                )
            }
        )
    else:
        purpose = "clinical"
    manifest = reseal(manifest, preprocessing_policy=pre)
    with pytest.raises(ValueError, match="research_metadata_declaration_binding"):
        limited_metadata(manifest, candidate, purpose)
    assert limited_metadata(manifest, candidate, None) is False  # formal never opts in


def test_default_strict_and_known_conflicts(setup, monkeypatch):
    f = installed_shape(setup, monkeypatch, training_metadata=True)
    constraint = f.candidate.inputs[0].constraint
    q = f.f.q.model_copy(
        update={"scope": f.f.q.scope.model_copy(update={"laterality": Laterality.UNKNOWN})}
    )
    original = f.input_record.metadata
    a = original.model_copy(
        update=dict(
            laterality=Laterality.UNKNOWN,
            acquired_at=None,
            coordinate_space=constraint.coordinate_space,
            unit=constraint.unit,
            qc_status=constraint.qc_required[0],
        )
    )
    assert "required_condition_unknown" in constraint_compatibility(constraint, q)
    assert "time_unknown" in artifact_compatibility(a, constraint, q)
    assert not artifact_compatibility(
        a, constraint, q, laterality_required=False, acquisition_time_required=False
    )
    # Time-dependent tools retain time_unknown even when eye metadata is not needed.
    assert "time_unknown" in artifact_compatibility(a, constraint, q, laterality_required=False)
    for left, right in [(Laterality.LEFT, Laterality.RIGHT), (Laterality.RIGHT, Laterality.LEFT)]:
        known = a.model_copy(update={"laterality": left})
        scope = q.model_copy(update={"scope": q.scope.model_copy(update={"laterality": right})})
        assert "laterality_mismatch" in artifact_compatibility(
            known, constraint, scope, laterality_required=False, acquisition_time_required=False
        )
        assert "laterality_mismatch" in constraint_compatibility(
            constraint.model_copy(update={"laterality": left}), scope, laterality_required=False
        )
    cross = a.model_copy(update={"case_code": "CASE:OTHER"})
    assert "scope_mismatch" in artifact_compatibility(
        cross, constraint, q, laterality_required=False, acquisition_time_required=False
    )


def test_training_registration_admission_execution_and_recovery_pg(database, setup, monkeypatch):
    f = installed_shape(
        setup, monkeypatch, target=manage.test_target("runtime"), training_metadata=True
    )
    adapter, _ = fair(setup.root / "synthetic-training")
    sample = adapter.prepare("data_00000.npz", split="training")
    f.bundle["training_selection"] = dict(
        root=str(setup.root / "synthetic-training"),
        filename="data_00000.npz",
        dataset_ref=sample.dataset.reference(),
        pack_ref=sample.pack.reference(),
    )
    f.config["historical_visible_image_checksum"] = checksum(sample.images[0][0])
    f.write_bundle()
    assembled = deployment.assemble(f.config, object())
    service, requests = assembled.service, assembled.requests
    admission = service.research.compiler.admission
    assert admission.inputs is requests and service.research.inputs is requests
    # Public factory preparation, actual 07A capture/register and persisted visible read.
    sample = assembled.prepare_sample()
    registration = service.preparation.register(sample, key="metadata-case:visible")
    visible = service.preparation.read_visible(registration.visible.case.case_code)
    assert visible == registration.visible
    request = requests.register(visible, "metadata-case")
    assert requests.read_request(request.code, request.version, request.checksum) == request
    with pytest.raises(ValueError):
        reseal(request, metadata_requirements=FAIRCLIP_METADATA)
    assert request.scope.laterality == Laterality.UNKNOWN
    record = requests.read_input(
        request.scope, request.lock.run_id, request.artifacts[0].asset_code
    )
    assert record.metadata.laterality == Laterality.UNKNOWN
    assert record.metadata.acquired_at is None
    serialized = request.model_dump_json()
    assert not any(x in serialized for x in ["SECRET_DIAGNOSIS", "SECRET_SUMMARY", "glaucoma"])
    assert admission.admit_research(request).retained_roots

    # Missing/invalid declarations are loaded from the controlled manifest port,
    # never from caller input. Registration/readback remain the original methods.
    original_invocation = requests.invocation
    for label, declaration in (
        ("missing", None),
        ("version", {**FAIRCLIP_METADATA, "version": "999.0.0"}),
        ("use", {**FAIRCLIP_METADATA, "use": "longitudinal_comparison"}),
        ("time", {**FAIRCLIP_METADATA, "acquired_at": "required"}),
    ):
        pre = copy.deepcopy(f.manifest.preprocessing_policy)
        pre["metadata_requirements"] = declaration
        changed = reseal(f.manifest, preprocessing_policy=pre)
        with monkeypatch.context() as mutation:
            mutation.setattr(f.catalogs, "read_manifest", lambda *args, m=changed: m)
            requests.invocation = original_invocation.model_copy(
                update={"manifest_checksum": changed.checksum}
            )
            negative = requests.register(visible, "metadata-negative-" + label)
            assert (
                requests.read_request(negative.code, negative.version, negative.checksum)
                == negative
            )
            rejected = admission.admit_research(negative)
            assert not rejected.retained_roots
            reasons = {r.reason for r in rejected.rejected}
            assert (
                "required_condition_unknown"
                if declaration is None
                else "research_metadata_declaration_binding"
            ) in reasons
    requests.invocation = original_invocation

    other = service.preparation.register(sample, key="metadata-other:visible").visible
    other_request = requests.register(other, "metadata-other")
    with pytest.raises((KeyError, StopIteration, ValueError)):
        requests.read_input(
            other_request.scope, other_request.lock.run_id, record.artifact.asset_code
        )
    with pytest.raises(ValueError, match="research_scope_mismatch"):
        reseal(request, scope=other_request.scope)
    forged = reseal(request, version="999.0.0")
    with pytest.raises(ValueError, match="research_registration_identity"):
        admission.admit_research(forged)

    # Public single-case entry repeats idempotent 07A preparation and registration.
    report = service.run(sample, key="metadata-case")
    assert report["technical_status"] == "SUCCEEDED", json.dumps(report, indent=2)
    assert report["run_status"] == "COMPLETED_INSUFFICIENT_EVIDENCE"
    assert report["graph"] is None and report["graph_scores"] == "unavailable"
    assert not report["accepted_validations"]
    assert report["model_outputs"][0]["output"]["normalized_model_scores"] == [0.5, 0.5]
    assert service.runtime.lifecycle(request.lock.run_id)["phase"] == "FINISHED"
    assert service.research.read(request.lock.run_id).request == request
    from packages.tool_execution.trusted_service import TrustedLocalGPUResearchService

    original_bind = service.bind_plan
    with monkeypatch.context() as failure:

        def lost_ack(plan, key):
            executor, scores = original_bind(plan, key)
            failure.setattr(
                executor._repo,
                "confirm",
                lambda *a: (_ for _ in ()).throw(RuntimeError("synthetic_ack_lost")),
            )
            return executor, scores

        failure.setattr(service, "bind_plan", lost_ack)
        uncertain = service.run(sample, key="metadata-unknown")
        still_unknown = service.run_registered(
            requests.for_key("metadata-unknown"), key="metadata-unknown", recover_only=True
        )
        assert still_unknown == uncertain
    assert uncertain["technical_status"] == "UNKNOWN_JOB" and not uncertain["terminal"]
    unknown_request = requests.for_key("metadata-unknown")
    assert service.runtime.lifecycle(unknown_request.lock.run_id)["phase"] == "UNKNOWN"
    monkeypatch.setattr(
        TrustedLocalGPUResearchService,
        "execute",
        lambda *a, **k: pytest.fail("idempotent recovery must not execute twice"),
    )
    assert service.run_registered(request, key="metadata-case", recover_only=True) == report
    assert service.run(sample, key="metadata-case") == report
    recovered = service.run_registered(unknown_request, key="metadata-unknown", recover_only=True)
    assert recovered["technical_status"] == "SUCCEEDED" and recovered["terminal"]
    assert service.runtime.lifecycle(unknown_request.lock.run_id)["phase"] == "FINISHED"
    assert requests.for_key("metadata-case").artifacts[0].acquired_at is None
    assert not (setup.lock_root / "active.json").exists()
    evidence = os.environ.get("MMAGENT_D01_METADATA_EVIDENCE")
    if evidence:
        path = Path(evidence).resolve()
        assert path.is_relative_to(deployment.ROOT / ".local")
        (path / "synthetic-training-report.json").write_text(
            json.dumps(
                dict(
                    kind="SYNTHETIC_TRAINING_REAL_REGISTRATION_PORTS",
                    actual_Training_admitted=False,
                    model_forward=False,
                    input_ports_substituted=False,
                    native_catalog_and_GPU_observation="test doubles",
                    request=request.model_dump(mode="json"),
                    report=report,
                    unknown=uncertain,
                    recovered=recovered,
                ),
                indent=2,
            )
        )


def test_readiness_requires_same_reviewed_declaration(setup, monkeypatch):
    f = installed_shape(setup, monkeypatch)
    monkeypatch.setattr(deployment, "ResearchLifecycleRepository", lambda target: object())
    f.config["metadata_requirements"] = {**FAIRCLIP_METADATA, "use": "caller_opt_out"}
    with pytest.raises(ValueError, match="research_fixed_metadata_review_required"):
        deployment.assemble(f.config, object())
