import pytest

from packages.grounding.models import Candidate, CatalogSnapshot, digest, seal, validate
from packages.grounding.synthetic import SyntheticCatalogReader
from packages.tool_admission.models import ResearchRequest
from tests.grounding.query_support import NOW
from tests.tool_admission.support import fixture, register_deployments, reseal


def research(f, tmp_path, *, status="active", validation="smoke_tested", purpose="shadow"):
    tools = [t.model_dump(mode="json") for t in f.catalog.tools]
    for t in tools:
        t["catalog_status"], t["validation_status"] = status, validation
    catalog = reseal(f.catalog, version="9.0.0", tools=tools)
    f.service.catalogs = SyntheticCatalogReader([catalog])
    invocations = register_deployments(f.q, catalog, f.owners, tmp_path)
    c = validate(
        Candidate,
        {**f.mapping.candidates[0].model_dump(mode="json"), "validation_required": validation},
    )
    request = seal(
        ResearchRequest,
        code="RESEARCH:SYNTHETIC:DEBUG",
        version="1.0.0",
        purpose=purpose,
        compatibility_policy=reseal(
            f.policy,
            code="POLICY:TOOL:RESEARCH_COMPATIBILITY",
            accepted_validation=["smoke_tested", "benchmark_passed", "auto_validated_research"],
        ),
        lock=dict(run_id=f.q.lock.run_id, catalog_ref=catalog.reference()),
        scope=f.q.scope,
        window_start=f.q.window_start,
        window_end=f.q.window_end,
        snapshot_at=NOW,
        satisfied_terms=f.q.satisfied_terms,
        artifacts=f.q.artifacts,
        candidate=c,
        inputs=[
            dict(
                port=i.port,
                asset_codes=[
                    a.asset_code
                    for a in f.q.artifacts
                    if a.artifact_ref == i.constraint.artifact_ref
                ],
            )
            for i in c.inputs
        ],
        invocation=invocations[0],
    )
    f.owners.requests[(request.code, request.version, request.checksum)] = request
    return request


@pytest.mark.parametrize("purpose", ["debug", "shadow"])
def test_explicit_registered_research_not_decisive(tmp_path, purpose):
    f = fixture(tmp_path)
    request = research(f, tmp_path, purpose=purpose)
    result = f.service.admit_research(request)
    assert result.status == "preparation_only" and result.purpose == purpose
    assert result.query_checksum is None and not result.capability_gaps
    assert not result.decisive_evidence and not result.execution_authorized
    assert f.service.validate_research_result(request, result) == result
    with pytest.raises(ValueError):
        f.service.validate_result(f.q, f.result, f.selection, result)


@pytest.mark.parametrize(
    "status,validation",
    [
        ("failed", "smoke_tested"),
        ("quarantined", "smoke_tested"),
        ("deprecated", "smoke_tested"),
        ("registered", "unvalidated"),
        ("active", "unvalidated"),
    ],
)
def test_shadow_cannot_allow_failed_or_unvalidated(tmp_path, status, validation):
    f = fixture(tmp_path)
    request = research(f, tmp_path, status=status, validation=validation)
    result = f.service.admit_research(request)
    assert result.status == "rejected" and not result.retained
    assert "contract_use_rejected" in {r.reason for r in result.rejected}


@pytest.mark.parametrize(
    "change", ["claim", "override", "purpose", "patient", "input", "candidate"]
)
def test_registered_research_rejects_caller_forgery(tmp_path, change):
    f = fixture(tmp_path)
    request = research(f, tmp_path)
    data = request.model_dump(mode="json")
    if change == "claim":
        data["clinical_claim"] = {"state": "Draft"}
    elif change == "override":
        data["allow_failed"] = True
    elif change == "purpose":
        data["purpose"] = "debug"
    elif change == "patient":
        data["scope"]["patient_hash"] = digest("other")
    elif change == "input":
        data["inputs"][0]["asset_codes"] = []
    else:
        data["candidate"]["tool_ref"]["version"] = "2.0.0"
    data.pop("checksum")
    data["checksum"] = digest(data)
    with pytest.raises((ValueError, KeyError)):
        f.service.admit_research(request.model_copy(update=data))


def test_research_current_revocation_and_old_catalog(tmp_path):
    f = fixture(tmp_path)
    request = research(f, tmp_path)
    old = f.service.admit_research(request)
    f.service.catalogs.disable(request.candidate.implementation_ref)
    assert f.service.admit_research(request).status == "rejected"
    with pytest.raises(ValueError, match="stale_or_forged"):
        f.service.validate_research_result(request, old)
    newer = reseal(f.service.catalogs.read_snapshot(request.lock.catalog_ref), version="10.0.0")
    f.service.catalogs.add(validate(CatalogSnapshot, newer))
    assert f.service.admit_research(request).status == "rejected"


def test_real_06a_projection_shape_stays_unadmitted(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from packages.integrations.medgraph.models import DeploymentManifest, SnapshotEnvelope

    f = fixture(tmp_path)
    root = Path(__file__).resolve().parents[2]
    envelope = validate(
        SnapshotEnvelope,
        json.loads((root / "docs/acceptance/06a/examples/snapshot.synthetic.json").read_text()),
    )
    manifest = validate(
        DeploymentManifest,
        json.loads((root / "docs/acceptance/06a/examples/manifest.synthetic.json").read_text()),
    )
    tool = next(
        t for t in envelope.catalog.tools if t.implementation_ref == manifest.implementation_ref
    )
    request = research(f, tmp_path)
    candidate = validate(
        Candidate,
        dict(
            code="CAPABILITY:SYNTHETIC:LEGACY",
            capability_ref=tool.capabilities[0],
            tool_ref=tool.tool_ref,
            implementation_ref=tool.implementation_ref,
            workflow_ref=None,
            output_port=tool.output_ports[0],
            output=tool.outputs[0],
            inputs=tool.inputs,
            validation_required=tool.validation_status,
            intended_use=tool.intended_use,
            execution_policy="research",
            estimate_ref=None,
        ),
    )
    request = reseal(
        request,
        candidate=candidate,
        lock=dict(run_id=request.lock.run_id, catalog_ref=envelope.catalog.reference()),
        inputs=[dict(port=i.port, asset_codes=[]) for i in tool.inputs],
        invocation=dict(
            implementation_ref=tool.implementation_ref,
            manifest_checksum=manifest.checksum,
            profile_checksum=digest("no profile"),
            parameters={},
        ),
    )
    f.owners.requests[(request.code, request.version, request.checksum)] = request
    f.service.catalogs = SyntheticCatalogReader([envelope.catalog])
    f.owners.manifests[(manifest.catalog_ref, manifest.tool_ref, manifest.implementation_ref)] = (
        manifest
    )
    monkeypatch.setattr(
        f.objects, "read_bounded", lambda *a: pytest.fail("unadmitted tool read input")
    )
    result = f.service.admit_research(request)
    assert result.status == "rejected" and not result.retained
    assert {"tool_not_active", "required_condition_unknown", "contract_use_rejected"} <= {
        r.reason for r in result.rejected
    }
    assert manifest.binding.gaps and manifest.execution_authorized is False
    assert result.deployment_gaps[0].gaps == manifest.binding.gaps
    assert result.deployment_gaps[0].manifest_checksum == manifest.checksum


def test_synthetic_only_policy_cannot_admit_real_catalog(tmp_path):
    from packages.grounding.ports import ToolSafety

    f = fixture(tmp_path)
    request = research(f, tmp_path)
    catalog = reseal(
        f.service.catalogs.read_snapshot(request.lock.catalog_ref), version="9.1.0", synthetic=False
    )
    invocations = register_deployments(f.q, catalog, f.owners, tmp_path)

    class ControlledCatalog:
        def read_snapshot(self, reference):
            assert reference == catalog.reference()
            return catalog

        def current_safety(self, reference):
            self.read_snapshot(reference)
            return ToolSafety(catalog_ref=reference, disabled_implementations=(), revision=0)

    f.service.catalogs = ControlledCatalog()
    request = reseal(
        request,
        lock=dict(run_id=request.lock.run_id, catalog_ref=catalog.reference()),
        invocation=invocations[0],
    )
    f.owners.requests[(request.code, request.version, request.checksum)] = request
    result = f.service.admit_research(request)
    assert result.status == "rejected" and not result.retained
    assert any(
        r.reason == "synthetic_policy_on_real_catalog" and r.category == "policy_rejection"
        for r in result.rejected
    )
    # Explicitly service-registered non-synthetic policy is a distinct request.
    real_policy = reseal(request.compatibility_policy, synthetic_only=False)
    request = reseal(request, compatibility_policy=real_policy)
    f.owners.requests[(request.code, request.version, request.checksum)] = request
    assert f.service.admit_research(request).status == "preparation_only"
