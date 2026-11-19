import pytest

from packages.grounding.models import digest, validate
from packages.tool_admission import AdmissionResult
from tests.tool_admission.support import fixture


@pytest.mark.parametrize("chain", [False, True])
def test_formal_single_and_multistep_no_execution(tmp_path, monkeypatch, chain):
    f = fixture(tmp_path, chain=chain)
    import subprocess

    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **k: pytest.fail("admission started process")
    )
    before_files = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = f.service.admit(f.q, f.result, f.selection)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before_files
    assert result.status == "preparation_only"
    assert len(result.retained_roots) == 2
    assert len(result.retained) == (3 if chain else 2)
    assert all(len(c.inputs) == len(c.candidate.inputs) for c in result.retained)
    assert all(c.pending_dynamic_checks for c in result.retained)
    assert not result.execution_authorized and not result.decisive_evidence
    assert f.service.validate_result(f.q, f.result, f.selection, result) == result
    assert f.locks.read(f.q.lock.run_id) == f.q.lock


@pytest.mark.parametrize("mutation", ["purpose", "scope", "status", "input", "extra"])
@pytest.mark.parametrize("boundary", ["copy", "construct", "json"])
def test_forged_results_cannot_be_resealed(tmp_path, mutation, boundary):
    f = fixture(tmp_path)
    result = f.service.admit(f.q, f.result, f.selection)
    data = result.model_dump(mode="json")
    if mutation == "purpose":
        data["purpose"], data["query_checksum"] = "shadow", None
    elif mutation == "scope":
        data["scope"]["patient_hash"] = digest("other patient")
    elif mutation == "status":
        data["execution_authorized"] = True
    elif mutation == "input":
        data["retained"][0]["inputs"][0]["artifacts"][0]["checksum"] = digest("wrong bytes")
    else:
        data["approved"] = True
    data.pop("checksum")
    data["checksum"] = digest(data)
    forged = (
        data
        if boundary == "json"
        else (
            result.model_copy(update=data)
            if boundary == "copy"
            else AdmissionResult.model_construct(**data)
        )
    )
    with pytest.raises(ValueError):
        f.service.validate_result(f.q, f.result, f.selection, forged)


def test_exact_candidate_membership_and_old_safety(tmp_path):
    f = fixture(tmp_path)
    wrong = f.selection.model_copy(update={"candidate_refs": (digest("invented"),)})
    with pytest.raises(ValueError, match="candidate_outside"):
        f.service.admit(f.q, f.result, wrong)
    f.catalogs.disable(f.catalog.tools[0].implementation_ref)
    with pytest.raises(ValueError, match="stale_or_forged_query_result"):
        f.service.admit(f.q, f.result, f.selection)


@pytest.mark.parametrize(
    "parameters", [{"threshold": 2}, {"threshold": "0.3"}, {"threshold": True}, {"unexpected": 1}]
)
def test_illegal_parameters_cannot_revive_by_rank(tmp_path, parameters):
    f = fixture(tmp_path)
    first, second = f.selection.invocations
    first = validate(type(first), {**first.model_dump(mode="json"), "parameters": parameters})
    selection = f.selection.model_copy(update={"invocations": (first, second)})
    result = f.service.admit(f.q, f.result, selection)
    assert len(result.retained_roots) == 1
    assert result.retained[0].candidate.implementation_ref == second.implementation_ref
    assert any(r.reason == "parameter_invalid" for r in result.rejected)


@pytest.mark.parametrize(
    "change,reason",
    [
        ("missing", "patient_data_missing"),
        ("checksum", "checksum_mismatch"),
        ("size", "size_mismatch"),
        ("dimensions", "dimensions_mismatch"),
        ("format", "format_mismatch"),
        ("note", "input_content_forbidden"),
        ("uri", "input_uri_rejected"),
        ("patient", "input_record_mismatch"),
        ("time", "input_record_mismatch"),
        ("qc", "input_record_mismatch"),
        ("laterality", "input_record_mismatch"),
        ("unit", "input_record_mismatch"),
        ("coordinate", "input_record_mismatch"),
        ("modality", "input_record_mismatch"),
    ],
)
def test_actual_input_failures_preserve_and(tmp_path, change, reason):
    f = fixture(tmp_path)
    code = f.q.artifacts[0].asset_code
    record = f.owners.records[code]
    data = record.model_dump(mode="json")
    if change == "missing":
        del f.owners.records[code]
    elif change == "checksum":
        f.objects._path(record.artifact.uri).write_bytes(b"x" * record.byte_size)
    else:
        if change == "size":
            data["byte_size"] += 1
        elif change == "dimensions":
            data["artifact"]["dimensions"] = [3, 2]
        elif change == "format":
            data["artifact"]["media_type"] = "image/jpeg"
        elif change == "note":
            data["content_kind"] = "clinical_note"
        elif change == "uri":
            data["artifact"]["uri"] = "object://knowledge/objects/not-controlled"
        elif change == "patient":
            data["metadata"]["patient_hash"] = digest("other")
        elif change == "time":
            data["metadata"]["acquired_at"] = None
        elif change == "qc":
            data["metadata"]["qc_status"] = data["artifact"]["qc_status"] = "FAIL"
        elif change == "laterality":
            data["metadata"]["laterality"] = data["artifact"]["laterality"] = "LEFT"
        elif change == "unit":
            data["metadata"]["unit"] = {"state": "known", "code": "CONCEPT:UNIT:OTHER"}
            data["artifact"]["unit"] = "mm"
            from tests.tool_admission.support import reseal

            data["type_binding"] = reseal(
                record.type_binding, unit_term=data["metadata"]["unit"], contract_unit="mm"
            )
        elif change == "coordinate":
            data["metadata"]["coordinate_space"] = data["artifact"]["coordinate_space"] = "PIXEL_2D"
        elif change == "modality":
            data["metadata"]["modality"] = {"state": "known", "code": "CONCEPT:MODALITY:OTHER"}
        f.owners.records[code] = validate(type(record), data)
    result = f.service.admit(f.q, f.result, f.selection)
    assert result.status == "rejected"
    assert reason in {r.reason for r in result.rejected}
    assert "required_input_missing" in {r.reason for r in result.rejected}
    assert not result.capability_gaps  # Missing patient input is not missing tool.


@pytest.mark.parametrize(
    "field,state,reason",
    [
        ("interpreter", "unknown", "environment_unknown"),
        ("dependencies", "unknown", "environment_unknown"),
        ("gpu", "unknown", "environment_unknown"),
        ("resources", "unavailable", "environment_unavailable"),
        ("preconditions", "rejected", "precondition_rejected"),
        ("profile_checksum", digest("other"), "probe_binding_mismatch"),
    ],
)
def test_exact_environment_and_or_isolation(tmp_path, field, state, reason):
    from tests.tool_admission.support import reseal

    f = fixture(tmp_path)
    invocation = f.selection.invocations[0]
    probe = f.owners.probes[invocation.profile_checksum]
    f.owners.probes[invocation.profile_checksum] = reseal(probe, **{field: state})
    result = f.service.admit(f.q, f.result, f.selection)
    assert len(result.retained_roots) == 1
    assert reason in {r.reason for r in result.rejected}
    if field != "preconditions":
        assert len(result.capability_gaps) == 1
        assert result.capability_gaps[0].capability_code == f.catalog.tools[0].capabilities[0].code


def test_missing_producer_is_not_consumer_gap(tmp_path):
    from tests.tool_admission.support import reseal

    f = fixture(tmp_path, chain=True)
    invocation = f.selection.invocations[-1]
    probe = f.owners.probes[invocation.profile_checksum]
    f.owners.probes[invocation.profile_checksum] = reseal(probe, interpreter="unknown")
    result = f.service.admit(f.q, f.result, f.selection)
    assert result.status == "rejected"
    assert len(result.capability_gaps) == 1
    failures = [r for r in result.rejected if r.port is None]
    assert all(r.implementation_ref == invocation.implementation_ref for r in failures)
    assert result.capability_gaps[0].capability_code == failures[0].capability_ref.code


def test_probe_expired_and_resource_drift(tmp_path):
    from datetime import timedelta

    from tests.grounding.query_support import NOW

    f = fixture(tmp_path)
    f.service.clock = lambda: NOW + timedelta(days=2)
    result = f.service.admit(f.q, f.result, f.selection)
    assert result.status == "rejected" and all(r.reason == "probe_stale" for r in result.rejected)
    f.service.clock = lambda: NOW
    (tmp_path / "synthetic-interpreter.bin").write_bytes(b"drift")
    result = f.service.admit(f.q, f.result, f.selection)
    assert result.status == "rejected"
    assert all(r.reason == "interpreter_checksum_mismatch" for r in result.rejected)


def test_bounded_truncation_cannot_be_misread_as_no_tool(tmp_path):
    from packages.tool_admission import AdmissionPolicy

    f = fixture(tmp_path)
    f.service.policy = AdmissionPolicy(max_checks=1)
    result = f.service.admit(f.q, f.result, f.selection)
    assert result.status == "truncated" and not result.retained
    assert not result.capability_gaps and result.clinical_effect == "NONE"


def test_current_safety_read_failure_is_technical_error(tmp_path, monkeypatch):
    f = fixture(tmp_path)

    def fail(*a):
        raise TimeoutError("synthetic backend timeout")

    monkeypatch.setattr(f.catalogs, "current_safety", fail)
    with pytest.raises(TimeoutError):
        f.service.admit(f.q, f.result, f.selection)


def test_note_never_reads_body(tmp_path, monkeypatch):
    f = fixture(tmp_path)
    for code, record in f.owners.records.items():
        f.owners.records[code] = record.model_copy(update={"content_kind": "clinical_note"})
    monkeypatch.setattr(f.objects, "read_bounded", lambda *a: pytest.fail("read note body"))
    assert f.service.admit(f.q, f.result, f.selection).status == "rejected"


@pytest.mark.parametrize("field", ["manifest_checksum", "profile_checksum", "implementation_ref"])
def test_stale_or_cross_version_invocation(tmp_path, field):
    f = fixture(tmp_path)
    i, second = f.selection.invocations
    value = (
        digest("forged")
        if field != "implementation_ref"
        else i.implementation_ref.model_copy(update={"version": "99.0.0"})
    )
    changed = validate(type(i), {**i.model_dump(mode="json"), field: value})
    selection = f.selection.model_copy(update={"invocations": (changed, second)})
    result = f.service.admit(f.q, f.result, selection)
    assert len(result.retained_roots) == 1
    assert not any(c.candidate.implementation_ref == i.implementation_ref for c in result.retained)


@pytest.mark.parametrize(
    "mutation", ["ports", "preprocessing", "postprocessing", "schema", "binding"]
)
def test_modified_controlled_manifest_requires_new_binding(tmp_path, mutation):
    from tests.tool_admission.support import reseal

    f = fixture(tmp_path)
    key, manifest = next(iter(f.owners.manifests.items()))
    if mutation == "ports":
        changes = {"ports": manifest.ports[1:]}
    elif mutation in {"preprocessing", "postprocessing"}:
        changes = {mutation + "_policy": {"version": "99.0.0", "policy_code": "POLICY:UNKNOWN"}}
    elif mutation == "schema":
        changes = {"parameter_schema": {"type": "object", "additionalProperties": True}}
    else:
        changes = {"config_checksum": digest("other deployment")}
    f.owners.manifests[key] = reseal(manifest, **changes)
    result = f.service.admit(f.q, f.result, f.selection)
    assert len(result.retained_roots) == 1
    assert "manifest_binding_mismatch" in {r.reason for r in result.rejected}


def test_upstream_no_candidate_retains_layered_gaps(tmp_path):
    from packages.tool_admission import AdmissionSelection
    from tests.grounding.query_support import updated

    f = fixture(tmp_path)
    q = updated(f.q, artifacts=[])
    query = f.grounding.query(q)
    assert not query.candidates and query.product_demands
    result = f.service.admit(q, query, AdmissionSelection(candidate_refs=(), invocations=()))
    assert (
        result.product_demands == query.product_demands and result.upstream_findings == query.gaps
    )
    assert not result.capability_gaps


def test_optional_bad_input_does_not_block_required_inputs(tmp_path):
    from packages.tool_admission import AdmissionSelection
    from tests.grounding.query_support import metadata, updated
    from tests.tool_admission.support import register_inputs

    f = fixture(tmp_path)
    optional = next(i for i in f.mapping.candidates[0].inputs if not i.required)
    q = updated(f.q, artifacts=[*f.q.artifacts, metadata(f.q, optional.constraint, "OPTIONAL")])
    q = register_inputs(q, f.owners, f.objects)
    result = f.grounding.query(q)
    code = next(a.asset_code for a in q.artifacts if a.asset_code.endswith("OPTIONAL"))
    del f.owners.records[code]
    selection = AdmissionSelection(
        candidate_refs=tuple(c.identity for c in result.candidates),
        invocations=f.selection.invocations,
    )
    admitted = f.service.admit(q, result, selection)
    assert len(admitted.retained_roots) == 2
    assert all(
        next(p for p in c.inputs if not p.required).optional_missing for c in admitted.retained
    )


def test_safety_change_during_file_read_fails_closed(tmp_path, monkeypatch):
    f = fixture(tmp_path)
    read = f.objects.read_bounded
    changed = False

    def revoke(uri, bound):
        nonlocal changed
        if not changed:
            f.catalogs.disable(f.catalog.tools[0].implementation_ref)
            changed = True
        return read(uri, bound)

    monkeypatch.setattr(f.objects, "read_bounded", revoke)
    with pytest.raises(ValueError, match="safety_changed"):
        f.service.admit(f.q, f.result, f.selection)


def test_independent_or_producer_failure_is_diagnostic_not_capability_gap(tmp_path):
    from packages.grounding.synthetic import SyntheticCatalogReader, ref
    from packages.tool_admission import AdmissionSelection
    from tests.grounding.attribution_support import alternative_graph
    from tests.grounding.query_support import updated
    from tests.tool_admission.support import register_deployments, register_inputs, reseal

    f = fixture(tmp_path)
    q, views = alternative_graph()
    producer = views[1].record.candidates[0]
    q = updated(
        q,
        environment=[
            *q.environment,
            dict(
                implementation_ref=producer.implementation_ref,
                availability="available",
                preconditions="satisfied",
                source_ref=ref("source_version", "SOURCE:SYNTHETIC:ENV"),
            ),
        ],
    )
    q = register_inputs(q, f.owners, f.objects)
    catalog = views[0].catalog
    catalogs = SyntheticCatalogReader([catalog])
    f.locks.q = q
    f.locks.query_paths = lambda *a: views
    f.grounding.mappings.catalogs = catalogs
    f.service.catalogs = catalogs
    invocations = register_deployments(q, catalog, f.owners, tmp_path)
    result = f.grounding.query(q)
    assert len(result.candidates) == 2
    inv = next(i for i in invocations if i.implementation_ref == producer.implementation_ref)
    probe = f.owners.probes[inv.profile_checksum]
    f.owners.probes[inv.profile_checksum] = reseal(probe, dependencies="unknown")
    selection = AdmissionSelection(
        candidate_refs=tuple(c.identity for c in result.candidates), invocations=invocations
    )
    admitted = f.service.admit(q, result, selection)
    assert len(admitted.retained_roots) == 2 and len(admitted.retained) == 3
    assert any(r.reason == "environment_unknown" for r in admitted.rejected)
    assert not admitted.capability_gaps
    assert not any(
        c.candidate.implementation_ref == producer.implementation_ref for c in admitted.retained
    )


def test_permutation_of_selection_does_not_rerank_or_change_admission(tmp_path):
    f = fixture(tmp_path)
    old = f.service.admit(f.q, f.result, f.selection)
    selection = f.selection.model_copy(
        update={
            "candidate_refs": f.selection.candidate_refs[::-1],
            "invocations": f.selection.invocations[::-1],
        }
    )
    assert f.service.admit(f.q, f.result, selection) == old


def test_formally_rejected_candidate_preserves_06a_binding_gaps(tmp_path):
    from packages.tool_admission import AdmissionSelection
    from tests.tool_admission.support import reseal

    f = fixture(tmp_path)
    key, manifest = next(iter(f.owners.manifests.items()))
    binding = manifest.binding.model_copy(
        update={"gaps": ("native_scope_missing", "validation_gates_missing")}
    )
    manifest = reseal(manifest, binding=binding)
    f.owners.manifests[key] = manifest
    f.catalogs.disable(manifest.implementation_ref)
    query = f.grounding.query(f.q)
    selection = AdmissionSelection(
        candidate_refs=tuple(c.identity for c in query.candidates),
        invocations=f.selection.invocations[1:],
    )
    result = f.service.admit(f.q, query, selection)
    assert len(result.retained_roots) == 1
    assert len(result.deployment_gaps) == 1
    assert result.deployment_gaps[0].origin == "official_rejection"
    assert result.deployment_gaps[0].gaps == binding.gaps
    assert result.deployment_gaps[0].manifest_checksum == manifest.checksum
    assert all(
        c.candidate.implementation_ref != manifest.implementation_ref for c in result.retained
    )
