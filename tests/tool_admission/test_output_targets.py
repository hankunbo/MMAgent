"""Fully rebound controlled declarations: semantic failures, not stale seals."""

import pytest

from packages.grounding.models import digest, validate
from packages.grounding.synthetic import SyntheticCatalogReader, ref
from packages.tool_admission import AdmissionResult, AdmissionSelection
from tests.grounding.attribution_support import alternative_graph
from tests.grounding.query_support import updated
from tests.tool_admission.support import fixture, register_deployments, register_inputs, reseal
from tests.tool_admission.test_research import research

A = "CONCEPT:SYNTHETIC"
B = "CONCEPT:OTHER_ANATOMY"


def change_output(owners, invocation, port, targets):
    profile = owners.profiles[invocation.manifest_checksum]
    rules = [
        validate(type(p), {**p.model_dump(mode="json"), "anatomical_targets": targets})
        if p.port == port
        else p
        for p in profile.output_rules
    ]
    changed = reseal(profile, output_rules=rules)
    owners.profiles[invocation.manifest_checksum] = changed
    probe = reseal(owners.probes[profile.checksum], profile_checksum=changed.checksum)
    owners.probes[changed.checksum] = probe
    return validate(
        type(invocation),
        {
            **invocation.model_dump(mode="json"),
            "profile_checksum": changed.checksum,
        },
    )


def replace_invocation(selection, invocation):
    return validate(
        AdmissionSelection,
        {
            **selection.model_dump(mode="json"),
            "invocations": [
                invocation if i.implementation_ref == invocation.implementation_ref else i
                for i in selection.invocations
            ],
        },
    )


def assert_output_rejection(result, implementation):
    assert not any(c.candidate.implementation_ref == implementation for c in result.retained)
    failures = [r for r in result.rejected if r.implementation_ref == implementation]
    assert any(r.reason == "output_anatomical_target_mismatch" for r in failures)
    assert not result.capability_gaps
    assert result.clinical_effect == "NONE" and not result.execution_authorized


@pytest.mark.parametrize("both_in_request", [False, True])
def test_root_rebound_output_or_and_request_targets(tmp_path, both_in_request):
    f = fixture(tmp_path)
    if both_in_request:
        f.q = updated(f.q, satisfied_terms=[*f.q.satisfied_terms, B])
        f.result = f.grounding.query(f.q)
    good = f.service.admit(f.q, f.result, f.selection)
    assert len(good.retained_roots) == 2
    c = good.retained[0].candidate
    assert B not in c.output.scope_terms and A in c.output.scope_terms
    inv = next(i for i in f.selection.invocations if i.implementation_ref == c.implementation_ref)
    inv = change_output(f.owners, inv, c.output_port, [B])
    selection = replace_invocation(f.selection, inv)
    result = f.service.admit(f.q, f.result, selection)
    assert_output_rejection(result, c.implementation_ref)
    assert len(result.retained_roots) == len(result.retained) == 1
    assert f.service.validate_result(f.q, f.result, selection, result) == result
    restored = change_output(f.owners, inv, c.output_port, [A])
    assert f.service.admit(f.q, f.result, replace_invocation(selection, restored)) == good


@pytest.mark.parametrize("purpose", ["debug", "shadow"])
def test_registered_research_rebound_output(tmp_path, purpose):
    f = fixture(tmp_path)
    request = research(f, tmp_path, purpose=purpose)
    assert f.service.admit_research(request).status == "preparation_only"
    invocation = change_output(f.owners, request.invocation, request.candidate.output_port, [B])
    request = reseal(request, invocation=invocation)
    f.owners.requests[(request.code, request.version, request.checksum)] = request
    result = f.service.admit_research(request)
    assert_output_rejection(result, request.candidate.implementation_ref)
    assert not result.retained_roots and result.status == "rejected"
    assert f.service.validate_research_result(request, result) == result


def alternative_fixture(tmp_path):
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
    f.q = register_inputs(q, f.owners, f.objects)
    f.catalog = views[0].catalog
    catalogs = SyntheticCatalogReader([f.catalog])
    f.locks.q = f.q
    f.locks.query_paths = lambda *a: views
    f.grounding.mappings.catalogs = catalogs
    f.service.catalogs = catalogs
    invocations = register_deployments(f.q, f.catalog, f.owners, tmp_path)
    f.result = f.grounding.query(f.q)
    f.selection = AdmissionSelection(
        candidate_refs=tuple(c.identity for c in f.result.candidates),
        invocations=invocations,
    )
    return f, producer


@pytest.mark.parametrize("alternative", [False, True])
def test_nested_producer_rebound_output_no_orphan(tmp_path, alternative):
    if alternative:
        f, producer = alternative_fixture(tmp_path)
    else:
        f = fixture(tmp_path, chain=True)
        producer = f.mapping.candidates[-1]
    before = f.service.admit(f.q, f.result, f.selection)
    assert before.retained_roots
    inv = next(
        i for i in f.selection.invocations if i.implementation_ref == producer.implementation_ref
    )
    inv = change_output(f.owners, inv, producer.output_port, [B])
    selection = replace_invocation(f.selection, inv)
    result = f.service.admit(f.q, f.result, selection)
    assert_output_rejection(result, producer.implementation_ref)
    if alternative:
        assert len(result.retained_roots) == 2 and len(result.retained) == 3
    else:
        assert not result.retained_roots and not result.retained
    reachable = set(result.retained_roots)
    nodes = {c.identity: c for c in result.retained}
    for c in result.retained:
        for port in c.inputs:
            assert set(port.producer_refs) <= nodes.keys()
            reachable.update(port.producer_refs)
    assert reachable == nodes.keys()
    assert f.service.validate_result(f.q, f.result, selection, result) == result


def legacy_wrong_result(f):
    """Reconstruct the exact old bug output, with every changed digest correctly bound."""
    old = f.service.admit(f.q, f.result, f.selection)
    c = old.retained[0]
    inv = next(
        i for i in f.selection.invocations if i.implementation_ref == c.candidate.implementation_ref
    )
    inv = change_output(f.owners, inv, c.candidate.output_port, [B])
    selection = replace_invocation(f.selection, inv)
    normalized = validate(
        AdmissionSelection,
        dict(
            candidate_refs=sorted(selection.candidate_refs),
            invocations=sorted(selection.invocations, key=lambda i: digest(i.implementation_ref)),
        ),
    )
    profile = f.owners.profiles[inv.manifest_checksum]
    data = old.model_dump(mode="json")
    data["selection_checksum"] = digest(normalized)
    target = next(n for n in data["retained"] if n["identity"] == c.identity)
    target.update(
        profile_checksum=profile.checksum,
        probe_checksum=f.owners.probes[profile.checksum].checksum,
        invocation_checksum=digest(inv),
        output_rule=next(
            p.model_dump(mode="json")
            for p in profile.output_rules
            if p.port == c.candidate.output_port
        ),
    )
    data.pop("checksum")
    data["checksum"] = digest(data)
    return selection, data


@pytest.mark.parametrize("boundary", ["json", "copy", "construct"])
def test_old_semantically_wrong_result_revalidation(tmp_path, boundary):
    f = fixture(tmp_path)
    selection, data = legacy_wrong_result(f)
    # Structural construction is not contextual admission authority.
    parsed = validate(AdmissionResult, data)
    forged = (
        data
        if boundary == "json"
        else (
            parsed.model_copy(update=data)
            if boundary == "copy"
            else AdmissionResult.model_construct(**data)
        )
    )
    with pytest.raises(ValueError, match="stale_or_forged_admission_result"):
        f.service.validate_result(f.q, f.result, selection, forged)


def test_output_rejection_precedes_probe_resources_and_input_bytes(tmp_path, monkeypatch):
    import packages.tool_admission.service as module

    f = fixture(tmp_path)
    c = f.result.candidates[0].candidate
    inv = next(i for i in f.selection.invocations if i.implementation_ref == c.implementation_ref)
    inv = change_output(f.owners, inv, c.output_port, [B])
    selection = AdmissionSelection(
        candidate_refs=(f.result.candidates[0].identity,), invocations=(inv,)
    )

    def forbidden(*args, **kwargs):
        pytest.fail("output contradiction reached probe, resources or input read")

    for owner, name in (
        (f.owners, "read_probe"),
        (f.owners, "read_input"),
        (f.objects, "read_bounded"),
        (module, "observe_resource"),
        (module, "observe_manifest_resources"),
    ):
        monkeypatch.setattr(owner, name, forbidden)
    result = f.service.admit(f.q, f.result, selection)
    assert_output_rejection(result, c.implementation_ref)
    assert all(
        r.category == "output_incompatible" and r.port == c.output_port and r.constraint == c.output
        for r in result.rejected
    )


@pytest.mark.parametrize("mutation", ["absent", "other_port", "other_type", "other_version"])
def test_unknown_output_binding_is_not_a_tool_or_environment_gap(tmp_path, mutation):
    f = fixture(tmp_path)
    inv = f.selection.invocations[0]
    profile = f.owners.profiles[inv.manifest_checksum]
    binding = profile.output_target_bindings[0]
    if mutation == "absent":
        bindings = []
    else:
        if mutation == "other_port":
            binding = reseal(
                binding, port={**binding.port.model_dump(mode="json"), "port_name": "other"}
            )
        else:
            constraint = binding.constraint.model_dump(mode="json")
            if mutation == "other_type":
                constraint["artifact_ref"]["code"] = "ARTIFACT:SYNTHETIC:OTHER"
            else:
                constraint["artifact_ref"]["version"] = "99.0.0"
            binding = reseal(binding, constraint=constraint)
        bindings = [binding]
    changed = reseal(profile, output_target_bindings=bindings)
    f.owners.profiles[inv.manifest_checksum] = changed
    f.owners.probes[changed.checksum] = reseal(
        f.owners.probes[profile.checksum], profile_checksum=changed.checksum
    )
    inv = validate(type(inv), {**inv.model_dump(mode="json"), "profile_checksum": changed.checksum})
    selection = replace_invocation(f.selection, inv)
    result = f.service.admit(f.q, f.result, selection)
    assert len(result.retained_roots) == 1 and not result.capability_gaps
    assert any(
        r.reason == "output_anatomical_target_unknown" and r.category == "output_incompatible"
        for r in result.rejected
    )
    assert f.service.validate_result(f.q, f.result, selection, result) == result


def test_typed_subset_does_not_treat_every_scope_term_as_anatomy(tmp_path):
    from packages.tool_admission.models import DeploymentProfile

    f = fixture(tmp_path)
    c = f.result.candidates[0].candidate
    inv = next(i for i in f.selection.invocations if i.implementation_ref == c.implementation_ref)
    profile = f.owners.profiles[inv.manifest_checksum]
    context = "CONCEPT:SYNTHETIC_APPLICABILITY"
    constraint = validate(
        type(c.output), {**c.output.model_dump(mode="json"), "scope_terms": [A, context]}
    )
    c = validate(type(c), {**c.model_dump(mode="json"), "output": constraint})
    binding = reseal(profile.output_target_bindings[0], constraint=constraint)
    profile = reseal(profile, output_target_bindings=[binding])
    assert profile.check_output_targets(c, (A, context)).anatomical_targets == (A,)
    rules = [p.model_copy(update={"anatomical_targets": (context,)}) for p in profile.output_rules]
    changed = reseal(profile, output_rules=rules)
    # Normal construction, JSON and copy/construct reparse all use the same contextual check.
    for value in (
        changed,
        changed.model_dump(mode="json"),
        changed.model_copy(),
        DeploymentProfile.model_construct(**changed.model_dump()),
    ):
        checked = validate(DeploymentProfile, value)
        with pytest.raises(ValueError, match="output_anatomical_target_mismatch"):
            checked.check_output_targets(c, (A, context))


def test_binding_cannot_claim_anatomy_absent_from_exact_constraint(tmp_path):
    f = fixture(tmp_path)
    binding = next(iter(f.owners.profiles.values())).output_target_bindings[0]
    with pytest.raises(ValueError, match="output_target_binding_outside_constraint"):
        reseal(binding, anatomical_targets=[B])
