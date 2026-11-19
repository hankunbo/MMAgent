"""Safety vetoes on extension suggestions, using actual synthetic query evaluation."""

import pytest

from packages.grounding import CatalogSnapshot, Mapping, Ref, digest, validate
from packages.grounding.compatibility import tool_compatibility
from packages.grounding.query import evaluate_query
from packages.grounding.query_models import QueryResult
from packages.grounding.synthetic import SyntheticCatalogReader
from packages.grounding.validation import MappingPaths, PathAdmission
from tests.grounding.attribution_support import alternative_graph, producer_ready
from tests.grounding.query_support import metadata, paths, request_for, updated
from tests.grounding.test_models import reseal


def safety_case(environment="unknown", validation="auto_validated_research"):
    q, m, p, c, k, reader = producer_ready(validation=validation)
    producer = m.candidates[-1]
    q = updated(q, artifacts=[*q.artifacts, metadata(q, producer.inputs[0].constraint, "IMAGE001")])
    facts = [e for e in q.environment if e.implementation_ref != producer.implementation_ref]
    if environment != "unknown":
        fact = next(e for e in q.environment if e.implementation_ref == producer.implementation_ref)
        facts.append({**fact.model_dump(mode="json"), "availability": environment})
    return updated(q, environment=facts), m, p, c, k, reader


def run(data):
    q, m, p, _, k, reader = data
    return evaluate_query(q, paths(m, p, k, reader))


@pytest.mark.parametrize(
    "environment,validation,reason",
    [
        ("unknown", "auto_validated_research", "environment_unknown"),
        ("unavailable", "auto_validated_research", "environment_unavailable"),
        ("available", "smoke_tested", "tool_validation_rejected"),
    ],
)
def test_revocation_vetoes_coexisting_extensible_reason(environment, validation, reason):
    data = safety_case(environment, validation)
    q, m, _, _, _, reader = data
    reader.disable(m.candidates[-1].implementation_ref)
    result = run(data)
    assert {"tool_safety_revoked", reason, "required_input_missing"} <= {
        f.reason for f in result.gaps
    }
    assert result.status == "gaps" and result.clinical_effect == "NONE"
    assert result.product_demands and len(result.product_demands[0].affected_inputs) == 2
    assert not result.suggestions
    assert result.request.lock == q.lock


@pytest.mark.parametrize("revoked", [False, True])
def test_environment_only_and_revocation_only_controls(revoked):
    data = safety_case("available" if revoked else "unknown")
    if revoked:
        data[-1].disable(data[1].candidates[-1].implementation_ref)
    result = run(data)
    assert bool(result.suggestions) == (not revoked)
    if not revoked:
        assert "required_condition_unknown" in {f.reason for f in result.gaps}


@pytest.mark.parametrize("alternative_available", [False, True])
def test_same_capability_independent_implementation_is_not_vetoed(alternative_available):
    q, views = alternative_graph()
    producer = next(c for c in views[1].record.candidates if c.code.endswith("INTERMEDIATE"))
    reader = SyntheticCatalogReader([views[0].catalog])
    reader.disable(producer.implementation_ref)
    safety = reader.current_safety(q.lock.catalog_ref)
    current = tuple(
        validate(
            MappingPaths,
            {
                **v.model_dump(mode="json"),
                "safety": safety,
                "paths": [
                    PathAdmission(
                        candidate=p.candidate,
                        reasons=tool_compatibility(p.candidate, v.catalog, q.lock.policy, safety),
                    )
                    for p in v.paths
                ],
            },
        )
        for v in views
    )
    if not alternative_available:
        q = updated(q, environment=q.environment[:2])
    result = evaluate_query(q, current)
    assert producer.implementation_ref not in {
        s.target_implementation_ref for s in result.suggestions
    }
    assert any(f.reason == "tool_safety_revoked" for f in result.gaps)
    if alternative_available:
        assert len(result.candidates) == 2 and not result.product_demands and not result.suggestions
    else:
        assert result.status == "gaps" and len(result.suggestions) == 1
        assert result.suggestions[0].target_capability_ref == producer.capability_ref
        assert result.suggestions[0].target_implementation_ref != producer.implementation_ref


def test_revocation_of_another_exact_version_does_not_veto_current_target():
    data = safety_case()
    implementation = data[1].candidates[-1].implementation_ref
    other = validate(
        Ref,
        {
            **implementation.model_dump(mode="json"),
            "version": "9.0.0",
            "checksum": digest("future synthetic implementation"),
        },
    )
    data[-1].disable(other)
    result = run(data)
    assert {s.target_implementation_ref for s in result.suggestions} == {implementation}
    assert "tool_safety_revoked" not in {f.reason for f in result.gaps}


@pytest.mark.parametrize("boundary", ["json", "copy", "construct"])
def test_resealed_revoked_suggestion_fails_shared_result_validation(boundary):
    data = safety_case()
    old = run(data)
    assert old.suggestions
    data[-1].disable(data[1].candidates[-1].implementation_ref)
    current = run(data)
    fields = current.model_dump(mode="json")
    fields["suggestions"] = [s.model_dump(mode="json") for s in old.suggestions]
    fields.pop("checksum")
    fields["checksum"] = digest(fields)
    with pytest.raises(ValueError, match="attribution mismatch"):
        if boundary == "copy":
            validate(QueryResult, current.model_copy(update=fields))
        elif boundary == "construct":
            validate(QueryResult, QueryResult.model_construct(**fields))
        else:
            validate(QueryResult, fields)


@pytest.mark.parametrize(
    "rejection", ["tool_not_active", "intended_use_rejected", "precondition_rejected"]
)
def test_other_explicit_refusals_veto_same_source_environment_suggestion(rejection):
    q, m, p, catalog, k, _ = safety_case("unavailable")
    if rejection == "precondition_rejected":
        q = updated(
            q,
            environment=[
                {
                    **e.model_dump(mode="json"),
                    "preconditions": "rejected"
                    if e.implementation_ref == m.candidates[-1].implementation_ref
                    else e.preconditions,
                }
                for e in q.environment
            ],
        )
    else:
        fields = catalog.model_dump(mode="json")
        fields["tools"][-1]["enabled" if rejection == "tool_not_active" else "intended_use"] = (
            False if rejection == "tool_not_active" else "clinical_deployment"
        )
        catalog = reseal(CatalogSnapshot, fields)
        m = reseal(Mapping, {**m.model_dump(mode="json"), "catalog_ref": catalog.reference()})
        q = request_for(
            m, p, artifacts=q.artifacts, environment=q.environment, patient_data_types=[]
        )
    result = run((q, m, p, catalog, k, SyntheticCatalogReader([catalog])))
    assert {rejection, "environment_unavailable"} <= {f.reason for f in result.gaps}
    assert result.product_demands and not result.suggestions
