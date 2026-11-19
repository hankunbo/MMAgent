"""05B deterministic query and adversarial boundary examples; synthetic only."""

from datetime import timedelta

import pytest

from packages.grounding import Mapping, digest, validate
from packages.grounding.compatibility import artifact_compatibility, tool_compatibility
from packages.grounding.query import evaluate_query, normalized_request
from packages.grounding.query_models import QueryConfig, QueryResult
from packages.grounding.synthetic import SyntheticCatalogReader, fixture
from packages.retrieval.knowledge.models import RetrievalContext
from tests.grounding.query_support import metadata, paths, ready, updated
from tests.grounding.test_models import reseal


def test_revoked_or_path_does_not_hide_legal_alternative():
    from packages.grounding.validation import evaluate_paths

    mapping, policy, catalog, knowledge, _ = fixture()
    reader = SyntheticCatalogReader([catalog])
    reader.disable(mapping.candidates[0].implementation_ref)
    result = evaluate_paths(mapping, policy, knowledge, reader, RetrievalContext())
    assert result.paths[0].reasons == ("tool_safety_revoked",)
    assert result.paths[1].reasons == ()
    assert result.safety.catalog_ref == catalog.reference()


def test_ready_and_existing_artifact_priority_unknown_scores():
    q, m, p, c, k, reader = ready()
    result = evaluate_query(q, paths(m, p, k, reader))
    assert result.status == "candidates" and len(result.candidates) == 2
    assert len(result.top_k) == 1 and result.enumeration == "complete"
    assert result.candidates[0].candidate.code.endswith("PATH0")
    assert all(
        c.state == "unknown" and c.value is None for x in result.candidates for c in x.components
    )
    assert result.candidates[0].dependencies[-1].optional_missing
    assert result == evaluate_query(q, paths(m, p, k, reader))
    existing = updated(q, artifacts=[metadata(q, m.accepted_artifacts[0], "EXIST001")])
    reader.disable(m.candidates[0].implementation_ref)
    value = evaluate_query(existing, paths(m, p, k, reader))
    assert value.status == "existing_artifact" and value.evaluations == 0
    assert value.candidates == () and value.clinical_effect == "NONE"


def test_and_missing_is_data_gap_not_tool_gap():
    q, m, p, _, k, reader = ready()
    q = updated(q, artifacts=q.artifacts[:1])
    result = evaluate_query(q, paths(m, p, k, reader))
    assert result.status == "gaps" and len(result.rejected) == 2
    assert {g.category for g in result.gaps} == {"patient_data_missing", "input_unsatisfied"}
    assert not result.suggestions
    q = updated(q, patient_data_types=[])
    result = evaluate_query(q, paths(m, p, k, reader))
    assert "unresolved_producer" in {g.category for g in result.gaps}
    assert result.product_demands and not result.suggestions


def test_or_safety_and_unknown_environment_do_not_revive_in_ranking():
    q, m, p, _, k, reader = ready()
    reader.disable(m.candidates[0].implementation_ref)
    result = evaluate_query(q, paths(m, p, k, reader))
    assert len(result.candidates) == len(result.rejected) == 1
    assert result.candidates[0].candidate.code.endswith("PATH1")
    reader.disable(m.candidates[1].implementation_ref)
    result = evaluate_query(q, paths(m, p, k, reader))
    assert result.status == "gaps" and all(g.category == "safety_restriction" for g in result.gaps)
    assert not result.suggestions
    q, m, p, _, k, reader = ready()
    result = evaluate_query(updated(q, environment=[]), paths(m, p, k, reader))
    assert result.status == "gaps"
    assert {"environment_unknown", "required_condition_unknown"} <= {g.reason for g in result.gaps}


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("laterality", "LEFT", "laterality_mismatch"),
        ("coordinate_space", "PIXEL_2D", "coordinate_mismatch"),
        ("unit", {"state": "known", "code": "CONCEPT:UNIT:RATIO"}, "unit_mismatch"),
        ("modality", {"state": "unknown", "code": None}, "modality_mismatch"),
        ("qc_status", "FAIL", "qc_rejected"),
        ("acquired_at", None, "time_unknown"),
        ("scope_terms", [], "scope_mismatch"),
    ],
)
def test_metadata_hard_filters(field, value, reason):
    q, m, p, _, k, reader = ready()
    values = q.model_dump(mode="json")["artifacts"]
    values[0][field] = value
    result = evaluate_query(updated(q, artifacts=values), paths(m, p, k, reader))
    assert reason in {g.reason for g in result.gaps} and not result.candidates


def test_time_scope_cross_patient_and_exact_requirement():
    q, m, p, _, k, reader = ready()
    values = q.model_dump(mode="json")["artifacts"]
    values[0]["acquired_at"] = (q.window_start - timedelta(days=1)).isoformat()
    result = evaluate_query(updated(q, artifacts=values), paths(m, p, k, reader))
    assert "time_mismatch" in {g.reason for g in result.gaps}
    for field, value in [
        ("case_code", "CASE:OID_DIFFERENT"),
        ("encounter_code", "ENCOUNTER:OID_DIFFERENT"),
        ("patient_hash", digest("other")),
        ("run_id", "RUN:OID_DIFFERENT"),
    ]:
        bad = q.model_dump(mode="json")["artifacts"]
        bad[0][field] = value
        with pytest.raises(ValueError):
            updated(q, artifacts=bad)
    with pytest.raises(ValueError):
        updated(q, requirement_ref="free text requirement")
    with pytest.raises(ValueError):
        updated(
            q,
            requirement_ref={
                **q.requirement_ref.model_dump(mode="json"),
                "checksum": digest("other"),
            },
        )


def test_multiple_step_dependency_and_cycles_and_budgets():
    q, m, p, _, k, reader = ready(chain=True)
    result = evaluate_query(q, paths(m, p, k, reader))
    # The intermediate is also an explicitly accepted synthetic artifact, but is absent initially.
    root = next(c for c in result.candidates if c.candidate.code.endswith("PATH0"))
    assert any(d.producer_options for d in root.dependencies)
    assert result.enumeration == "complete"
    q, m, p, _, k, reader = ready(chain=True, cycle=True)
    result = evaluate_query(q, paths(m, p, k, reader))
    assert "cycle_detected" in {g.reason for g in result.gaps}
    for field, value, reason in [
        ("max_depth", 1, "depth_exhausted"),
        ("max_evaluations", 1, "budget_exhausted"),
    ]:
        config = reseal(QueryConfig, {**q.config.model_dump(mode="json"), field: value})
        result = evaluate_query(updated(q, config=config), paths(m, p, k, reader))
        assert result.status == "truncated" and result.enumeration == reason and result.deferred


def test_permutation_tie_break_and_config_checksum():
    q, m, p, _, k, reader = ready()
    reversed_q = updated(
        q, artifacts=tuple(reversed(q.artifacts)), environment=tuple(reversed(q.environment))
    )
    assert normalized_request(q) == normalized_request(reversed_q)
    assert evaluate_query(q, paths(m, p, k, reader)) == evaluate_query(
        reversed_q, paths(m, p, k, reader)
    )
    config = reseal(
        QueryConfig, {**q.config.model_dump(mode="json"), "information_gain_enabled": False}
    )
    result = evaluate_query(updated(q, config=config), paths(m, p, k, reader))
    assert all(c.components[0].state == "disabled" for c in result.candidates)
    assert result.binding_checksum != evaluate_query(q, paths(m, p, k, reader)).binding_checksum


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("catalog_status", "registered", "tool_not_active"),
        ("validation_status", "smoke_tested", "tool_validation_rejected"),
        ("intended_use", "clinical_deployment", "intended_use_rejected"),
    ],
)
def test_catalog_filters_are_independent(field, value, reason):
    q, m, p, c, _, reader = ready()
    fields = c.model_dump(mode="json")
    fields["tools"][0][field] = value
    catalog = reseal(type(c), fields)
    safety = reader.current_safety(c.reference()).model_copy(
        update={"catalog_ref": catalog.reference()}
    )
    assert reason in tool_compatibility(m.candidates[0], catalog, p, safety)


def test_exact_catalog_tool_port_and_unknown_required_constraint():
    q, m, p, c, k, reader = ready()
    bad = m.candidates[0].model_dump(mode="json")
    bad["output_port"]["tool_ref"] = m.candidates[1].tool_ref.model_dump(mode="json")
    with pytest.raises(ValueError):
        tool_compatibility(bad, c, p, reader.current_safety(c.reference()))
    wrong = m.candidates[0].inputs[0].constraint.model_copy(update={"coordinate_space": "UNKNOWN"})
    reasons = artifact_compatibility(q.artifacts[1], wrong, q)
    assert "required_condition_unknown" in reasons
    with pytest.raises(ValueError):
        evaluate_query(
            q,
            paths(
                reseal(
                    Mapping,
                    {
                        **m.model_dump(mode="json"),
                        "catalog_ref": {
                            **m.catalog_ref.model_dump(mode="json"),
                            "version": "9.0.0",
                        },
                    },
                ),
                p,
                k,
                reader,
            ),
        )


def test_forged_result_construct_copy_and_resealed_inconsistency():
    q, m, p, _, k, reader = ready()
    result = evaluate_query(q, paths(m, p, k, reader))
    for updates in (
        {"status": "gaps"},
        {"candidates": (result.candidates[0],) * 2},
        {"top_k": ()},
        {"clinical_effect": "SUPPORT"},
        {"extra": "bypass"},
    ):
        with pytest.raises(ValueError):
            validate(QueryResult, result.model_copy(update=updates))
    fields = result.model_dump(mode="json")
    fields["candidates"][0]["dependencies"] = []
    with pytest.raises(ValueError):
        reseal(QueryResult, fields)
    forged = QueryResult.model_construct(**{**vars(result), "status": "gaps"})
    with pytest.raises(ValueError):
        validate(QueryResult, forged)


def test_attributed_estimates_order_changes_and_technical_failure():
    from packages.grounding import Ref
    from packages.grounding.query_models import METRICS, Component, RankingEstimate
    from packages.grounding.synthetic import ref
    from tests.grounding.query_support import request_for

    q, m, p, catalog, knowledge, _ = ready()
    sets = []
    references = []
    for index in range(2):
        values = [1.0 + index, 0.5, 1.0 + index * 9, 5.0, None]
        components = tuple(
            Component(
                name=name,
                state="unknown" if value is None else "known",
                value=value,
                source_ref=None
                if value is None
                else ref("source_version", "SOURCE:SYNTHETIC:ESTIMATE"),
            )
            for name, value in zip(METRICS, values, strict=True)
        )
        sets.append(components)
        references.append(
            Ref(
                kind="estimate",
                code=f"SOURCE:SYNTHETIC:ESTIMATE{index}",
                version="1.0.0",
                checksum=digest(components),
            )
        )
    catalog = reseal(type(catalog), {**catalog.model_dump(mode="json"), "estimates": references})
    mapping = reseal(
        Mapping,
        {
            **m.model_dump(mode="json"),
            "catalog_ref": catalog.reference(),
            "candidates": [
                {**c.model_dump(mode="json"), "estimate_ref": r}
                for c, r in zip(m.candidates, references, strict=True)
            ],
        },
    )
    q = request_for(mapping, p, artifacts=q.artifacts)
    reader = SyntheticCatalogReader([catalog])

    class Estimates:
        def read_estimate(self, catalog_ref, estimate_ref):
            return RankingEstimate(
                catalog_ref=catalog_ref,
                estimate_ref=estimate_ref,
                components=sets[references.index(estimate_ref)],
            )

    result = evaluate_query(q, paths(mapping, p, knowledge, reader), estimates=Estimates())
    assert result.candidates[0].candidate.code.endswith("PATH1")
    config = reseal(
        QueryConfig, {**q.config.model_dump(mode="json"), "order": ["cost", "information_gain"]}
    )
    cheap = evaluate_query(
        updated(q, config=config), paths(mapping, p, knowledge, reader), estimates=Estimates()
    )
    assert cheap.candidates[0].candidate.code.endswith("PATH0")
    assert result.checksum != cheap.checksum
    assert all(c.components[-1].state == "unknown" for c in result.candidates)
    fields = result.model_dump(mode="json")
    fields["candidates"].reverse()
    fields["top_k"] = [fields["candidates"][0]["identity"]]
    with pytest.raises(ValueError, match="ordering"):
        reseal(QueryResult, fields)
    with pytest.raises(ValueError, match="estimate_reader_required"):
        evaluate_query(q, paths(mapping, p, knowledge, reader))

    class Failed:
        def read_estimate(self, *_):
            raise OSError("synthetic estimate provider failed")

    with pytest.raises(OSError):
        evaluate_query(q, paths(mapping, p, knowledge, reader), estimates=Failed())
    reader.disable(mapping.candidates[1].implementation_ref)
    assert (
        evaluate_query(q, paths(mapping, p, knowledge, reader), estimates=Estimates())
        .candidates[0]
        .candidate.code.endswith("PATH0")
    )


def test_multi_reason_missing_registry_and_security_filters():
    q, m, p, catalog, _, reader = ready()
    candidate = m.candidates[0]
    safety = reader.current_safety(catalog.reference())
    for field, expected in [
        ("artifacts", "artifact_outside_catalog"),
        ("tools", "tool_outside_catalog"),
    ]:
        data = catalog.model_dump(mode="json")
        if field == "artifacts":
            # No remaining tool produces this missing accepted type in the sparse snapshot.
            data["tools"] = data["tools"][1:]
            data["artifacts"] = data["artifacts"][1:]
        else:
            data["tools"] = data["tools"][1:]
        sparse = reseal(type(catalog), data)
        safety = validate(
            type(safety), {**safety.model_dump(mode="json"), "catalog_ref": sparse.reference()}
        )
        assert expected in tool_compatibility(candidate, sparse, p, safety)
    changed = catalog.model_dump(mode="json")
    changed["tools"][0]["intended_use"] = "clinical_deployment"
    changed["tools"][0]["validation_status"] = "smoke_tested"
    changed["tools"][0]["enabled"] = False
    catalog = reseal(type(catalog), changed)
    safety = validate(
        type(safety), {**safety.model_dump(mode="json"), "catalog_ref": catalog.reference()}
    )
    assert {"intended_use_rejected", "tool_not_active", "tool_validation_rejected"} <= set(
        tool_compatibility(candidate, catalog, p, safety)
    )


@pytest.mark.parametrize("field,value", [("max_operations", 1), ("max_findings", 2)])
def test_all_search_limits_are_explicit(field, value):
    q, m, p, _, k, reader = ready()
    if field == "max_findings":
        q = updated(q, artifacts=[], environment=[])
    config = reseal(QueryConfig, {**q.config.model_dump(mode="json"), field: value})
    result = evaluate_query(updated(q, config=config), paths(m, p, k, reader))
    assert result.status == "truncated" and result.enumeration == "budget_exhausted"
    assert result.operations <= config.max_operations and len(result.gaps) <= config.max_findings


def test_or_producers_do_not_count_as_multiple_required_artifacts():
    from packages.grounding.query_models import Dependency

    q, m, p, _, k, reader = ready(chain=True)
    result = evaluate_query(q, paths(m, p, k, reader))
    dep = next(d for c in result.candidates for d in c.dependencies if d.producer_options)
    with pytest.raises(ValueError, match="unsatisfied dependency"):
        validate(
            Dependency, {**dep.model_dump(mode="json"), "cardinality_min": 2, "cardinality_max": 2}
        )


def test_unknown_satisfied_terms_and_preconditions_are_not_authority():
    q, m, p, _, k, reader = ready()
    result = evaluate_query(updated(q, satisfied_terms=[]), paths(m, p, k, reader))
    assert not result.candidates and "required_condition_unknown" in {g.reason for g in result.gaps}
    environment = q.model_dump(mode="json")["environment"]
    environment[0]["preconditions"] = "rejected"
    result = evaluate_query(updated(q, environment=environment), paths(m, p, k, reader))
    assert len(result.candidates) == 1 and "precondition_rejected" in {
        g.reason for g in result.gaps
    }


def test_query_schemas_config_example_and_offline_boundary():
    import json
    from pathlib import Path

    from packages.grounding.query_models import QueryRequest
    from tests.grounding.query_support import config

    root = Path(__file__).resolve().parents[2]
    for model in (QueryConfig, QueryRequest, QueryResult):
        saved = json.loads(
            (root / "packages/grounding/schemas" / f"{model.__name__}.schema.json").read_text()
        )
        assert saved == model.model_json_schema()
    assert (
        QueryConfig.model_validate_json((root / "configs/grounding_query.json").read_text())
        == config()
    )
    example = QueryResult.model_validate_json(
        (root / "packages/grounding/examples/QueryResult.synthetic.json").read_text()
    )
    q, m, p, _, k, reader = ready(chain=True)
    assert example == evaluate_query(q, paths(m, p, k, reader))


@pytest.mark.parametrize("limit", [2, 3, 4, 5])
def test_small_combined_search_limits_return_results(limit):
    q, m, p, _, k, reader = ready(chain=True, cycle=True)
    config = reseal(
        QueryConfig,
        {
            **q.config.model_dump(mode="json"),
            "max_findings": limit,
            "max_evaluations": limit,
            "max_depth": 2,
        },
    )
    result = evaluate_query(updated(q, config=config), paths(m, p, k, reader))
    assert result.status == "truncated"
    assert len(result.gaps) <= limit and result.evaluations <= limit
