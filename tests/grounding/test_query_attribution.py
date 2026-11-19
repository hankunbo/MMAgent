"""05B attribution repair red/regression: product demand is not consumer capability."""

import pytest

from packages.grounding import digest, validate
from packages.grounding.query import evaluate_query
from packages.grounding.query_models import QueryConfig, QueryResult
from tests.grounding.attribution_support import DISC_CAPABILITY, producer_ready
from tests.grounding.query_support import metadata, paths, ready, updated
from tests.grounding.test_models import reseal


def evaluate(data):
    q, m, p, _, k, reader = data
    return evaluate_query(q, paths(m, p, k, reader))


@pytest.mark.parametrize("both", [False, True])
def test_unknown_product_never_targets_existing_consumer_and_keeps_distinct_demands(both):
    q, m, p, c, k, reader = ready()
    q = updated(q, artifacts=[] if both else q.artifacts[:1], patient_data_types=[])
    result = evaluate((q, m, p, c, k, reader))
    assert not result.suggestions, "missing mask must not request the registered ratio capability"
    demands = result.product_demands
    assert len(demands) == (2 if both else 1)
    assert {d.resolution for d in demands} == {"unresolved_producer"}
    assert all(
        d.capability_gap_status == "not_constructible_from_product_requirement" for d in demands
    )
    assert all(len(d.affected_inputs) == 2 for d in demands)
    assert all(d.catalog_ref == q.lock.catalog_ref and not d.catalog_producers for d in demands)
    assert len({d.constraint.artifact_ref for d in demands}) == len(demands)
    for demand in demands:
        for impact in demand.affected_inputs:
            assert impact.input_port.tool_ref == impact.source.candidate.tool_ref
            assert impact.source.path[0].endswith(("PATH0", "PATH1"))


def test_catalog_producer_without_mapping_is_not_unregistered_or_no_tool():
    data = producer_ready(mapped=False)
    q, m, _, catalog, _, _ = data
    assert any(t.outputs[0] == m.candidates[0].inputs[0].constraint for t in catalog.tools)
    result = evaluate(data)
    assert "capability_unregistered" not in {g.reason for g in result.gaps}
    assert "unmapped_in_locked_view" in {g.reason for g in result.gaps}
    assert not result.suggestions
    (demand,) = result.product_demands
    assert demand.resolution == "unmapped_in_locked_view"
    assert demand.catalog_ref == q.lock.catalog_ref
    assert demand.catalog_producers[0].capability_refs[0].code == DISC_CAPABILITY


@pytest.mark.parametrize("failure", ["environment", "validation", "disabled"])
def test_failed_producer_target_is_exact_and_parent_only_affected_context(failure):
    data = list(
        producer_ready(
            validation="smoke_tested" if failure == "validation" else "auto_validated_research"
        )
    )
    q, m, _, _, _, reader = data
    # Supply the producer's raw image so only its own failure blocks the masks.
    q = updated(
        q, artifacts=[*q.artifacts, metadata(q, m.candidates[-1].inputs[0].constraint, "IMAGE001")]
    )
    if failure == "environment":
        q = updated(
            q,
            environment=[
                e
                for e in q.environment
                if e.implementation_ref != m.candidates[-1].implementation_ref
            ],
        )
    if failure == "disabled":
        reader.disable(m.candidates[-1].implementation_ref)
    data[0] = q
    result = evaluate(data)
    if failure == "disabled":
        assert not result.suggestions
        assert any(
            g.reason == "tool_safety_revoked"
            and g.source.candidate.capability_ref.code == DISC_CAPABILITY
            for g in result.gaps
        )
    else:
        assert {s.capability_gap.capability_code for s in result.suggestions} == {DISC_CAPABILITY}
        (suggestion,) = result.suggestions
        assert suggestion.target_capability_ref == m.candidates[-1].capability_ref
        assert suggestion.constraint == m.candidates[-1].output
        assert len(suggestion.affected_inputs) == 2
        assert {p.candidate.capability_ref.code for p in suggestion.source_paths} == {
            DISC_CAPABILITY
        }
    assert result.status == "gaps"


def test_parent_own_environment_failure_still_has_legitimate_suggestion():
    q, m, p, c, k, reader = ready()
    result = evaluate((updated(q, environment=[]), m, p, c, k, reader))
    assert result.suggestions
    assert {s.capability_gap.capability_code for s in result.suggestions} == {
        m.candidates[0].capability_ref.code
    }
    assert {s.constraint for s in result.suggestions} == {c.output for c in m.candidates}
    assert not result.product_demands


def test_raw_missing_patient_input_is_not_algorithm_extension():
    q, m, p, c, k, reader = ready()
    result = evaluate((updated(q, artifacts=q.artifacts[:1]), m, p, c, k, reader))
    assert not result.suggestions
    assert {d.resolution for d in result.product_demands} == {"patient_data_missing"}


def test_satisfied_alternative_asset_failure_is_only_diagnostic():
    q, m, p, c, k, reader = ready()
    bad = metadata(q, m.candidates[0].inputs[0].constraint, "BAD00001")
    bad["qc_status"] = "FAIL"
    q = updated(q, artifacts=[*q.artifacts, bad], environment=[q.environment[0]])
    result = evaluate((q, m, p, c, k, reader))
    assert any(g.reason == "qc_rejected" for g in result.gaps)
    assert all("INPUT_MISMATCH" not in s.capability_gap.gap_codes for s in result.suggestions)
    assert not result.product_demands


def test_shared_demands_and_truncation_are_deterministic():
    q, m, p, c, k, reader = ready()
    q = updated(q, artifacts=[], patient_data_types=[])
    result = evaluate((q, m, p, c, k, reader))
    assert len(result.product_demands) == 2
    assert result == evaluate(
        (updated(q, environment=tuple(reversed(q.environment))), m, p, c, k, reader)
    )
    limit = reseal(QueryConfig, {**q.config.model_dump(mode="json"), "max_evaluations": 1})
    truncated = evaluate((updated(q, config=limit), m, p, c, k, reader))
    assert truncated.status == "truncated" and not truncated.suggestions
    assert all(len(d.affected_inputs) == 1 for d in truncated.product_demands)


@pytest.mark.parametrize("mutation", ["target", "findings", "constraint", "copy", "construct"])
def test_resealed_suggestion_semantics_revalidated(mutation):
    q, m, p, c, k, reader = ready()
    result = evaluate((updated(q, environment=[]), m, p, c, k, reader))
    fields = result.model_dump(mode="json")
    suggestion = fields["suggestions"][0]
    if mutation in {"target", "copy", "construct"}:
        suggestion["capability_gap"]["capability_code"] = DISC_CAPABILITY
    elif mutation == "findings":
        suggestion["finding_checksums"] = [
            digest(
                next(
                    g
                    for g in result.gaps
                    if g.reason == "environment_unknown"
                    and digest(g) not in result.suggestions[0].finding_checksums
                )
            )
        ]
    else:
        suggestion["constraint"] = m.candidates[0].inputs[0].constraint.model_dump(mode="json")
    fields.pop("checksum")
    fields["checksum"] = digest(fields)
    with pytest.raises(ValueError):
        if mutation == "copy":
            validate(QueryResult, result.model_copy(update=fields))
        elif mutation == "construct":
            validate(QueryResult, QueryResult.model_construct(**fields))
        else:
            validate(QueryResult, fields)


def test_legal_or_producer_suppresses_failed_option_extension():
    from tests.grounding.attribution_support import alternative_graph

    q, views = alternative_graph()
    result = evaluate_query(q, views)
    assert result.status == "candidates" and len(result.candidates) == 2
    assert any(g.reason == "environment_unknown" for g in result.gaps)
    assert not result.suggestions and not result.product_demands
    # Both producer options failing remains one product need, with exact producer target.
    q = updated(q, environment=q.environment[:2])
    failed = evaluate_query(q, views)
    assert failed.status == "gaps" and len(failed.product_demands) == 1
    assert len(failed.product_demands[0].affected_inputs) == 2
    assert {s.target_capability_ref.code for s in failed.suggestions} == {DISC_CAPABILITY}
    assert len(failed.suggestions) == 2  # Different exact implementations remain separate.


@pytest.mark.parametrize(
    "field,value",
    [
        ("laterality", "LEFT"),
        ("coordinate_space", "PIXEL_2D"),
        ("unit", {"state": "known", "code": "CONCEPT:UNIT:RATIO"}),
    ],
)
def test_different_input_constraints_never_merge(field, value):
    from packages.grounding import CatalogSnapshot, Mapping
    from packages.grounding.synthetic import SyntheticCatalogReader
    from tests.grounding.query_support import request_for

    q, m, p, c, k, _ = ready()
    fields = c.model_dump(mode="json")
    fields["tools"][1]["inputs"][0]["constraint"][field] = value
    c = reseal(CatalogSnapshot, fields)
    fields = m.model_dump(mode="json")
    fields["catalog_ref"] = c.reference()
    fields["candidates"][1]["inputs"][0]["constraint"][field] = value
    m = reseal(Mapping, fields)
    q = request_for(m, p, artifacts=q.artifacts[:1], patient_data_types=[])
    result = evaluate((q, m, p, c, k, SyntheticCatalogReader([c])))
    assert len(result.product_demands) == 2
    assert all(len(d.affected_inputs) == 1 for d in result.product_demands)
    assert not result.suggestions


def test_old_result_schema_cannot_be_silently_resealed():
    result = evaluate(ready())
    fields = result.model_dump(mode="json")
    fields["schema_version"] = "grounding-query-result/1.0.0"
    with pytest.raises(ValueError):
        reseal(QueryResult, fields)


def test_attribution_examples_schema_and_public_ports():
    import json
    from pathlib import Path

    import packages.grounding as public
    from packages.grounding import query_models
    from tests.grounding.attribution_support import attribution_examples

    for name in (
        "Finding",
        "PathSource",
        "CatalogProducer",
        "InputImpact",
        "ProductDemand",
        "ExtensionSuggestion",
    ):
        assert getattr(public, name) is getattr(query_models, name)
    examples = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "packages/grounding/examples/Attribution.synthetic.json"
        ).read_text()
    )
    assert examples["synthetic"] is True
    assert {
        k: validate(QueryResult, v) for k, v in examples["results"].items()
    } == attribution_examples()
