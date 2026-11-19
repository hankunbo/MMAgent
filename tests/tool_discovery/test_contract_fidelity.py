"""Independent contract-information probes; original producer plus synthetic candidates."""

import time
from copy import deepcopy

import pytest

from packages.grounding.models import Candidate, digest, validate
from packages.tool_discovery import Discovery, DiscoveryPolicy
from packages.tool_discovery.discovery import contract, query
from packages.tool_discovery.models_v11 import InputMapping, Provenance, Semantics
from packages.tool_discovery.versions import parse
from tests.tool_discovery.conftest import offline_only
from tests.tool_discovery.support import record, source
from tests.tool_discovery.test_discovery import evaluate
from tests.tool_gap_queue.support import setup


@pytest.mark.parametrize("change", ["two_required_images", "optional_input"])
def test_different_input_cardinality_must_not_be_exact_task_match(tmp_path, change):
    h = setup(tmp_path)
    original = h.request.candidate
    data = original.model_dump(mode="json")
    assert data["inputs"][0]["required"] is True
    assert data["inputs"][0]["cardinality_min"] == data["inputs"][0]["cardinality_max"] == 1
    data["inputs"][0].update(
        dict(required=True, cardinality_min=2, cardinality_max=2)
        if change == "two_required_images"
        else dict(required=False, cardinality_min=0, cardinality_max=1)
    )
    alternate = validate(Candidate, data)
    assert alternate.inputs != original.inputs
    p = record(h, semantics=contract(alternate.model_dump(mode="json")))
    d = Discovery(gap_queue=h.service, repository=None, sources=(source([p]),))
    a = h.service.prepare_research(h.context.run_id, key="contract")
    q = query(h.service.recheck(a), DiscoveryPolicy())
    plan = dict(
        query=q.model_dump(mode="json"),
        sources=list(d.bindings()),
        policy=d.policy.model_dump(mode="json"),
    )
    result = d._evaluate("TASK:SYNTHETIC:CONTRACT", plan, deadline=time.monotonic() + 10)
    print(
        dict(
            change=change,
            original=original.inputs[0].model_dump(mode="json"),
            alternative=alternate.inputs[0].model_dump(mode="json"),
            projected_equal=contract(original.model_dump(mode="json")) == p.semantics,
            selected=len(result.selected),
            disposition=result.candidates[0].disposition,
            task_match=result.candidates[0].dimensions["task_match"],
        )
    )
    assert contract(original.model_dump(mode="json")) != p.semantics, (
        "input cardinality/requiredness silently erased"
    )


# These fixture imports are deliberately installed in this module.
__all__ = ["offline_only"]


@pytest.fixture
def h(tmp_path):
    return setup(tmp_path)


def changed(h, **fields):
    data = contract(h.request.candidate).model_dump(mode="json")
    data["inputs"][0].update(fields)
    return validate(Semantics, data)


@pytest.mark.parametrize(
    "fields,status,reason",
    [
        ({}, "exact", "input_contract_exact"),
        (
            {"cardinality_min": 2, "cardinality_max": 2},
            "incompatible",
            "input_cardinality_not_covered",
        ),
        (
            {"required": False, "cardinality_min": 0},
            "compatible",
            "input_count_interval_covers_requirement",
        ),
        ({"cardinality_max": 3}, "compatible", "input_count_interval_covers_requirement"),
        ({"cardinality_max": None}, "missing", "input_contract_fields_missing"),
        ({"required": None}, "missing", "input_contract_fields_missing"),
        ({"port": None}, "missing", "input_contract_fields_missing"),
    ],
)
def test_count_coverage_retains_fields(h, fields, status, reason):
    semantics = changed(h, **fields)
    result = evaluate(h, (source([record(h, semantics=semantics)]),))
    c = result.candidates[0]
    assert c.input_match.status == status and reason in c.input_match.reasons
    assert c.records[0].semantics == semantics
    assert bool(result.selected) == (status in {"exact", "compatible"})
    assert c.dimensions["task_match"] == {"exact": 1.0, "compatible": 0.75}.get(status, 0.0)
    assert parse("DiscoveryResult", result.model_dump(mode="json")) == result


@pytest.mark.parametrize(
    "fields",
    [
        {"cardinality_min": 2},
        {"required": False},
        {"cardinality_max": 0},
        {"cardinality_min": "1"},
    ],
)
def test_illegal_input_is_protocol_error(h, fields):
    with pytest.raises(ValueError):
        changed(h, **fields)


def test_empty_and_legacy_contract_are_missing_not_complete(h):
    from packages.tool_discovery.models import Provenance as LegacyProvenance

    data = record(h).model_dump(mode="json")
    data["schema_version"] = "tool-candidate-provenance/1.0.0"
    data.pop("input_mappings")
    data["semantics"].pop("schema_version")
    data["semantics"]["inputs"] = [i["constraint"] for i in data["semantics"]["inputs"]]
    old = validate(LegacyProvenance, data)
    empty = contract(h.request.candidate).model_dump(mode="json")
    empty["inputs"] = []
    result = evaluate(
        h,
        (source([old, record(h, name="empty", semantics=validate(Semantics, empty)), record(h)]),),
    )
    assert len(result.candidates) == 3 and len(result.selected) == 1
    legacy = next(c for c in result.candidates if c.records[0].schema_version.endswith("1.0.0"))
    assert legacy.input_match.status == "missing" and "legacy_input_contract" in legacy.missing
    assert legacy.records[0].model_dump(mode="json") == data
    assert parse("Provenance", data).model_dump(mode="json") == data
    with pytest.raises(ValueError, match="unsupported"):
        parse("Provenance", {**data, "schema_version": "tool-candidate-provenance/9.0.0"})


def mapped_record(h, *, extra=False):
    required = contract(h.request.candidate)
    offered = required.model_dump(mode="json")
    offered["inputs"][0]["port"]["tool_ref"]["code"] = "TOOL:SYNTHETIC:REPLACEMENT"
    offered["inputs"][0]["port"]["port_name"] = "renamed_role"
    mapping = dict(
        requirement_contract_checksum=digest(required),
        bindings=[
            dict(source=item["port"], target=target.port.model_dump(mode="json"))
            for item, target in zip(offered["inputs"], required.inputs, strict=True)
        ],
        omitted_optional=[],
    )
    if extra:
        optional = deepcopy(offered["inputs"][0])
        optional["port"]["port_name"] = "optional_auxiliary"
        optional.update(required=False, cardinality_min=0, cardinality_max=2)
        offered["inputs"].append(optional)
        mapping["omitted_optional"].append(optional["port"])
    return record(
        h, semantics=validate(Semantics, offered), input_mappings=(validate(InputMapping, mapping),)
    )


@pytest.mark.parametrize("extra", [False, True])
def test_explicit_role_mapping_and_optional_omission(h, extra):
    p = mapped_record(h, extra=extra)
    c = evaluate(h, (source([p]),)).candidates[0]
    assert c.disposition == "discovered" and c.input_match.status == "compatible"
    assert "explicit_input_role_mapping" in c.input_match.reasons
    assert bool(c.input_match.omitted_optional) == extra
    assert c.records[0].semantics == p.semantics
    unmapped = validate(Provenance, {**p.model_dump(mode="json"), "input_mappings": []})
    assert evaluate(h, (source([unmapped]),)).candidates[0].input_match.status == "missing"


def test_multiport_role_mapping_no_guessing_or_duplicate_targets(h):
    from packages.tool_discovery.matching import match_contract

    required = contract(h.request.candidate).model_dump(mode="json")
    second = deepcopy(required["inputs"][0])
    second["port"]["port_name"] = "second_role"
    second["constraint"]["laterality"] = "RIGHT"
    required["inputs"].append(second)
    required = validate(Semantics, required)
    offered = required.model_dump(mode="json")
    for i, item in enumerate(offered["inputs"]):
        item["port"]["port_name"] = f"candidate_role_{i}"
    mapping = dict(
        requirement_contract_checksum=digest(required),
        bindings=[
            dict(source=item["port"], target=target.port.model_dump(mode="json"))
            for item, target in zip(offered["inputs"], required.inputs, strict=True)
        ],
    )
    p = record(
        h, semantics=validate(Semantics, offered), input_mappings=(validate(InputMapping, mapping),)
    )
    assert match_contract(p, required).status == "compatible"
    wrong = deepcopy(mapping)
    wrong["bindings"][0]["target"], wrong["bindings"][1]["target"] = (
        wrong["bindings"][1]["target"],
        wrong["bindings"][0]["target"],
    )
    p = record(h, semantics=p.semantics, input_mappings=(validate(InputMapping, wrong),))
    assert match_contract(p, required).status == "incompatible"
    mapping["bindings"][1]["target"] = mapping["bindings"][0]["target"]
    with pytest.raises(ValueError, match="not_one_to_one"):
        validate(InputMapping, mapping)


def test_optional_owner_cannot_be_satisfied_by_required_candidate(h):
    from packages.tool_discovery.matching import match_contract

    required = changed(h, required=False, cardinality_min=0, cardinality_max=3)
    assert match_contract(record(h), required).status == "incompatible"
    offered = changed(h, required=False, cardinality_min=0, cardinality_max=4)
    assert match_contract(record(h, semantics=offered), required).status == "compatible"


def test_dedupe_preserves_distinct_counts_roles_and_all_mirror_proofs(h):
    a, b = record(h, name="a"), record(h, name="b")
    optional = record(h, name="optional", semantics=changed(h, required=False, cardinality_min=0))
    mapped = mapped_record(h)
    result = evaluate(
        h, (source([a], name="a"), source([b], name="b"), source([optional, mapped], name="other"))
    )
    assert len(result.candidates) == len(result.selected) == 3
    mirror = next(c for c in result.candidates if len(c.records) == 2)
    assert {r.record_id for r in mirror.records} == {"a", "b"}
    assert len(mirror.identity_evidence) == 2 and all(
        c.records[0].semantics.complete for c in result.candidates
    )


def test_05_and_06_formal_use_same_complete_owner_projection(tmp_path):
    from packages.tool_gap_queue import GapQueue
    from tests.grounding.query_support import updated
    from tests.tool_admission.support import fixture

    f = fixture(tmp_path)
    request = updated(
        f.q,
        environment=[
            dict(e.model_dump(mode="json"), availability="unavailable") for e in f.q.environment
        ],
    )
    f.locks.q = request
    g = GapQueue(
        repository=None,
        grounding_owner=f.grounding,
        admission_owner=f.service,
        grounding_request_reader=lambda run_id, ref: f.locks.q,
    )
    a = g.prepare_grounding(request, key="05")
    formal = g.prepare_grounding(
        request, key="formal", selection={"candidate_refs": [], "invocations": []}
    )
    assert formal.owner == "06_formal"
    expected = query(g.recheck(a), DiscoveryPolicy()).contracts
    assert query(g.recheck(formal), DiscoveryPolicy()).contracts == expected
    for c in expected:
        assert c.complete
        for i in c.inputs:
            assert set(i.model_dump()) == {
                "port",
                "constraint",
                "required",
                "cardinality_min",
                "cardinality_max",
            }


def test_absent_fields_not_defaulted_and_forged_validation_not_accepted(h):
    p = record(h).model_dump(mode="json")
    for name in ("required", "cardinality_min", "cardinality_max"):
        p["semantics"]["inputs"][0].pop(name)
    parsed = validate(Provenance, p)
    c = evaluate(h, (source([parsed]),)).candidates[0]
    assert c.input_match.status == "missing"
    assert c.records[0].semantics.inputs[0].required is None
    assert c.records[0].semantics.inputs[0].cardinality_min is None
    with pytest.raises(ValueError):
        validate(Provenance, {**p, "matched": True})
    with pytest.raises(ValueError, match="owner_input_contract_incomplete"):
        from packages.tool_discovery.models_v11 import DiscoveryQuery

        q = query(h.service.prepare_research(h.context.run_id, key="unknown"), DiscoveryPolicy())
        validate(DiscoveryQuery, {**q.model_dump(mode="json"), "contracts": [p["semantics"]]})


def test_different_inference_configuration_never_merges(h):
    p = record(h)
    other = record(h, name="other", inference_entry="another-never-executed-entry")
    result = evaluate(h, (source([p, other]),))
    assert len(result.selected) == len(result.candidates) == 2


def test_mapping_cannot_omit_required_or_guess_missing_endpoint(h):
    p = mapped_record(h, extra=True).model_dump(mode="json")
    p["semantics"]["inputs"][-1].update(required=True, cardinality_min=1)
    c = evaluate(h, (source([validate(Provenance, p)]),)).candidates[0]
    assert "required_input_cannot_be_omitted" in c.rejection_codes
    p = mapped_record(h).model_dump(mode="json")
    p["input_mappings"][0]["bindings"][0]["source"]["port_name"] = "nonexistent"
    c = evaluate(h, (source([validate(Provenance, p)]),)).candidates[0]
    assert "input_mapping_endpoint_invalid" in c.rejection_codes
