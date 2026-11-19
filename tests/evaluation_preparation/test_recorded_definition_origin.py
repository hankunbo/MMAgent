"""Narrow closure-to-definition boundary, using the existing original-owner fixture."""

import json
from copy import deepcopy

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import campaign_renewal as r
from tests.evaluation_preparation import test_recorded_acquisition_origin as original_fixtures

closure = original_fixtures.closure
local_closure = original_fixtures.local_closure


def saved(tmp_path, f):
    (tmp_path / "result.json").write_text(json.dumps(f.saved))
    return tmp_path


def test_original_owner_proof_authorizes_new_definition_origin_only(local_closure, tmp_path):
    f = local_closure
    before = deepcopy(f.values)
    result = r.recorded_definition_origin(f.parent, f.old, saved(tmp_path, f))
    assert result["proof"]["confirmation"]["acceptance_key"] == "recorded-partition-finalization:1"
    assert f.calls == ["current", "saved-proof"]
    assert f.old["complete"] is False and f.values == before


@pytest.mark.parametrize(
    "change", ["wrong_file", "wrong_original", "already_complete", "symlink", "owner_missing"]
)
def test_unverified_local_closure_never_authorizes(local_closure, tmp_path, change):
    f = local_closure
    p = saved(tmp_path, f)
    original = deepcopy(f.old)
    if change == "wrong_file":
        wrong = deepcopy(f.saved)
        wrong["new_requests"] = 1
        (p / "result.json").write_text(json.dumps(wrong))
    elif change == "wrong_original":
        original["new_requests"] = 99
    elif change == "already_complete":
        original["complete"] = True
    elif change == "symlink":
        link = tmp_path / "alias"
        link.symlink_to(tmp_path, target_is_directory=True)
        p = link
    else:
        f.values.pop("recorded-partition-finalization:result", None)
        f.saved["complete"] = False
    with pytest.raises(ValueError):
        r.recorded_definition_origin(f.parent, original, p)


def test_cold_check_revalidates_owner_and_detects_closure_change(local_closure, tmp_path):
    f = local_closure
    closed = tmp_path / "closed"
    closed.mkdir()
    saved(closed, f)
    facts = tmp_path / "facts"
    facts.mkdir()
    (facts / "result.json").write_text(json.dumps(f.old))
    proof = r.recorded_definition_origin(f.parent, f.old, closed)
    plan = dict(
        facts=str(facts),
        tip=dict(path=str(facts)),
        original_fact_result=digest(f.old),
        recorded_definition_origin=proof,
    )
    r.recorded_definition_check(f.parent, plan)
    f.saved["finalization_grant_checksum"] = digest("changed")
    with pytest.raises(ValueError):
        r.recorded_definition_check(f.parent, plan)


def test_version_is_explicit_and_old_protocol_not_reinterpreted():
    assert r.current_increment(dict(schema_version=r.RECORDED_DEFINITION_VERSION))
    assert not r.current_increment(dict(schema_version="multimodal-campaign-renewal/1.47.0"))
    from packages.multimodal_research.execution_lease import PROFILE

    plan = dict(
        schema_version=r.RECORDED_DEFINITION_VERSION,
        mode="definitions",
        definition_increment_only=True,
        expansion_policy=dict(schema_version="campaign-definition-expansion/1.35.0"),
        execution_lease_profile=PROFILE,
        new_public_search_allowed=False,
        recorded_definition_origin={"proof": "bound_by_owner"},
    )
    r.validate_definition_runtime(plan)
    plan.pop("recorded_definition_origin")
    with pytest.raises(ValueError, match="recorded_definition_protocol_version"):
        r.validate_definition_runtime(plan)
    plan["schema_version"] = r.RUNTIME_DEFINITION_VERSION
    r.validate_definition_runtime(plan)


def test_new_version_partial_operation_closure_keeps_exact_binding():
    from packages.multimodal_research.stage_failure import check_definition_binding
    from tests.evaluation_preparation.test_interrupted_definition_closure import fixture

    c, docs, values, _ = fixture()
    c.protocol["schema_version"] = r.RECORDED_DEFINITION_VERSION
    for records in values.values():
        records[0]["binding"]["protocol_checksum"] = digest(c.protocol)
    before = deepcopy(values)
    check_definition_binding(c, docs)
    assert values == before
    values["definition:operation:0"][0]["binding"]["target"] = "GOAL:WRONG"
    with pytest.raises(ValueError):
        check_definition_binding(c, docs)
