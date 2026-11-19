"""New hypotheses share arithmetic, never legacy SLO semantics."""

import pytest

from packages.evaluation_preparation.quality_materials import upgrade
from packages.evidence_details.acceptance_models import DomainState, TaskPolicyV5
from packages.evidence_scoring.accepted import build_accepted, rows_from_candidate
from packages.evidence_scoring.api import validate as validate_scores
from packages.evidence_scoring.continuous import summarize
from packages.evidence_scoring.task_policy import POLICY, SCOPES
from packages.grounding.models import digest, validate
from tests.evaluation_preparation.test_dynamic_evidence import dynamic_fixture, generate


def targeted(f, target="GOAL:OBSERVED_CHANGE"):
    seed = upgrade(f.owner.current())
    raw = seed.model_dump(mode="json")
    p = raw["policy"]
    p.update(
        schema_version="research-task-policy/5.0.0",
        target_code=target,
        target_summary="Synthetic source-bound change hypothesis",
        aggregation_checksum=digest(POLICY),
    )
    for g in p["groups"]:
        g["importance"] = dict(zip(SCOPES, g["importance"].values(), strict=True))
    p = validate(TaskPolicyV5, p)
    m = raw["material"]
    m.update(goal_code=p.target_code, goal_summary=p.target_summary)
    m["scope"]["policy_checksum"] = m["policy_ref"]["checksum"] = digest(p)
    m["scope"].pop("material_checksum")
    m["scope"]["material_checksum"] = digest(m)
    raw["policy"] = p
    f.owner.state = validate(DomainState, raw)
    return f


def test_new_target_original_acceptance_score_and_exact_replay():
    f = targeted(dynamic_fixture(1))
    sub = generate(f)
    f.acceptance.submit(sub, key="task")
    assert f.acceptance.accept(key="task").status == "accepted"
    score = f.acceptance.read(key="task")[2]["scores"]
    assert score["schema_version"] == "case-research-scoring/9.0.0"
    assert score["target"]["code"] == "GOAL:OBSERVED_CHANGE"
    assert set(score["root"]["coverage"]) == set(SCOPES)
    assert score["reviews"][0]["target"] == "GOAL:OBSERVED_CHANGE"
    validate_scores(dict(state=f.owner.current(), candidate=sub.candidate), score)
    assert score["removals"][0]["summary"]["contributing_ids"] == []
    assert f.acceptance.accept(key="task").status == "accepted"
    assert f.budget.used == 1


def test_target_cannot_be_changed_without_matching_material():
    f = targeted(dynamic_fixture(1))
    raw = f.owner.current().model_dump(mode="json")
    raw["policy"]["target_code"] = "GOAL:OTHER"
    with pytest.raises(ValueError, match="task_score_target_binding"):
        validate(DomainState, raw)


def test_no_evidence_retains_task_gaps_not_disease_negative():
    f = targeted(dynamic_fixture(1, empty=True))
    score = build_accepted(f.owner.current(), None)
    assert score["root"]["summary"]["confidence"]["value"] is None
    assert score["requirement_evaluations"][0]["gap_score"] == 1
    assert not score["reviews"]


def test_mixed_targets_are_not_pooled():
    f = targeted(dynamic_fixture(1))
    sub = generate(f)
    rows = rows_from_candidate(f.owner.current(), sub.candidate)
    with pytest.raises(ValueError, match="continuous_mixed_targets"):
        summarize([rows[0], rows[0].model_copy(update=dict(id="other", target="GOAL:OTHER"))])
