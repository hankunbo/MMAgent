"""Separate hypothesis targets with shared facts; no synthetic clinical labels."""

from types import SimpleNamespace

import pytest

from packages.evidence_details.acceptance_models import DomainState, TaskPolicyV6
from packages.evidence_scoring.accepted import build_accepted
from packages.evidence_scoring.api import build
from packages.evidence_scoring.api import validate as validate_scores
from packages.grounding.models import digest, validate
from packages.multimodal_research.hypotheses import REGISTRY, expand_policy
from tests.evaluation_preparation.test_dynamic_evidence import dynamic_fixture, generate
from tests.evaluation_preparation.test_task_scoring import targeted


def fixture(empty=False):
    f = targeted(dynamic_fixture(1, empty=empty), "GOAL:FOUR")
    raw = f.owner.current().model_dump(mode="json")
    p = expand_policy(f.owner.current().policy)
    raw["policy"] = p
    raw["nodes"] = [p.target_code, *(i.node for i in p.items)]
    m = raw["material"]
    m["scope"]["policy_checksum"] = m["policy_ref"]["checksum"] = digest(p)
    m["scope"].pop("material_checksum")
    m["scope"]["material_checksum"] = digest(m)
    f.owner.state = validate(DomainState, raw)
    return f


def test_four_targets_original_batch_acceptance_exact_score_and_no_pooling():
    f = fixture()
    sub = generate(f)
    assert len(sub.candidate.items) == 4 and len(f.owner.state.material.observations) == 1
    f.acceptance.submit(sub, key="hypotheses")
    assert f.acceptance.accept(key="hypotheses").status == "accepted"
    s = f.acceptance.read(key="hypotheses")[2]["scores"]
    assert s["schema_version"] == "case-research-scoring/10.0.0"
    assert len({r["target"] for r in s["reviews"]}) == 4
    assert s["root"]["summary"]["confidence"]["value"] is None
    assert len(s["root"]["summary"]["distinct_evidence_units"]) == 1
    assert s["hypothesis_comparison"]["status"] == "tied"
    assert s["hypothesis_comparison"]["class_label"] is None
    assert len(set(s["evaluation_mapping"].values())) == 3
    base = dict(state=f.owner.state, candidate=sub.candidate)
    validate_scores(base, s)
    with pytest.raises(ValueError, match="score_policy_version_mismatch"):
        build(base, "9.0.0")
    for removal in s["removals"]:
        assert removal["hypothesis_comparison"]["status"] == "unavailable"


def test_empty_four_hypotheses_preserve_gaps_without_invented_scores():
    f = fixture(empty=True)
    s = build_accepted(f.owner.state, None)
    assert len(s["root"]["summary"]["hypotheses"]) == 4
    assert all(
        h["summary"]["confidence"]["value"] is None for h in s["root"]["summary"]["hypotheses"]
    )
    assert len(s["requirements"]) == 4 and len(s["groups"]) == 1
    assert s["hypothesis_comparison"]["status"] == "unavailable"


@pytest.mark.parametrize("mutation", ["omit", "foreign", "duplicate", "model"])
def test_binding_cannot_omit_target_duplicate_vote_or_relabel_classifier(mutation):
    raw = fixture().owner.state.policy.model_dump(mode="json")
    key = next(iter(raw["hypothesis_bindings"]))
    if mutation == "omit":
        del raw["hypothesis_bindings"][key]
    elif mutation == "foreign":
        raw["hypothesis_bindings"][key] = "GOAL:FOREIGN"
    elif mutation == "duplicate":
        raw["hypothesis_bindings"][key] = list(raw["hypothesis_bindings"].values())[1]
    else:
        raw["items"][0].update(role="model", model_direction=0.5)
    with pytest.raises(ValueError):
        validate(TaskPolicyV6, raw)


def test_gamma_expanded_source_bound_checks_not_fixed_classifier(tmp_path):
    from packages.dataset_adapters.multimodal.adapters import prepare
    from packages.dataset_adapters.multimodal.tasks import load_intent
    from packages.multimodal_research import material
    from packages.multimodal_research.campaign import configuration
    from tests.dataset_adapters.test_multimodal import gamma
    from tests.evaluation_preparation.test_multimodal_campaign import sources

    p = prepare("gamma", gamma(tmp_path), "0001")
    task = load_intent("glaucoma_four_hypotheses", REGISTRY)[0]
    medical = sources().initial
    rows = material.requirements(p.manifest, medical, task)
    assert len(rows) == 7
    assert rows[-1]["missing_reason"] == "no_visible_functional_staging_input"
    base = dynamic_fixture(1).owner.state
    visible = SimpleNamespace(
        case=SimpleNamespace(case_code=base.material.scope.case_code, laterality="UNKNOWN"),
        checksum=base.material.scope.visible_checksum,
        components=SimpleNamespace(checksum=base.material.scope.input_checksum),
    )
    state = material.state(
        context=SimpleNamespace(
            run_id=base.material.scope.run_id, request=SimpleNamespace(task=task)
        ),
        visible=visible,
        root=SimpleNamespace(
            asset_code=base.material.scope.graph.code, checksum=base.material.scope.graph.checksum
        ),
        config=configuration(),
        medical=medical,
        ledger=rows,
        observations={
            r["id"]: material.observation(r["id"], {"finding": "synthetic finding"})
            for r in rows[:-1]
        },
        resources=base.resources,
    )
    assert isinstance(state.policy, TaskPolicyV6)
    assert len(state.policy.items) == 24 and len(state.policy.requirements) == 28
    assert len(state.policy.groups) == 7 and len(state.material.observations) == 6
    from packages.evaluation_preparation.generic_aliases import alias_review
    from packages.evidence_details.review_tasks import project_review_tasks

    aliased, _ = alias_review(state)
    projected = project_review_tasks(aliased)
    assert len(projected.tasks) == 24
    assert all("本条只评价命题" in t.definition.applicability_conditions for t in projected.tasks)
    assert all(t.goal == task.target_code for t in projected.tasks)
    # The legacy three-label protocol is not silently upgraded.
    assert len(load_intent("glaucoma_grading")[0].ordered_labels) == 3
