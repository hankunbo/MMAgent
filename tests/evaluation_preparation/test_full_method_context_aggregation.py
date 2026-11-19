"""The real Phase projection retains full visual and selected method context."""

from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import review_context
from packages.multimodal_research.accepted_aggregation import full_scope, read_plan
from tests.evaluation_preparation.test_retained_review_context import projection_fixture


def plan(version="1.13.0"):
    return read_plan(
        dict(
            schema_version="accepted-review-aggregation-plan/" + version,
            facts="facts",
            origins=[dict(campaign="old"), dict(campaign="new")],
            scope="all_current_obtained",
            accounting="fact_lineage_and_review_origins",
            reuse_kind="owner_verified_new_batches_only",
            selection_policy="new_context_accepted_null_revision/1.0.0",
            null_revisions=[
                dict(
                    assessment_id="check:rnfl",
                    previous_run_id="RUN:OLD",
                    replacement_run_id="RUN:NEW",
                    previous_submission_checksum=digest("old"),
                    replacement_submission_checksum=digest("new"),
                    previous_item_checksum=digest("oi"),
                    replacement_item_checksum=digest("ni"),
                    new_context_checksum=digest("context"),
                    reason="new_owner_bound_method_context",
                )
            ],
        )
    )


def context(value):
    return dict(
        unit_basis="synthetic-unit",
        source_limits=["synthetic research"],
        transform=dict(kind="unchanged"),
        numeric=value,
        interpretation_limits=dict(direction="not a disease vote"),
    )


def fixture(tmp_path, monkeypatch, version="1.13.0", *, conflict=False):
    phase, original, _ = projection_fixture(tmp_path)
    old_base = phase.base
    ids = list(old_base.observations)
    visual = {ids[0]: context(1)}
    method = {ids[0] if conflict else ids[-1]: context(2)}
    old_base.execution_profile = "relay-root-question-v1"
    old_base.review_display_context = method
    old_base.review_metadata_profile = "synthetic-owner-method"
    monkeypatch.setattr(review_context, "collect_campaign", lambda base, obs: visual)
    monkeypatch.setattr(review_context, "collect", lambda base, obs, selected, **kw: method)
    campaign = SimpleNamespace(base=old_base)
    full_scope(campaign, plan(version))
    phase.base = campaign.base
    return phase, original, old_base, visual, method


def test_new_wholecase_projection_keeps_both_owner_contexts_without_extra_votes(
    tmp_path, monkeypatch
):
    phase, original, parent, visual, method = fixture(tmp_path, monkeypatch)
    before = digest(original)
    current = phase.current_projection()
    values = current.material.observations
    assert all(o in values for o in review_context.supplementary(visual))
    assert all(o in values for o in review_context.supplementary(method))
    assert len(values) == len({o.reference.code for o in values})
    assert current.policy.items == original.policy.items
    assert digest(original) == before
    assert not hasattr(parent, "aggregation_method_context_policy")
    assert all(o in values for o in parent.observations.values())


def test_old_plan_preserves_original_projection(tmp_path, monkeypatch):
    phase, _, _, visual, method = fixture(tmp_path, monkeypatch, "1.11.0")
    current = phase.current_projection()
    assert all(o in current.material.observations for o in review_context.supplementary(method))
    assert any(o not in current.material.observations for o in review_context.supplementary(visual))


def test_context_identity_conflict_is_rejected_not_overwritten(tmp_path, monkeypatch):
    phase, _, _, _, _ = fixture(tmp_path, monkeypatch, conflict=True)
    with pytest.raises(ValueError, match="accepted_aggregation_method_context_conflict"):
        phase.current_projection()


@pytest.mark.parametrize("fault", ["selected", "policy", "profile"])
def test_invalid_new_projection_scope_is_not_silently_accepted(tmp_path, monkeypatch, fault):
    phase, _, _, _, _ = fixture(tmp_path, monkeypatch)
    if fault == "selected":
        phase.base.review_target_ids = tuple(phase.base.observations)[:1]
    elif fault == "policy":
        phase.base.aggregation_method_context_policy = "caller-trust-me"
    else:
        phase.base.review_metadata_profile = None
    with pytest.raises(ValueError, match="accepted_aggregation_method_context_scope"):
        phase.current_projection()


def test_identical_context_is_deduplicated(tmp_path, monkeypatch):
    phase, _, _, visual, _ = fixture(tmp_path, monkeypatch)
    phase.base.review_display_context = visual
    monkeypatch.setattr(review_context, "collect", lambda *args, **kw: visual)
    current = phase.current_projection()
    values = current.material.observations
    for item in review_context.supplementary(visual):
        assert values.count(item) == 1
