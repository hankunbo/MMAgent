"""The combined plan must retain both owner checks, without changing old plans."""

import json
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import accepted_aggregation as a
from packages.multimodal_research import campaign_renewal
from tests.evaluation_preparation.test_source_branch_accounting import fixture


def combined(raw):
    return a.read_plan(
        dict(
            raw,
            schema_version="accepted-review-aggregation-plan/1.8.0",
            reuse_kind="owner_verified_new_batches_only",
        )
    )


def test_combined_plan_keeps_source_budget_and_old_reviews(monkeypatch, tmp_path):
    f = fixture(monkeypatch, tmp_path)
    plan = combined(f.plan.model_dump(mode="json"))
    before = digest((f.old, f.readers[0].accounting))
    attempts = a.request_history(plan, f.campaign, f.old, f.readers)
    assert sum(x["requests"] for x in attempts) == 23
    assert f.calls == ["restore", "terminal"]
    assert digest((f.old, f.readers[0].accounting)) == before
    assert isinstance(plan, a.RetainedOriginAggregationPlan)
    assert a.read_plan(plan.model_dump(mode="json")) == plan


def test_actual_entry_selects_retained_origin_reader(monkeypatch, tmp_path):
    raw = dict(
        facts=str(tmp_path / "facts"),
        origins=[dict(campaign=str(tmp_path / "prior"))],
        source_budget_origins=[
            dict(campaign=str(tmp_path / "source"), result_checksum=digest("source"))
        ],
    )
    plan = combined(raw)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan.model_dump(mode="json")))
    monkeypatch.setattr(a, "plan_campaign", lambda _: (SimpleNamespace(config={}), {}, None))
    calls = []

    def owner(entry, **kwargs):
        calls.append((entry.campaign, kwargs["reused_origin_new_items_only"]))
        raise ValueError("end_retained_origin_routing_probe")

    monkeypatch.setattr(a, "origin_reader", owner)
    with pytest.raises(ValueError, match="^end_retained_origin_routing_probe$"):
        a.aggregate_entry(path, tmp_path / "new-output")
    assert calls == [(str(tmp_path / "prior"), True)]
    assert not (tmp_path / "new-output").exists()


def test_combined_plan_still_requires_original_source_owner(monkeypatch, tmp_path):
    f = fixture(monkeypatch, tmp_path)
    plan = combined(f.plan.model_dump(mode="json"))

    def reject(*args, **kwargs):
        raise ValueError("source_qualification_revoked")

    monkeypatch.setattr(campaign_renewal, "stopped_tip", reject)
    with pytest.raises(ValueError, match="^source_qualification_revoked$"):
        a.request_history(plan, f.campaign, f.old, f.readers)


def test_legacy_plan_modes_do_not_silently_gain_new_authority(tmp_path):
    base = dict(facts=str(tmp_path / "facts"), origins=[dict(campaign=str(tmp_path / "prior"))])
    retained = a.read_plan(dict(base, schema_version="accepted-review-aggregation-plan/1.5.0"))
    assert not isinstance(retained, a.SourceBudgetAggregationPlan)
    source = a.read_plan(
        dict(
            base,
            schema_version="accepted-review-aggregation-plan/1.7.0",
            source_budget_origins=[
                dict(campaign=str(tmp_path / "source"), result_checksum=digest("source"))
            ],
        )
    )
    assert not isinstance(source, a.RetainedOriginAggregationPlan)
    with pytest.raises(ValueError):
        combined(dict(base, source_budget_origins=[]))
