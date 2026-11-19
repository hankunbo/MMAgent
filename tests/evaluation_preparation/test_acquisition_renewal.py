"""Bounded command composition reserves only pending Review, keeps progress honest."""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research.campaign_renewal import RenewedCampaign


@pytest.mark.parametrize("used,acquired,stop", [(0, 5, False), (4, 9, False), (5, 10, True)])
def test_new_window_reserves_new_reviews_not_old_accepted_items(tmp_path, used, acquired, stop):
    c = RenewedCampaign.__new__(RenewedCampaign)
    c.output = tmp_path
    c.base = SimpleNamespace(
        active_schedule_v19=True,
        observations={str(i): None for i in range(acquired)},
        manifest=SimpleNamespace(checksum=digest("input")),
    )
    c.config = SimpleNamespace(review_batching=SimpleNamespace(batch_size=1))
    c.protocol = dict(
        mode="acquisition",
        max_new_requests=12,
        deadline=(datetime.now(UTC) + timedelta(seconds=5400)).isoformat(),
    )
    c.reused_review_count = 5
    c.usage = lambda: used
    c.previous_attempts = lambda: [dict(requests=11)]
    c.accepted_observations = lambda: {}
    assert c.active_review_reserved() is stop
    if stop:
        saved = json.loads((tmp_path / "acquisition-budget-stop.json").read_bytes())
        assert saved["acquired_at_decision"] == acquired - 5
        c.usage = lambda: 0
        assert c.active_review_reserved()  # Never revive after the closure decision.


def test_acquisition_mode_reuses_current_definitions_without_search():
    c = RenewedCampaign.__new__(RenewedCampaign)
    checked = []
    c.protocol = dict(mode="acquisition")
    c.base = SimpleNamespace(check=lambda: checked.append(True), definition_revision={"rows": []})
    c.targets = ("new-visual",)
    assert c.expand_definitions() == {"rows": []}
    assert checked == [True] and c.targets == ("new-visual",)


@pytest.mark.parametrize("new_status", ["obtained", "unknown", "ungradable", "missing"])
def test_intermediate_owner_snapshot_has_old_judgments_but_no_new_score(tmp_path, new_status):
    from packages.multimodal_research.report import export_acquisition_progress
    from tests.agent_provider.test_accepted_review_reuse import accepted_fixture

    f = accepted_fixture(new_observation=True)
    phase = dict(run_id="RUN:ACCEPTED_NEXT", target="item:additional", accepted=True)
    f.documents.write("dynamic-phase-result", (phase,))
    c = SimpleNamespace(
        output=tmp_path,
        campaign_id="synthetic-progress",
        protocol=dict(
            reuse_assessment_ids=list(
                f.owner.confirmed_review_reader.current(f.owner.current()).assessment_ids
            )
        ),
        documents=lambda _: f.documents,
        usage=lambda: 1,
        accepted_observations=lambda: {},
        base=SimpleNamespace(
            confirmed_review_reader=f.owner.confirmed_review_reader,
            observations={
                p.id: next(
                    o
                    for o in f.owner.state.material.observations
                    if o.reference.code == p.observation_code
                )
                for p in f.owner.state.policy.items
            },
            ledger=[
                dict(id=r.id, definition=r.definition) for r in f.owner.state.policy.requirements
            ],
        ),
    )
    additional = c.base.observations["item:additional"]
    c.base.observations["item:additional"] = additional.model_copy(update={"status": new_status})
    first = export_acquisition_progress(c, phase)
    snapshot = json.loads(__import__("pathlib").Path(first["json"]).read_bytes())
    assert snapshot["observations"] == (6 if new_status == "obtained" else 5)
    assert snapshot["observation_records"] == 6 and snapshot["new_accepted_score"] is None
    assert snapshot["observation_status_counts"][new_status] >= 1
    assert snapshot["historical_assessed"] > 0 and snapshot["final"] is False
    assert next(r for r in snapshot["rows"] if r["id"] == "item:additional")["status"] == (
        "observation_obtained_pending_review"
        if new_status == "obtained"
        else "observation_" + new_status
    )
    audit = len(f.documents.audit)
    assert export_acquisition_progress(c, phase) == first
    assert len(f.documents.audit) == audit
