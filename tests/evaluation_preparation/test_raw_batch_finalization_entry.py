"""Explicit CLI delegation only; original raw/acceptance integration is tested separately."""

import json

import pytest

from packages.multimodal_research import campaign_renewal, finalization
from tests.evaluation_preparation.test_active_recorded_finalization import saved as saved


def configure(saved, monkeypatch):
    (saved.previous / "entry.json").unlink()
    plan = dict(
        schema_version="multimodal-campaign-renewal/1.35.0",
        mode="review",
        root_direction_review={"owner_rechecks": True},
        text_review_method={"owner_rechecks": True},
    )
    (saved.previous / "protocol.json").write_text(json.dumps(plan))
    monkeypatch.setattr(
        campaign_renewal, "completed_parent", lambda p: (saved.campaign, saved.failure)
    )
    return plan


def test_new_raw_recovery_is_explicit_and_not_normal_execution(saved, monkeypatch):
    plan = configure(saved, monkeypatch)
    with pytest.raises(ValueError, match="renewal_version"):
        finalization.finalize_entry(saved.previous, saved.output)
    result = finalization.finalize_entry(saved.previous, saved.output, recover_confirmed_raw=True)
    assert result["scope"] == "delegation-only-not-real-acceptance"
    assert saved.calls[0][-1] == dict(complete=False, active=True, recover_confirmed_raw=True)
    assert json.loads((saved.previous / "protocol.json").read_bytes()) == plan


@pytest.mark.parametrize(
    "change",
    [
        dict(mode="acquisition"),
        dict(schema_version="multimodal-campaign-renewal/1.27.0"),
        dict(text_review_method=None),
        dict(root_direction_review=None),
    ],
)
def test_raw_recovery_refuses_wrong_original_protocol(saved, monkeypatch, change):
    plan = configure(saved, monkeypatch)
    plan.update(change)
    (saved.previous / "protocol.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="raw_batch_recovery_original_version"):
        finalization.finalize_entry(saved.previous, saved.output, recover_confirmed_raw=True)
    assert not saved.calls


@pytest.mark.parametrize(
    "flags", [dict(complete=True), dict(citation_rejection=True), dict(lossless_explanations=True)]
)
def test_raw_recovery_cannot_mix_authorities(saved, flags):
    with pytest.raises(ValueError, match="raw_batch_recovery_partial_deadline_only"):
        finalization.finalize_entry(
            saved.previous, saved.output, recover_confirmed_raw=True, **flags
        )
    assert not saved.calls
