"""Recover legitimate multi-stage lineage, not cycles or arbitrarily deep histories."""

import json

import pytest

from packages.multimodal_research.campaign_renewal import MAX_HISTORY_DEPTH, completed_parent


def test_self_referencing_renewal_fails_before_any_owner_connection(tmp_path):
    (tmp_path / "protocol.json").write_text(
        json.dumps(dict(schema_version="multimodal-campaign-renewal/1.20.0", facts=str(tmp_path)))
    )
    with pytest.raises(ValueError, match="renewal_history_cycle"):
        completed_parent(tmp_path)


def test_legitimate_deeper_history_still_reaches_original_owner(monkeypatch, tmp_path):
    (tmp_path / "protocol.json").write_text(json.dumps(dict(schema_version="original-test-only")))
    seen = []
    monkeypatch.setattr(
        "packages.multimodal_research.campaign_renewal.saved_campaign",
        lambda path: seen.append(path),
    )
    completed_parent(tmp_path, depth=5)
    assert seen == [tmp_path]
    with pytest.raises(ValueError, match="renewal_history_depth"):
        completed_parent(tmp_path, depth=MAX_HISTORY_DEPTH + 1)
    assert len(seen) == 1
