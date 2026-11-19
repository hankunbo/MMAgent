"""A fresh explicit trial window cannot enlarge its original entry or old trials."""

import json

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import definition_trial as trial
from tests.evaluation_preparation.test_component_trial import envelope

POLICY = "configs/research/multimodal_definition_gate_v18.json"


def prepare_entry(tmp_path, monkeypatch, seconds=5400):
    old, spec = envelope(tmp_path, monkeypatch, policy=POLICY)
    # This is an uninitialized synthetic test envelope, not a historical run.
    (tmp_path / "definition-trial.json").unlink()
    spec["seconds"] = seconds
    (tmp_path / "entry.json").write_text(json.dumps(spec))
    return old, spec


def prepare_window(tmp_path, *, seconds=5400, policy=POLICY):
    return trial.prepare(
        tmp_path,
        expansion_policy=policy,
        authorization="synthetic fresh entry bounded time window",
        max_new_requests=12,
        seconds=seconds,
        entry_time_window=True,
    )


def test_fresh_explicit_window_is_bounded_by_parent_entry(tmp_path, monkeypatch):
    _, spec = prepare_entry(tmp_path, monkeypatch)
    value = prepare_window(tmp_path)
    assert value["schema_version"] == "multimodal-definition-trial/1.14.0"
    assert value["seconds"] == spec["seconds"] == 5400
    assert value["max_new_requests"] == 12
    assert value["expansion_policy"]["schema_version"] == "campaign-definition-expansion/1.18.0"
    assert trial.read(tmp_path) == value
    assert not (tmp_path / "protocol.json").exists()


def test_old_trial_still_rejects_long_window_without_explicit_selection(tmp_path, monkeypatch):
    prepare_entry(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        trial.prepare(
            tmp_path,
            expansion_policy=POLICY,
            authorization="synthetic old window must stay bounded",
            seconds=5400,
        )
    assert not (tmp_path / "definition-trial.json").exists()


@pytest.mark.parametrize("seconds", [None, 3600, True])
def test_new_window_rejects_missing_or_shorter_parent(tmp_path, monkeypatch, seconds):
    prepare_entry(tmp_path, monkeypatch, seconds=seconds)
    with pytest.raises(ValueError, match="entry_time_window"):
        prepare_window(tmp_path)
    assert not (tmp_path / "definition-trial.json").exists()


@pytest.mark.parametrize("marker", ["protocol.json", "cancel", "active-start.json", "result.json"])
def test_new_window_never_reopens_started_or_stopped_entry(tmp_path, monkeypatch, marker):
    prepare_entry(tmp_path, monkeypatch)
    (tmp_path / marker).write_text("{}")
    with pytest.raises(ValueError, match="already_started"):
        prepare_window(tmp_path)
    assert not (tmp_path / "definition-trial.json").exists()


def test_cold_read_rejects_window_rebound_to_shorter_entry(tmp_path, monkeypatch):
    _, spec = prepare_entry(tmp_path, monkeypatch)
    value = prepare_window(tmp_path)
    spec["seconds"] = 3600
    value["entry_checksum"] = digest(spec)
    with pytest.raises(ValueError, match="entry_time_window"):
        trial.checked(value, spec)


@pytest.mark.parametrize("seconds", [5401, 7200, 899])
def test_new_window_rejects_out_of_range_duration(tmp_path, monkeypatch, seconds):
    prepare_entry(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        prepare_window(tmp_path, seconds=seconds)
    assert not (tmp_path / "definition-trial.json").exists()


def test_new_window_rejects_old_policy(tmp_path, monkeypatch):
    prepare_entry(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="entry_time_window"):
        prepare_window(tmp_path, policy="configs/research/multimodal_definition_gate_v17.json")
    assert not (tmp_path / "definition-trial.json").exists()


def test_public_cli_forwards_explicit_window_without_initialization(monkeypatch, capsys):
    import sys

    from apps import multimodal_case

    seen = []
    monkeypatch.setattr(trial, "prepare", lambda *a, **kw: seen.append((a, kw)) or {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "multimodal_case",
            "definition-trial-prepare",
            "--input",
            "synthetic-unused",
            "--expansion-policy",
            POLICY,
            "--authorization",
            "synthetic explicit window",
            "--seconds",
            "5400",
            "--entry-time-window",
        ],
    )
    assert multimodal_case.main() == 0
    assert seen[0][1]["entry_time_window"] is True
    assert seen[0][1]["seconds"] == 5400
    assert seen[0][1]["max_new_requests"] == 12
    assert json.loads(capsys.readouterr().out) == {}


@pytest.mark.parametrize("change", ["deadline", "duration", "none"])
def test_new_frozen_trial_keeps_original_lease(tmp_path, monkeypatch, change):
    from datetime import UTC, datetime, timedelta

    from packages.multimodal_research.campaign import MultimodalCampaign
    from tests.evaluation_preparation.test_definition_trial import scoped

    campaign = scoped(tmp_path, seconds=5400)
    start = datetime.now(UTC)
    lease = dict(
        started_at=start.isoformat(), deadline=(start + timedelta(seconds=5400)).isoformat()
    )
    campaign.base.active_start = lease
    campaign.trial.update(schema_version="multimodal-definition-trial/1.14.0", seconds=5400)
    campaign.protocol.update(lease)
    if change == "deadline":
        campaign.protocol["deadline"] = (start + timedelta(seconds=7200)).isoformat()
    elif change == "duration":
        campaign.trial["seconds"] = 3600
    monkeypatch.setattr(MultimodalCampaign, "freeze", lambda self: self.protocol)
    if change == "none":
        assert campaign.freeze() == campaign.protocol
    else:
        with pytest.raises(ValueError, match="lifetime_binding"):
            campaign.freeze()
