"""Generic identifiers, original key compatibility and durable cancellation."""

import json
from threading import Event

import pytest

from packages.evaluation_preparation.dynamic_campaign import Campaign
from packages.evaluation_preparation.generic_campaign import GenericCampaign
from packages.evaluation_preparation.generic_input import owner_key
from packages.research_runtime.single_case import ResearchSingleCaseService


def spec(version="1.1.0"):
    value = dict(
        schema_version="fairvlmed-run-input/" + version, key="x" * 64, sample="s" * 100 + ".npz"
    )
    if version == "1.1.0":
        value["owner_key_policy"] = "content-bound-owner-key/1.0.0"
    return value


def test_long_valid_ids_fit_original_owner_limits_and_are_distinct():
    s = spec()
    values = []
    for purpose in ("fairclip", "phase", "memory"):
        for target in ("review", "association:VASCULAR_SKELETON_DENSITY"):
            key = owner_key(s, purpose, target)
            assert key == owner_key(dict(s), purpose, target)
            ResearchSingleCaseService._validate_key(key)
            assert len(key) <= 80
            values.append(key)
    assert len(set(values)) == len(values)
    assert owner_key(dict(s, sample="other.npz"), "phase", "review") not in values
    assert owner_key(dict(s, key="other"), "phase", "review") not in values


def test_legacy_owner_keys_are_exact_and_unknown_versions_refused():
    s = spec("1.0.0")
    assert owner_key(s, "fairclip") == "fairvlmed-generic:" + s["key"] + ":fairclip"
    expected = "generic-v1:" + s["key"] + ":" + s["sample"] + ":review"
    assert owner_key(s, "phase", "review") == expected
    assert owner_key(s, "memory", "review") == "memory:" + expected
    for invalid in (spec("9.0.0"), dict(spec(), owner_key_policy="unknown")):
        with pytest.raises(ValueError, match="generic_owner_key_policy"):
            owner_key(invalid, "phase", "review")


def test_actual_generic_phase_and_archive_use_bound_keys():
    c = object.__new__(GenericCampaign)
    c.spec = spec()
    assert c.request_key(None) == owner_key(c.spec, "phase", "review")
    assert c.archive_key("glaucoma") == owner_key(c.spec, "memory", "glaucoma")
    legacy = object.__new__(Campaign)
    legacy.version, legacy.sample = "old", "sample"
    assert legacy.request_key(None) == "old:sample:review"


def test_generic_cancel_signal_observes_durable_intent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = object.__new__(GenericCampaign)
    c.spec = dict(key="cancel-probe")
    signal = c.cancellation_signal()
    assert not signal.is_set()
    path = tmp_path / ".local/fairvlmed-runs/cancel-probe"
    path.mkdir(parents=True)
    (path / "cancel-request.json").write_text(json.dumps(dict(key="cancel-probe")))
    assert signal.is_set()
    assert type(Campaign.cancellation_signal(c)) is Event
