"""Original Python entry/freeze boundaries; synthetic, no service assembly."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import (
    active_requirements,
    definition_trial,
    entry,
    implementation,
)
from packages.multimodal_research.campaign import MultimodalCampaign
from packages.multimodal_research.definition_context import requirement_context
from packages.object_store import checksum
from tests.evaluation_preparation.test_definition_requirement_context import rows
from tests.evaluation_preparation.test_trial_entry_window import prepare_entry


class NextOwnerGate(Exception):
    pass


@pytest.mark.parametrize("version", [19, 20, 21])
def test_context_policy_is_bound_in_original_implementation_freeze(version):
    path = Path(f"configs/research/multimodal_definition_gate_v{version}.json")
    assert implementation.freeze()[str(path)] == checksum(path.read_bytes())


@pytest.mark.parametrize(
    "version,trial_version", [(18, "1.14.0"), (19, "1.15.0"), (20, "1.16.0"), (21, "1.17.0")]
)
def test_original_entry_accepts_exact_explicit_trial_before_next_owner_gate(
    tmp_path, monkeypatch, version, trial_version
):
    _, spec = prepare_entry(tmp_path, monkeypatch)
    spec["active_acquisition"] = dict(
        path="synthetic-profile-not-opened", checksum=digest("profile")
    )
    (tmp_path / "entry.json").write_text(json.dumps(spec))
    value = definition_trial.prepare(
        tmp_path,
        expansion_policy=f"configs/research/multimodal_definition_gate_v{version}.json",
        authorization="synthetic original entry allowlist test",
        seconds=5400,
        entry_time_window=True,
    )
    assert value["schema_version"] == "multimodal-definition-trial/" + trial_version

    def next_gate(path):
        assert path == "synthetic-profile-not-opened"
        raise NextOwnerGate

    def forbidden():
        raise AssertionError("real owners must not be assembled by this CPU test")

    monkeypatch.setattr(active_requirements, "load", next_gate)
    monkeypatch.setattr(entry, "owners", forbidden)
    with pytest.raises(NextOwnerGate):
        entry.assemble_entry(tmp_path, trial_initialize=True)


@pytest.mark.parametrize("trial_version", ["1.15.0", "1.16.0", "1.17.0"])
def test_new_component_trial_does_not_fall_back_to_legacy_campaign(
    tmp_path, monkeypatch, trial_version
):
    monkeypatch.setattr(implementation, "freeze", lambda: {})
    c = object.__new__(MultimodalCampaign)
    c.output, c.campaign_id = tmp_path, "synthetic-context-version"
    checksum = SimpleNamespace(checksum=digest("synthetic"))
    c.base = SimpleNamespace(
        manifest=checksum,
        visible_value=checksum,
        task=checksum,
        view_checksum=digest("views"),
        sources=SimpleNamespace(initial=[]),
    )
    c.config, c.targets = {}, ("component",)
    c.previous_attempts = lambda: []
    c.trial = dict(
        schema_version="multimodal-definition-trial/" + trial_version, max_new_requests=12
    )
    actual = c.freeze()
    assert actual["schema_version"] == "multimodal-live-campaign/1.9.0"
    assert c.freeze() == actual


def test_component_identifier_suffix_does_not_escape_in_requirement_id():
    values = rows()
    # Exact original active_requirements.project naming rule for >1 same-modality component:
    # identifier = 'active:' + check['id'] + ':' + component.id.
    values[0].update(
        id="active:rim_diffuse:PRIVATE_COMPONENT_PATIENT", component="PRIVATE_COMPONENT_PATIENT"
    )
    plans = [dict(target_relation_gap_ids=[values[0]["id"]])]
    public = requirement_context(values, plans, {})
    assert public["requirements"][0]["requested_revision"] == "target_relation"
    assert "PRIVATE_COMPONENT_PATIENT" not in json.dumps(public)
