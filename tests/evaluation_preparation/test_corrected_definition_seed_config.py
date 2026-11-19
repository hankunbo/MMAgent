"""Exact public seed repair; no Knowledge/API calls and no historic rewrite."""

import json
import re
from pathlib import Path

from packages.multimodal_research import definition_trial, implementation
from packages.multimodal_research.definition_expansion import read_expansion_policy
from packages.object_store import checksum
from tests.evaluation_preparation.test_trial_entry_window import prepare_entry

CORRECTED = Path("configs/research/multimodal_definition_gate_v21_seed_r1.json")
PREVIOUS = Path("configs/research/multimodal_definition_gate_v21.json")


def load(path):
    return json.loads(path.read_bytes())


def test_corrected_public_seeds_are_exact_original_complete_identities():
    original = load(Path("configs/research/multimodal_definition_gate_v19.json"))
    value = load(CORRECTED)
    assert value["index_keys"] == original["index_keys"]
    assert all(re.fullmatch(r"sha256:[0-9a-f]{64}", k) for k in value["index_keys"])
    previous = load(PREVIOUS)
    assert value == dict(previous, index_keys=original["index_keys"])
    # Preserve the actually executed historical configuration, including its defect.
    assert len(previous["index_keys"][0]) == 69
    assert (
        checksum(PREVIOUS.read_bytes())
        == "sha256:18cb1d3190a80ff695f7177a2033faa8765f8d8054062286b043f8598597c36f"
    )


def test_corrected_public_configuration_is_frozen_without_a_schema_change():
    value = load(CORRECTED)
    assert read_expansion_policy(value).model_dump(mode="json") == value
    assert value["schema_version"] == "campaign-definition-expansion/1.21.0"
    frozen = implementation.freeze()
    assert frozen[str(CORRECTED)] == checksum(CORRECTED.read_bytes())
    assert frozen[str(PREVIOUS)] == checksum(PREVIOUS.read_bytes())


def test_original_trial_prepare_selects_corrected_public_config(tmp_path, monkeypatch):
    prepare_entry(tmp_path, monkeypatch)
    value = definition_trial.prepare(
        tmp_path,
        expansion_policy=str(CORRECTED),
        authorization="synthetic seed identity configuration contract",
        seconds=5400,
        entry_time_window=True,
    )
    assert value["schema_version"] == "multimodal-definition-trial/1.17.0"
    assert value["expansion_policy"] == load(CORRECTED)
    assert value["max_new_requests"] == 12 and value["seconds"] == 5400
