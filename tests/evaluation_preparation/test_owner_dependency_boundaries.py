"""Independent red cases adopted intact; no database or external calls."""

import copy

import pytest

from packages.evidence_reuse.definition_acceptance import requirement_rows
from tests.evaluation_preparation.test_owner_dependency_definition import confirmed, raw


@pytest.mark.parametrize("port", ["read", "read_for_report"])
def test_new_confirmed_receipt_type_drift_rejected(tmp_path, port):
    owner, value, _, _, documents = confirmed(tmp_path)
    receipt = copy.deepcopy(owner.accept("typed", value))
    receipt["confirmed_basis"]["technical_checks"]["artifact_integrity"] = 1
    original = documents.read
    documents.read = lambda key: (receipt,) if key == "definition:typed" else original(key)
    with pytest.raises(ValueError):
        getattr(owner, port)("typed")


def test_new_raw_projection_type_drift_rejected():
    owner, manifest, _, _, _, value = raw()
    receipt = copy.deepcopy(owner.accept("typed", value))
    receipt["dependency_assessment"]["items"][0]["satisfied"] = 1
    with pytest.raises(ValueError):
        requirement_rows(receipt, manifest, fact_reader=owner.fact_reader)


def test_final_input_callback_fact_change_rejected(tmp_path):
    owner, value, item, _, _ = confirmed(tmp_path)
    original = owner.input_reader
    calls = []

    def late_change():
        manifest = original()
        calls.append(True)
        if len(calls) == 2:
            item.value["late_changed_fact"] = True
        return manifest

    owner.input_reader = late_change
    with pytest.raises(ValueError, match="component_observation_changed"):
        owner.build(value)
