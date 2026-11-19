import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from packages.grounding.models import canonical, digest
from packages.multimodal_research import active_acquisition as active
from packages.multimodal_research import capability_window as window
from packages.multimodal_research.active_requirements import load
from tests.evaluation_preparation.test_capability_transition import fixture


@pytest.mark.parametrize("fault", [None, "ledger", "revision", "source", "parent", "cycle"])
def test_original_definition_owner_survives_two_selected_ledgers(tmp_path, fault):
    base, materials, revision = fixture(tmp_path / "data")
    base.contexts = {}
    base.observations = {}
    base.active_owner = None
    profile = load("configs/research/multimodal_catalog_acquisition_v7.json")
    first = active.successor_projection(base, profile)
    out = tmp_path / "selected"
    out.mkdir()
    (out / "input-capabilities.json").write_text(json.dumps(first["capabilities"]))
    selected = window.projected_base(SimpleNamespace(base=base), first, profile, out)
    assert digest(selected.ledger) != digest(revision["rows"])
    before = canonical(base.ledger)
    if fault == "ledger":
        selected.ledger[0]["definition"] = selected.ledger[1]["definition"]
    elif fault == "revision":
        selected.capability_transition = deepcopy(first)
        selected.capability_transition["definition_revision_checksum"] = digest("forged")
    elif fault == "source":
        materials.value["state"] = "withdrawn"
    elif fault == "parent":
        selected.capability_parent = None
    elif fault == "cycle":
        base.capability_parent = selected
    if fault:
        with pytest.raises(
            ValueError, match="capability_transition_|definition_draft_exact_excerpt"
        ):
            active.successor_projection(
                selected,
                load("configs/research/multimodal_catalog_acquisition_v8.json"),
                explicit_measurement_transition=True,
            )
    else:
        second = active.successor_projection(
            selected,
            load("configs/research/multimodal_catalog_acquisition_v8.json"),
            explicit_measurement_transition=True,
        )
        assert second["definition_revision_checksum"] == digest(revision)
        assert second["coverage"]["selected"] == len(second["requirements"])
        assert {r["id"] for r in base.ledger} <= {r["id"] for r in second["requirements"]}
        assert next(
            r for r in second["requirements"] if r["id"] == base.ledger[-1]["id"]
        ) == json.loads(canonical(base.ledger[-1]))
        assert (
            active.successor_projection(
                selected,
                load("configs/research/multimodal_catalog_acquisition_v8.json"),
                explicit_measurement_transition=True,
            )
            == second
        )
    assert canonical(base.ledger) == before


@pytest.mark.parametrize("changed", [False, True])
def test_native_reference_survives_saved_json_transition(tmp_path, changed, monkeypatch):
    from tests.evaluation_preparation.test_capability_transition import native_fixture

    base, native, reads = native_fixture(tmp_path / "data")
    base.contexts = {}
    from packages.multimodal_research import native_fundus

    def qualified_fixture(actual, record):
        # Unit boundary: native payload comes from the existing synthetic fixture.
        # Current input/definition/catalog owners remain real in this test.
        assert actual.manifest == base.manifest
        assert record == native
        reads.append("qualified_fixture")

    monkeypatch.setattr(native_fundus, "recheck", qualified_fixture)
    profile = load("configs/research/multimodal_catalog_acquisition_v7.json")
    first = active.successor_projection(base, profile)
    stored = json.loads(canonical(first))
    assert stored != first  # Original typed resource reference becomes durable JSON.
    out = tmp_path / "saved"
    out.mkdir()
    (out / "input-capabilities.json").write_text(json.dumps(stored["capabilities"]))
    if changed:
        stored["retained_native_observations"][0]["quality_source"]["code"] = "ASSET:WRONG"
    selected = window.projected_base(SimpleNamespace(base=base), stored, profile, out)
    if changed:
        with pytest.raises(ValueError, match="capability_transition_parent_owner_changed"):
            active.successor_projection(selected, profile)
    else:
        result = active.successor_projection(selected, profile)
        assert result["coverage"]["selected"] == len(result["requirements"])
        assert canonical(result["retained_native_observations"]) == canonical(
            first["retained_native_observations"]
        )
        assert reads
