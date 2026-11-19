"""Fresh structural coverage declares an existing bounded method alternative."""

from pathlib import Path

import pytest

from packages.dataset_adapters.multimodal.adapters import prepare
from packages.multimodal_research.active_acquisition import install
from packages.multimodal_research.active_requirements import load
from packages.multimodal_research.campaign import MultimodalCampaign
from packages.multimodal_research.entry_preflight import LocalReadinessError, check_local_route
from packages.multimodal_research.implementation import freeze
from tests.dataset_adapters.test_multimodal import gamma
from tests.evaluation_preparation.test_method_alternative import setup

PROFILE = "configs/research/multimodal_structural_coverage_v1.json"
OLD = "configs/research/multimodal_structural_context_v2.json"


def test_coverage_profile_retains_definitions_checks_and_budgets():
    original = Path(OLD).read_bytes()
    old, new = load(OLD), load(PROFILE)
    assert new == dict(
        old,
        id="glaucoma-structural-coverage-v1",
        native_unavailable_policy="unavailable-native-to-visible-description/1.0.0",
    )
    assert Path(OLD).read_bytes() == original
    assert PROFILE in freeze()


def test_single_color_component_uses_qualified_alternative_not_fake_measurement(
    tmp_path, monkeypatch
):
    factory, _, _ = setup(tmp_path, monkeypatch)
    base = factory()
    base.prepared = prepare("gamma", gamma(tmp_path / "gamma"), "0001", selection="training")
    base.manifest = base.prepared.manifest
    profile = load(PROFILE)
    with pytest.raises(LocalReadinessError, match="fundus_installation_resource_changed"):
        check_local_route(base.manifest, base.task, load(OLD), None)
    assert check_local_route(base.manifest, base.task, profile, None) == (
        "vision_alternative_native_unavailable"
    )
    install(base, profile, native_runner=lambda *a, **kw: pytest.fail("unqualified dispatch"))
    methods = base.active_owner.methods
    assert methods["readiness"]["native_qualified"] is False
    assert methods["readiness"]["method"] == "vision"
    ids = {t["id"] for t in methods["targets"]}
    assert ids == {"active:cup_geometry"}
    assert not ids.intersection(base.observations)
    assert base.active_owner.native is None
    selected = MultimodalCampaign.select_targets(base, {c.id for c in base.manifest.components})
    assert ids <= set(selected)
