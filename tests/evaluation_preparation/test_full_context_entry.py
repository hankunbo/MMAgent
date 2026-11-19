"""Opt-in full CLI configuration, context and legacy boundaries."""

import json
from pathlib import Path

import pytest

from packages.evidence_details.live_permission import validate_live_config
from packages.multimodal_research.campaign import configuration
from packages.multimodal_research.entry import assemble_entry, prepare_entry
from packages.multimodal_research.review_context import collect_campaign, supplementary
from tests.dataset_adapters.test_multimodal import gdp
from tests.evaluation_preparation.test_review_display_context import sample


def test_full_configuration_is_explicit_and_legacy_unchanged():
    options = dict(resilient_vision=True, directional=True, exact_primary_status=True)
    old = configuration(**options)
    new = configuration(**options, execution_profile="relay-context-v1")
    assert old == configuration(**options)
    assert new.transport.base_url == "https://api.relayrouter.ai/v1"
    assert new.transport.max_retries == 0 and new.transport.timeout_seconds == 300
    assert (
        next(r for r in new.roles if r.role == "review").prompt_version == "review-direction-2.4.0"
    )
    assert (
        next(r for r in new.roles if r.role == "vision").prompt_version
        == "vision-partial-multimodal-1.2.0"
    )
    assert new.max_requests == old.max_requests == 40
    assert new.schema_version == "agent-provider-config/9.0.0"
    assert new.review_response_policy == "review-optional-null/1.1.0"
    assert new.format_correction_budget == 1
    with pytest.raises(ValueError, match="contextual_execution_configuration"):
        configuration(execution_profile="relay-context-v1")


def test_snapshot_permission_keeps_exact_route_and_budget():
    config = configuration(
        resilient_vision=True,
        directional=True,
        exact_primary_status=True,
        execution_profile="relay-snapshot-context-v1",
    )
    assert validate_live_config(config) == config
    for field, value in (("model", "gpt-4o"), ("base_url", "https://other.invalid/v1")):
        changed = config.model_copy(
            update={"transport": config.transport.model_copy(update={field: value})}
        )
        with pytest.raises(ValueError, match="bounded_live_configuration_required"):
            validate_live_config(changed)
    with pytest.raises(ValueError, match="bounded_live_configuration_required"):
        validate_live_config(config.model_copy(update={"max_requests": 41}))


@pytest.mark.parametrize(
    "profile,version",
    [
        ("relay-context-v1", "multimodal-live-entry/1.18.0"),
        ("relay-snapshot-context-v1", "multimodal-live-entry/1.19.0"),
    ],
)
def test_actual_prepare_preflight_binds_full_profile(tmp_path, monkeypatch, profile, version):
    root, out = gdp(tmp_path / "source"), tmp_path / "case"
    prepare_entry(
        dataset="harvard_gdp",
        root=root,
        sample="data_0001.npz",
        task="advanced_glaucoma",
        output=out,
        registry="configs/research/multimodal_propositions_v1.json",
        active_acquisition="configs/research/multimodal_harvard_visual_priority_v2.json",
        resilient_vision=True,
        view_profile="detail-v1",
        input_semantics="harvard-gdp-v1",
        execution_profile=profile,
    )
    monkeypatch.setattr(
        "packages.multimodal_research.entry.owners",
        lambda: pytest.fail("preflight assembled live owner"),
    )
    assert assemble_entry(out, preflight_only=True)["status"] == "local_preflight_passed"
    path = out / "entry.json"
    raw = json.loads(path.read_bytes())
    assert raw["schema_version"] == version
    # The public selected profile is operator-versioned; historical entries
    # still retain their own exact transport. Do not pin new preparation to
    # the model that happened to occupy this profile when the test was written.
    profile_path = (
        "configs/agents/relayrouter_snapshot_evidence_v1.json"
        if profile == "relay-snapshot-context-v1"
        else "configs/agents/relayrouter_generic_evidence_v1.json"
    )
    selected_transport = json.loads(Path(profile_path).read_bytes())["transport"]
    for key in ("model", "base_url", "api_key_env"):
        assert raw["provider"]["transport"][key] == selected_transport[key]
    assert raw["seconds"] == 5400 and raw["max_requests"] == 40
    raw.pop("execution_profile")
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="contextual_execution_entry_version"):
        assemble_entry(out, preflight_only=True)


def test_campaign_context_selects_confirmed_images_not_missing_or_extra_votes():
    base, obs, _, _, _ = sample()
    base.ledger.append(dict(id="missing", component=None))
    values = {"map-check": obs}
    old = obs.model_dump(mode="json")
    context = collect_campaign(base, values)
    assert list(context) == ["map-check"] and len(supplementary(context)) == 5
    assert obs.model_dump(mode="json") == old
    assert collect_campaign(base, {}) == {}


def test_non_map_view_retains_sampling_without_invented_measurements():
    base, obs, c, _, view = sample()
    c.kind = "volume"
    view["indices"] = [0, 5, 9]
    view["available_members"] = 10
    view["inspected_fraction"] = 0.3
    context = collect_campaign(base, {"map-check": obs})["map-check"]
    assert context["numeric"] is None
    assert context["transform"]["indices"] == [0, 5, 9]
    assert len(supplementary({"map-check": context})) == 5
    altered = obs.model_copy(update=dict(image=obs.image.model_copy(update=dict(width=99))))
    with pytest.raises(ValueError, match="observation_image_binding"):
        collect_campaign(base, {"map-check": altered})


def test_snapshot_component_entry_keeps_original_owner_boundary(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from packages.dataset_adapters.multimodal.adapters import prepare
    from packages.dataset_adapters.multimodal.tasks import load_intent
    from packages.multimodal_research import entry
    from tests.dataset_adapters.test_multimodal import gamma
    from tests.evaluation_preparation.test_component_consumer import config

    root = gamma(tmp_path / "source")
    manifest = prepare("gamma", root, "0001", selection="training").manifest
    task, _ = load_intent("advanced_glaucoma", "configs/research/multimodal_propositions_v1.json")
    path, _ = config(tmp_path, manifest=manifest, task=task)
    raw = json.loads(path.read_bytes())
    raw["target"] = "active:cup_geometry"
    path.write_text(json.dumps(raw))

    def views(command, **kwargs):
        target = Path(command[4])
        target.mkdir(parents=True)
        (target / "views.json").write_text(json.dumps(dict(views=[])))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(entry.subprocess, "run", views)
    out = tmp_path / "prepared"
    prepare_entry(
        dataset="gamma",
        root=root,
        sample="0001",
        selection="training",
        task="advanced_glaucoma",
        output=out,
        registry="configs/research/multimodal_propositions_v1.json",
        active_acquisition="configs/research/multimodal_active_requirements_v9.json",
        resilient_vision=True,
        view_profile="detail-v1",
        component_consumer=path,
        execution_profile="relay-snapshot-context-v1",
    )
    spec = json.loads((out / "entry.json").read_bytes())
    assert spec["schema_version"] == "multimodal-live-entry/1.20.0"
    assert (
        assemble_entry(out, preflight_only=True)["route"]
        == "confirmed_component_current_owner_recheck"
    )

    def owners():
        raise RuntimeError("original_owner_boundary")

    monkeypatch.setattr(entry, "owners", owners)
    with pytest.raises(RuntimeError, match="original_owner_boundary"):
        assemble_entry(out)
    spec.pop("component_consumer")
    (out / "entry.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="component_consumer_entry_version"):
        assemble_entry(out, preflight_only=True)
