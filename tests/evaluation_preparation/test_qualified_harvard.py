"""Published meanings reach the real CLI without weakening clinical/score gates."""

import json
import struct
import subprocess

import pytest

from packages.dataset_adapters.multimodal.adapters import prepare
from packages.dataset_adapters.multimodal.preparation import persist, verify
from packages.dataset_adapters.multimodal.tasks import load_intent
from packages.multimodal_research.active_requirements import load, project
from packages.multimodal_research.campaign import configuration
from packages.multimodal_research.entry import assemble_entry, prepare_entry
from packages.multimodal_research.material import numeric, observation
from tests.dataset_adapters.support import npy
from tests.dataset_adapters.test_multimodal import archive, gdp
from tests.evaluation_preparation.test_active_acquisition import source_fixture

PROFILE = "configs/research/multimodal_active_requirements_v7.json"


def test_semantics_opt_in_old_bytes_and_payloads_unchanged(tmp_path):
    root = gdp(tmp_path / "source")
    old = prepare("harvard_gdp", root, "data_0001.npz")
    new = prepare("harvard_gdp", root, "data_0001.npz", semantic_profile="harvard-gdp-v1")
    assert old == prepare("harvard_gdp", root, "data_0001.npz")
    assert new.recheck() == new
    assert old.manifest.checksum != new.manifest.checksum
    assert old.payload_bytes == new.payload_bytes
    components = {c.concept: c for c in new.manifest.components}
    assert components["md"].unit == components["tds"].unit == "dB"
    assert components["rnflt"].unit == "um"
    assert components["age"].unit == "years"
    assert not any("99123" in str(c) for c in new.manifest.components)
    path = persist(new, tmp_path / "prepared")
    assert verify(path) == new.manifest


def test_signed_vector_zero_and_negative_map_codes_preserved(tmp_path):
    root = gdp(tmp_path / "source", nonfinite=True)
    archive(
        root / "RNFLT/data_0001.npz",
        {
            "rnflt.npy": npy(
                (225, 225), "<f8", struct.pack("<d", -2) + struct.pack("<d", 80) * 50624
            ),
            "tds.npy": npy((52,), "<f8", struct.pack("<52d", -3, 0, *([2] * 50))),
        },
    )
    p = prepare("harvard_gdp", root, "data_0001.npz", semantic_profile="harvard-gdp-v1")
    values = {c.concept: numeric(c, dict(p.payload_bytes)) for c in p.manifest.components}
    assert values["rnflt"]["negative_count"] == 1
    assert values["rnflt"]["negative_values"] == [-2]
    assert values["rnflt"]["mean"] == 80
    assert values["rnflt"]["valid_count"] == 50624
    assert values["tds"]["values"][:2] == [-3, 0]
    for k, v in values.items():
        if v:
            assert len(observation(k, v, unit=v["unit"]).value) <= 1000


def test_actual_requirements_get_semantics_and_modality_matched_query(tmp_path):
    p = prepare("harvard_gdp", gdp(tmp_path), "data_0001.npz", semantic_profile="harvard-gdp-v1")
    task = load_intent("advanced_glaucoma", "configs/research/multimodal_propositions_v1.json")[0]
    config = load(PROFILE)
    rows, plan = project(p.manifest, source_fixture(), task, config)
    md = next(r for r in rows if r["id"] == "check:rnflt:md")
    assert "平均偏差（dB）" in md["definition"].applicability_conditions
    visual = next(r for r in rows if r["id"] == "active:rnflt_distribution")
    assert visual["component"] == "rnflt:rnflt" and visual["acquisition_method"] == "vision"
    assert visual["evidence_unit"] == "rnfl"
    assert set(
        config["definition_expansion"]["public_topic_sets"]["Visual field target relationship"]
    ) == {"visual_field"}
    assert any(r["state"] == "input_gap" for r in plan["rows"])
    legacy = prepare("harvard_gdp", tmp_path, "data_0001.npz")
    with pytest.raises(ValueError, match="active_input_semantics_binding"):
        project(legacy.manifest, source_fixture(), task, config)


def test_cli_prepare_preflight_and_tamper_no_live_owners(tmp_path, monkeypatch):
    root = gdp(tmp_path / "source")
    out = tmp_path / "run"
    prepare_entry(
        dataset="harvard_gdp",
        root=root,
        sample="data_0001.npz",
        task="advanced_glaucoma",
        output=out,
        registry="configs/research/multimodal_propositions_v1.json",
        active_acquisition=PROFILE,
        resilient_vision=True,
        view_profile="detail-v1",
        input_semantics="harvard-gdp-v1",
    )

    def forbidden():
        pytest.fail("preflight must not assemble database/API owners")

    monkeypatch.setattr("packages.multimodal_research.entry.owners", forbidden)
    result = assemble_entry(out, preflight_only=True)
    assert result["status"] == "local_preflight_passed"
    spec = json.loads((out / "entry.json").read_text())
    assert spec["schema_version"] == "multimodal-live-entry/1.17.0"
    views = json.loads((out / "views/views.json").read_text())
    assert views["schema_version"] == "multimodal-inspection-views/1.2.0"
    assert views["views"][0]["transform"]["unit"] == "um"
    del spec["input_semantics"]
    (out / "entry.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="qualified_input_entry_version"):
        assemble_entry(out, preflight_only=True)


def test_prompt_actually_installed_not_force_nonzero():
    config = configuration(
        resilient_vision=True,
        directional=True,
        exact_primary_status=True,
        qualified_observations=True,
    )
    review = next(r for r in config.roles if r.role == "review")
    assert review.prompt_version == "review-direction-2.2.0"
    assert config.transport.timeout_seconds == 300 and config.transport.max_retries == 0
    assert config.review_batching.batch_size == 1


def test_campaign_consumes_qualified_config_and_schedules_map(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from packages.multimodal_research.campaign import MultimodalCampaign

    p = prepare(
        "harvard_gdp", gdp(tmp_path / "source"), "data_0001.npz", semantic_profile="harvard-gdp-v1"
    )
    task = load_intent("advanced_glaucoma", "configs/research/multimodal_propositions_v1.json")[0]
    rows, _ = project(p.manifest, source_fixture(), task, load(PROFILE))
    base = SimpleNamespace(
        output=tmp_path,
        manifest=p.manifest,
        ledger=rows,
        inputs=None,
        definition_expansion_policy={},
        exact_primary_status=True,
        resilient_vision=True,
        active_owner=object(),
        observations={},
        views={"views": [{"component": "rnflt:rnflt", "status": "obtained"}]},
    )
    monkeypatch.setattr(
        "packages.multimodal_research.campaign.ResearchRunRepository", lambda _: None
    )
    monkeypatch.setattr("packages.multimodal_research.campaign.PlannerRepository", lambda _: None)
    monkeypatch.setattr("packages.multimodal_research.campaign.TaskRuns", lambda **_: None)
    campaign = MultimodalCampaign(base, None, campaign_id="synthetic")
    assert "active:rnflt_distribution" in campaign.targets
    assert campaign.config == configuration(
        resilient_vision=True,
        directional=True,
        exact_primary_status=True,
        qualified_observations=True,
    )


def test_public_cli_exposes_explicit_opt_in():
    result = subprocess.run(
        [
            "/opt/data/test/envs/MMAgent1/bin/python",
            "-m",
            "apps.multimodal_case",
            "live-prepare",
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0 and "--input-semantics" in result.stdout
