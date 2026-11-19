"""Original multimodal projection/phase ports; no API or patient tools."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.dataset_adapters.multimodal.adapters import prepare
from packages.dataset_adapters.multimodal.tasks import load_intent
from packages.dataset_adapters.reading import grayscale_image
from packages.grounding.models import digest
from packages.multimodal_research.active_acquisition import install
from packages.multimodal_research.active_requirements import load
from packages.multimodal_research.campaign import (
    InputEvidence,
    MultimodalCampaign,
    Phase,
    configuration,
)
from packages.multimodal_research.input_scope import (
    configuration as survey_configuration,
)
from packages.multimodal_research.input_scope import (
    current_entry,
    project_plan,
)
from packages.multimodal_research.input_scope import (
    install as install_scope,
)
from packages.object_store import checksum
from tests.dataset_adapters.test_multimodal import gdp
from tests.evaluation_preparation.test_active_acquisition import source_fixture
from tests.evaluation_preparation.test_dynamic_evidence import dynamic_fixture

PROFILE = "configs/research/multimodal_catalog_acquisition_v3.json"


def prepared_view(tmp_path):
    p = prepare("harvard_gdp", gdp(tmp_path / "data"), "data_0001.npz")
    out = tmp_path / "run"
    views = out / "views"
    views.mkdir(parents=True)
    c = next(c for c in p.manifest.components if c.kind == "map")
    raw = grayscale_image((8, 8), bytes(64))
    name = checksum(c.id.encode())[7:31] + ".png"
    (views / name).write_bytes(raw)
    v = dict(
        component=c.id,
        status="obtained",
        file=name,
        checksum=checksum(raw),
        width=8,
        height=8,
        modality=c.modality,
        indices=[0],
        available_members=1,
        inspected_fraction=1,
        transform=dict(kind="synthetic_display"),
        limits=["Synthetic fixture; source numeric map is not a photograph"],
    )
    (views / "views.json").write_text(
        json.dumps(dict(input_checksum=p.manifest.checksum, views=[v]))
    )
    return p, out


def setup_base(p, out, registered, inputs, runtime=None, objects=None, *, profile=None):
    sources = source_fixture()
    base = InputEvidence(
        registered=registered,
        inputs=inputs,
        runtime=runtime,
        objects=objects,
        prepared=p,
        task=load_intent("advanced_glaucoma", "configs/research/multimodal_propositions_v1.json")[
            0
        ],
        output=out,
        sources=SimpleNamespace(initial=sources, read=lambda: sources),
    )
    install(base, load(PROFILE) if profile is None else profile, defer_native=True)
    return base


def fixture(tmp_path, *, profile=None):
    p, out = prepared_view(tmp_path)
    seed = dynamic_fixture(1).owner.current()
    visible = SimpleNamespace(
        case=SimpleNamespace(
            case_code=seed.material.scope.case_code,
            laterality="UNKNOWN",
            artifacts=[
                SimpleNamespace(
                    asset_code=seed.material.scope.graph.code,
                    checksum=seed.material.scope.graph.checksum,
                )
            ],
        ),
        checksum=seed.material.scope.visible_checksum,
        components=p.manifest,
    )
    inputs = SimpleNamespace(read_visible=lambda _: visible)
    base = setup_base(p, out, SimpleNamespace(visible=visible), inputs, profile=profile)
    campaign = object.__new__(MultimodalCampaign)
    campaign.base, campaign.output = base, out
    campaign.config = configuration()
    campaign.targets = MultimodalCampaign.available_targets(base)
    install_scope(campaign)
    return base, campaign, seed


def test_original_phase_projection_does_not_add_disease_requirements(tmp_path):
    base, campaign, seed = fixture(tmp_path)
    before = digest(base.ledger)
    assert len(campaign.scope_targets) == 1
    target = campaign.scope_targets[0]
    entry = current_entry(base, target)
    assert entry["task"]["modality"] == "OCT_RNFLT"
    assert entry["task"]["representation"] == "map"
    assert entry["task"]["requested_regions"] == ["displayed_measurement_map"]
    rid = seed.material.scope.run_id
    base.contexts[rid] = SimpleNamespace(run_id=rid, request=SimpleNamespace(task=base.task))
    phase = Phase(base, target=target)
    config = survey_configuration(campaign.config, base.input_scope_plan)
    state, _ = phase.project(run_id=rid, config=config)
    assert len(state.policy.items) == 1
    assert state.policy.items[0].method == "input_scope_survey"
    assert state.material.sources[0].reference.code == "POLICY:INPUT_SCOPE_PROCEDURE"
    assert state.material.observations[0].image.reference.checksum == checksum(
        base.image(state.material.observations[0].image.reference)
    )
    assert digest(base.ledger) == before
    assert not any(r["id"].startswith("input-scope:") for r in base.ledger)
    assert campaign.phase_path(target) in campaign.paths()
    assert campaign.config != config and campaign.config.transport == config.transport


@pytest.mark.parametrize("change", ["saved", "view", "component", "prompt"])
def test_current_plan_changed_is_not_reused(tmp_path, monkeypatch, change):
    base, campaign, _ = fixture(tmp_path)
    target = campaign.scope_targets[0]
    if change == "saved":
        path = base.output / "input-scope-plan.json"
        raw = json.loads(path.read_bytes())
        raw["entries"][0]["task"]["requested_regions"] = ["other"]
        path.write_text(json.dumps(raw))
    elif change == "view":
        base.views["views"][0]["indices"] = [99]
    elif change == "component":
        base.prepared = SimpleNamespace(manifest=None)
    else:
        original = Path.read_text
        monkeypatch.setattr(
            Path,
            "read_text",
            lambda self, *a, **kw: (
                original(self, *a, **kw)
                + ("changed" if self.name == "vision-input-scope-1.0.0.txt" else "")
            ),
        )
    with pytest.raises(
        ValueError, match="input_scope_plan_changed|active_capability_prepared_binding"
    ):
        current_entry(base, target)


def test_scope_config_restored_on_refusal_before_dispatch(tmp_path):
    base, campaign, _ = fixture(tmp_path)
    original = campaign.config
    campaign.protocol = {}
    with pytest.raises(ValueError, match="input_scope_campaign_not_frozen"):
        campaign.execute_phase(campaign.scope_targets[0])
    assert campaign.config is original and campaign._acquisition_target is None


def test_plan_is_original_input_bound_without_disease_or_dataset_selection(tmp_path):
    base, campaign, _ = fixture(tmp_path)
    assert project_plan(base, base.input_scope_plan["policy"]) == base.input_scope_plan
    assert base.input_scope_plan["disease_votes"] == 0
    assert "dataset" not in base.input_scope_plan
    assert not any("advanced" in str(e["task"]) for e in base.input_scope_plan["entries"])
    assert len(base.input_scope_plan["gaps"]) >= 1


def test_scope_policy_schema_and_current_reference():
    from packages.multimodal_research.input_scope import RegionQuestions, ScopePolicy, read_policy

    for model in (RegionQuestions, ScopePolicy):
        path = Path("packages/multimodal_research/schemas", model.__name__ + ".json")
        assert json.loads(path.read_bytes()) == model.model_json_schema()
    profile = load(PROFILE)
    assert read_policy(profile["input_scope_policy"]).max_components == 4
    wrong = dict(profile["input_scope_policy"], checksum=digest("wrong"))
    with pytest.raises(ValueError, match="input_scope_policy_changed"):
        read_policy(wrong)


def test_staged_source_command_runs_intake_before_expansion(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from packages.multimodal_research import input_scope, stages
    from tests.evaluation_preparation.test_stage_r01 import campaign_at

    c = campaign_at(tmp_path, monkeypatch, 600)
    c.protocol.update(
        schema_version="multimodal-live-campaign/1.20.0",
        entry_execution_lock="multimodal-runtime-entry/1.0.0",
    )
    (tmp_path / "protocol.json").write_text(json.dumps(c.protocol))
    # This fixture tests stage ordering with the selected original lease port;
    # actual Runtime persistence/phase consumption is covered by the PG test.
    (tmp_path / "stage-plan.json").unlink()
    stages.prepare(tmp_path)
    c.scope_targets = ("input-scope:synthetic",)
    c.definition_expansion = SimpleNamespace(documents=SimpleNamespace(run_id="SYNTHETIC"))
    calls = []

    @contextmanager
    def session():
        calls.append("lease_enter")
        try:
            yield
        finally:
            calls.append("lease_exit")

    c.execution_session = session
    monkeypatch.setattr(input_scope, "execute", lambda owner: calls.append("scope"))
    c.expand_definitions = lambda **kw: calls.append(kw["until"]) or {"complete": True}
    result = stages.run(c, "sources")
    assert result["stage"] == "sources"
    assert calls == ["lease_enter", "scope", "sources", "lease_exit"]


def test_scope_report_links_same_component_without_changing_scores(tmp_path, monkeypatch):
    from packages.multimodal_research import input_scope

    base, campaign, _ = fixture(tmp_path)
    entry = base.input_scope_plan["entries"][0]
    record = dict(
        task=entry["task"],
        records_checksum=digest("saved"),
        observation=dict(reference={"code": "OBS:SCOPE:SYNTHETIC"}),
        regions=[dict(region="displayed_measurement_map", visibility="partly_visible")],
    )
    monkeypatch.setattr(input_scope, "read_results", lambda *a, **kw: [record])
    checks = [r for r in base.ledger if r["component"] == entry["task"]["component_id"]]
    assert checks
    report = dict(
        scores={"e": 0.0},
        requirements=[
            dict(id=checks[0]["id"], presentation_sections=dict(acquisition={}, dependencies={}))
        ],
    )
    result = dict(input_scopes=[record], input_scope_plan_checksum=digest(base.input_scope_plan))
    input_scope.attach_report(campaign, report, result)
    assert report["scores"] == {"e": 0.0} and len(report["requirements"]) == 1
    context = report["requirements"][0]["presentation_sections"]["acquisition"][
        "input_scope_context"
    ]
    assert context["records_checksum"] == digest("saved")
    assert report["input_scope_assessment"]["disease_votes"] == 0
    with pytest.raises(ValueError, match="input_scope_report_result_binding"):
        input_scope.attach_report(
            campaign, report, dict(result, input_scope_plan_checksum=digest("other"))
        )
