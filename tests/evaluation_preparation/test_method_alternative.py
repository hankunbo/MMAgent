"""Declared pre-dispatch alternatives retain failed qualification, not false measurements."""

import json
from types import SimpleNamespace

import pytest

from apps import fundus_tool_registry as tool
from packages.dataset_adapters.multimodal.adapters import prepare
from packages.dataset_adapters.multimodal.tasks import load_intent
from packages.multimodal_research.active_acquisition import install, structure
from packages.multimodal_research.active_requirements import load
from packages.multimodal_research.entry_preflight import LocalReadinessError, check_local_route
from packages.object_store import checksum
from tests.dataset_adapters.test_multimodal import grape
from tests.evaluation_preparation.test_active_acquisition import source_fixture


def setup(tmp_path, monkeypatch):
    source = tmp_path / "tool-module.py"
    source.write_bytes(b"updated public source")
    lock = tmp_path / "installed-tool.json"
    lock.write_text(json.dumps(dict(resources={str(source): checksum(b"old source")})))
    monkeypatch.setattr(tool, "LOCK", lock)
    monkeypatch.setattr(tool, "load_verified", lambda: pytest.fail("drifted tool cannot qualify"))
    prepared = prepare(
        "grape",
        grape(tmp_path / "source", with_images=True),
        "1:OD",
        cutoff_visit=1,
        semantic_profile="grape-v1",
    )
    profile = load("configs/research/multimodal_grape_active_requirements_v2.json")
    task = load_intent("advanced_glaucoma", "configs/research/multimodal_propositions_v1.json")[0]
    sources = source_fixture()
    output = tmp_path / "run"
    output.mkdir()

    def base():
        return SimpleNamespace(
            output=output,
            prepared=prepared,
            manifest=prepared.manifest,
            task=task,
            sources=SimpleNamespace(initial=sources, read=lambda: sources),
            resources=(),
        )

    return base, profile, lock


def test_preflight_alternative_does_not_approve_drifted_tool(tmp_path, monkeypatch):
    factory, profile, lock = setup(tmp_path, monkeypatch)
    base = factory()
    old = dict(profile)
    old.pop("native_unavailable_policy")
    with pytest.raises(LocalReadinessError, match="fundus_installation_resource_changed"):
        check_local_route(base.manifest, base.task, old, None)
    assert (
        check_local_route(base.manifest, base.task, profile, None)
        == "vision_alternative_native_unavailable"
    )
    # I/O and malformed installation are technical errors, not a qualified fallback.
    lock.write_text("not JSON")
    with pytest.raises(json.JSONDecodeError):
        check_local_route(base.manifest, base.task, profile, None)


def test_install_cold_read_and_report_keep_explicit_alternative(tmp_path, monkeypatch):
    from packages.multimodal_research.campaign import MultimodalCampaign

    factory, profile, _ = setup(tmp_path, monkeypatch)
    base = factory()

    def forbidden(*a, **kw):
        pytest.fail("unqualified native tool must not dispatch")

    install(base, profile, native_runner=forbidden)
    methods = base.active_owner.methods
    assert methods["readiness"]["native_qualified"] is False
    assert methods["readiness"]["reason"] == "fundus_installation_resource_changed"
    ids = {t["id"] for t in methods["targets"]}
    assert len(ids) == 2 and not ids.intersection(base.observations)
    assert base.active_owner.native is None
    selected = MultimodalCampaign.select_targets(base, {"visit:0:fundus", "visit:1:fundus"})
    assert ids <= set(selected)
    cold = factory()
    install(cold, profile, native_runner=forbidden, read_only=True)
    assert cold.active_owner.report() == base.active_owner.report()
    report = dict(
        requirements=[
            dict(id=r["id"], execution_binding={}, presentation_sections={}) for r in base.ledger
        ],
        report_production={},
        scores=None,
    )
    structure(report, cold)
    for r in report["requirements"]:
        if r["id"] in ids:
            route = r["execution_binding"]["method_selection"]
            assert route["selected_method"] == "vision" and route["native_qualification"] is False
            assert "native_record" not in r["execution_binding"]
    assert report["scores"] is None
    path = base.output / "active-method-selection.json"
    changed = json.loads(path.read_bytes())
    changed["readiness"]["reason"] = "changed"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="active_method_selection_changed"):
        cold.active_owner.check()


def test_existing_native_work_cannot_be_reinterpreted_as_alternative(tmp_path, monkeypatch):
    factory, profile, _ = setup(tmp_path, monkeypatch)
    base = factory()
    (base.output / "native-fundus").mkdir()
    with pytest.raises(ValueError, match="active_native_prior_execution_requires_confirmation"):
        install(base, profile, native_runner=lambda *a, **kw: pytest.fail("must not dispatch"))
    assert not (base.output / "active-method-selection.json").exists()


def test_successor_inventory_keeps_original_visual_alternative(tmp_path, monkeypatch):
    from packages.multimodal_research.campaign_renewal import unattempted_targets

    factory, profile, _ = setup(tmp_path, monkeypatch)
    base = factory()
    install(base, profile, native_runner=lambda *a, **kw: pytest.fail("native not qualified"))
    base.views = dict(
        views=[
            dict(component="visit:0:fundus", status="obtained"),
            dict(component="visit:1:fundus", status="obtained"),
        ]
    )
    parent = SimpleNamespace(base=base, initial_targets=())
    ids = [t["id"] for t in base.active_owner.methods["targets"]]
    options = dict(full_inventory=True, owner_methods=True)
    assert set(ids) <= set(unattempted_targets(parent, set(), {}, **options))
    assert ids[0] not in unattempted_targets(parent, {ids[0]}, {}, **options)
    assert ids[1] not in unattempted_targets(parent, set(), {ids[1]: object()}, **options)
    base.views["views"][0]["status"] = "missing"
    assert ids[0] not in unattempted_targets(parent, set(), {}, **options)
    base.known_only = True
    assert not unattempted_targets(parent, set(), {}, **options)
    base.known_only = False
    path = base.output / "active-method-selection.json"
    raw = json.loads(path.read_bytes())
    raw["readiness"]["method"] = "native"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="active_method_selection_changed"):
        unattempted_targets(parent, set(), {}, **options)
