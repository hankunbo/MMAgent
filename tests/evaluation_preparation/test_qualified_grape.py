"""GRAPE source codes must not masquerade as signed clinical measurements."""

import json

from packages.dataset_adapters.multimodal.adapters import prepare
from packages.dataset_adapters.multimodal.models import Component
from packages.multimodal_research.material import numeric
from tests.dataset_adapters.test_multimodal import grape


def test_grape_semantics_opt_in_preserves_old_payloads_and_recheck(tmp_path):
    root = grape(tmp_path)
    old = prepare("grape", root, "1:OD", cutoff_visit=1)
    new = prepare("grape", root, "1:OD", cutoff_visit=1, semantic_profile="grape-v1")
    assert old == prepare("grape", root, "1:OD", cutoff_visit=1)
    assert old.payload_bytes == new.payload_bytes
    assert old.manifest.checksum != new.manifest.checksum
    assert new.manifest.adapter_version == "multimodal-adapters/1.2.0"
    assert new.recheck() == new
    assert "98765" not in new.manifest.model_dump_json()
    vf = next(c for c in new.manifest.components if c.id == "visit:0:vf")
    assert "Octopus_G1_light_sensitivity_61_not_Humphrey_TD52" in vf.semantic_limits


def test_documented_grape_codes_do_not_enter_sensitivity_summary():
    from packages.dataset_adapters.multimodal.grape_semantics import fields

    values = [-1, -1, -3, None, 0, *([20] * 56)]
    component = Component(
        id="visit:0:vf",
        kind="vector",
        concept="vf",
        modality="VF",
        status="partial",
        reason="missing_or_nonfinite_points",
        payload_ids=("vf",),
        source_locator="VF_and_clinical_information.xlsx#Baseline!3",
        shape=(61,),
        **fields("vf"),
    )
    observed = numeric(component, {"vf": json.dumps({"values": values}).encode()})
    assert observed["coded_unseen_or_blind_count"] == 2
    assert observed["invalid_negative_count"] == 1
    assert observed["missing_count"] == 1
    assert observed["valid_count"] == 57
    assert observed["minimum"] == 0
    assert observed["mean"] == 1120 / 57
    assert observed["values"] == values
    assert observed.get("md") is None and observed.get("e") is None


def test_missing_vf_retains_definition_semantics_without_numeric_observation(tmp_path):
    prepared = prepare(
        "grape",
        grape(tmp_path, vf_missing=True),
        "1:OD",
        cutoff_visit=1,
        semantic_profile="grape-v1",
    )
    components = [c for c in prepared.manifest.components if c.concept == "vf"]
    assert len(components) == 2
    for component in components:
        assert component.status == "missing" and not component.payload_ids
        assert "Octopus_G1_light_sensitivity_61_not_Humphrey_TD52" in component.semantic_limits
        assert numeric(component, dict(prepared.payload_bytes)) is None


def test_all_coded_values_preserve_scope_without_invented_zero():
    from packages.dataset_adapters.multimodal.grape_semantics import fields

    component = Component(
        id="visit:0:vf",
        kind="vector",
        concept="vf",
        modality="VF",
        status="obtained",
        payload_ids=("vf",),
        source_locator="VF_and_clinical_information.xlsx#Baseline!3",
        shape=(61,),
        **fields("vf"),
    )
    observed = numeric(component, {"vf": json.dumps({"values": [-1] * 61}).encode()})
    assert observed["valid_count"] == 0
    assert observed["coded_unseen_or_blind_count"] == 61
    assert observed["mean"] is None
    assert observed["values"] == [-1] * 61


def test_bounded_chunks_keep_grape_coding_without_new_definitions(tmp_path):
    from types import SimpleNamespace

    from packages.multimodal_research.numeric_detail import expand

    prepared = prepare(
        "grape", grape(tmp_path), "1:OD", cutoff_visit=1, semantic_profile="grape-v1"
    )
    component = next(c for c in prepared.manifest.components if c.id == "visit:0:vf")
    numbers = [-1, None, -3, 0, *([20] * 57)]
    definition = object()
    row = dict(id="vf", component=component.id, definition=definition)
    rows, observations = expand(
        [row],
        {},
        SimpleNamespace(components=[component]),
        {component.payload_ids[0]: json.dumps({"values": numbers}).encode()},
    )
    chunks = [json.loads(observations[r["id"]].value) for r in rows]
    assert [n for chunk in chunks for n in chunk["values"]] == numbers
    assert all(r["definition"] is definition for r in rows)
    assert all(chunk["semantic_profile"] == "grape-v1" for chunk in chunks)
    assert all("not_negative_measurement" in chunk["coded_meaning"] for chunk in chunks)
    assert sum(chunk["coded_count"] for chunk in chunks) == 1
    assert sum(chunk["missing_count"] for chunk in chunks) == 1
    assert sum(chunk["invalid_negative_count"] for chunk in chunks) == 1
    assert sum(chunk["valid_count"] for chunk in chunks) == 58
    assert all(len(o.value) <= 1000 for o in observations.values())


def test_paired_source_codes_do_not_create_false_change(tmp_path):
    from packages.multimodal_research.material import paired_numeric
    from tests.evaluation_preparation.test_multimodal_campaign import sources

    prepared = prepare(
        "grape", grape(tmp_path), "1:OD", cutoff_visit=1, semantic_profile="grape-v1"
    )
    components = [c for c in prepared.manifest.components if c.concept == "vf"]
    assert len(components) == 2
    payloads = dict(prepared.payload_bytes)
    before = [-1, 10, None, -3, 0, *([20] * 56)]
    after = [10, -1, 10, 10, 2, *([21] * 56)]
    for c, values in zip(components, (before, after), strict=True):
        payloads[c.payload_ids[0]] = json.dumps({"values": values}).encode()
    _, observations = paired_numeric(prepared.manifest, payloads, sources().initial)
    comparison = json.loads(observations["comparison:vf"].value)
    assert comparison["aligned_count"] == 57
    assert comparison["excluded_pair_count"] == 4
    assert comparison["aligned_indices"] == list(range(4, 61))
    assert comparison["delta_range"] == [1, 2]
    assert comparison["delta_mean"] == 58 / 57
    assert len(observations["comparison:vf"].value) <= 1000
    assert "e" not in comparison and "md" not in comparison


def test_requirement_projection_consumes_grape_meaning_not_harvard(tmp_path):
    import pytest

    from packages.dataset_adapters.multimodal.tasks import load_intent
    from packages.multimodal_research.active_requirements import load, project
    from tests.evaluation_preparation.test_active_acquisition import source_fixture

    root = grape(tmp_path)
    p = prepare("grape", root, "1:OD", cutoff_visit=1, semantic_profile="grape-v1")
    task = load_intent("advanced_glaucoma", "configs/research/multimodal_propositions_v1.json")[0]
    config = dict(
        load("configs/research/multimodal_active_requirements_v7.json"), input_semantics="grape-v1"
    )
    rows, trace = project(p.manifest, source_fixture(), task, config)
    vf = [r for r in rows if r["component"] == "visit:0:vf"]
    assert vf
    assert all("Octopus G1 61点" in r["definition"].applicability_conditions for r in vf)
    assert trace["input_checksum"] == p.manifest.checksum
    with pytest.raises(ValueError, match="active_input_semantics_binding"):
        project(
            prepare("grape", root, "1:OD", cutoff_visit=1).manifest, source_fixture(), task, config
        )
    with pytest.raises(ValueError, match="active_input_semantics_binding"):
        project(p.manifest, source_fixture(), task, dict(config, input_semantics="harvard-gdp-v1"))


def test_grape_cli_prepare_uses_explicit_version_and_no_live_owner(tmp_path, monkeypatch):
    import pytest

    from packages.multimodal_research.active_requirements import load
    from packages.multimodal_research.entry import assemble_entry, prepare_entry

    root = grape(tmp_path / "source")
    profile = tmp_path / "grape-profile.json"
    configuration = dict(
        load("configs/research/multimodal_active_requirements_v7.json"),
        input_semantics="grape-v1",
    )
    profile.write_text(json.dumps(configuration))
    output = tmp_path / "run"

    def forbidden():
        pytest.fail("local prepare/preflight must not install live owners")

    monkeypatch.setattr("packages.multimodal_research.entry.owners", forbidden)
    result = prepare_entry(
        dataset="grape",
        root=root,
        sample="1:OD",
        cutoff_visit=1,
        task="advanced_glaucoma",
        output=output,
        registry="configs/research/multimodal_propositions_v1.json",
        active_acquisition=profile,
        resilient_vision=True,
        view_profile="detail-v1",
        input_semantics="grape-v1",
        execution_profile="relay-snapshot-context-v1",
    )
    assert result["api_requests"] == result["database_writes"] == 0
    spec = json.loads((output / "entry.json").read_text())
    assert spec["schema_version"] == "multimodal-live-entry/1.23.0"
    assert assemble_entry(output, preflight_only=True)["status"] == "local_preflight_passed"
    spec["schema_version"] = "multimodal-live-entry/1.19.0"
    (output / "entry.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="qualified_input_entry_version"):
        assemble_entry(output, preflight_only=True)


def test_public_grape_profile_binds_real_semantics_and_avoids_duplicate_vf_demand(tmp_path):
    from packages.dataset_adapters.multimodal.tasks import load_intent
    from packages.multimodal_research.active_requirements import load, project
    from packages.multimodal_research.implementation import freeze
    from tests.evaluation_preparation.test_active_acquisition import source_fixture

    path = "configs/research/multimodal_grape_active_requirements_v1.json"
    profile = load(path)
    assert path in freeze()
    assert profile["input_semantics"] == "grape-v1"
    assert profile["native_component_policy"]["max_components"] == 2
    assert all(c["id"] != "functional_stage" for c in profile["checks"])
    prepared = prepare(
        "grape",
        grape(tmp_path, with_images=True),
        "1:OD",
        cutoff_visit=1,
        semantic_profile="grape-v1",
    )
    task = load_intent("advanced_glaucoma", "configs/research/multimodal_propositions_v1.json")[0]
    rows, _ = project(prepared.manifest, source_fixture(), task, profile)
    vf = [r for r in rows if r["component"] in {"visit:0:vf", "visit:1:vf"}]
    assert {r["id"] for r in vf} == {"check:visit:0:vf", "check:visit:1:vf"}
    queries = profile["definition_expansion"]["query_requirements"]
    assert set(queries["Visual field target relationship"]) == {r["id"] for r in vf}
    passage = profile["definition_expansion"]["passage_plans"]["Visual field target relationship"]
    assert "Octopus" in passage["queries"]
    assert "mean deviation" not in passage["queries"]


def test_missing_grape_fundus_retains_demand_without_launch(tmp_path):
    from types import SimpleNamespace

    import pytest

    from packages.dataset_adapters.multimodal.tasks import load_intent
    from packages.multimodal_research.active_acquisition import install
    from packages.multimodal_research.active_requirements import load
    from tests.evaluation_preparation.test_active_acquisition import source_fixture

    prepared = prepare(
        "grape", grape(tmp_path / "source"), "1:OD", cutoff_visit=1, semantic_profile="grape-v1"
    )
    output = tmp_path / "install"
    output.mkdir()
    sources = source_fixture()
    base = SimpleNamespace(
        output=output,
        prepared=prepared,
        manifest=prepared.manifest,
        task=load_intent("advanced_glaucoma", "configs/research/multimodal_propositions_v1.json")[
            0
        ],
        sources=SimpleNamespace(initial=sources, read=lambda: sources),
        resources=(),
    )
    profile = dict(
        load("configs/research/multimodal_active_requirements_v7.json"), input_semantics="grape-v1"
    )
    install(
        base,
        profile,
        native_runner=lambda *a, **kw: pytest.fail("missing image cannot run native tool"),
    )
    missing = [r for r in base.active_owner.plan["rows"] if r["method"] == "disc_cup_measurement"]
    assert missing and all(r["state"] == "input_gap" for r in missing)
    assert base.active_owner.native is None
    assert all(r["id"] not in base.observations for r in missing)


def test_per_component_native_scope_keeps_old_directory_and_rejects_duplicates(tmp_path):
    from types import SimpleNamespace

    import pytest

    from packages.multimodal_research.native_fundus import records, result_directory

    base = SimpleNamespace(output=tmp_path)
    a = dict(id="check:a", component="visit:0:fundus")
    b = dict(id="check:b", component="visit:1:fundus")
    assert result_directory(base, a) == tmp_path / "native-fundus"
    assert result_directory(base, a, component_scoped=True) != result_directory(
        base, b, component_scoped=True
    )
    first = dict(
        schema_version="native-fundus-acceptance/1.1.0",
        plan=dict(target=a["id"], component=a["component"]),
    )
    second = dict(
        schema_version="native-fundus-acceptance/1.1.0",
        plan=dict(target=b["id"], component=b["component"]),
    )
    assert records(
        dict(schema_version="native-fundus-collection/1.0.0", records=[first, second])
    ) == (first, second)
    with pytest.raises(ValueError, match="native_fundus_collection_binding"):
        records(dict(schema_version="native-fundus-collection/1.0.0", records=[first, first]))


def test_two_visits_dispatch_separately_and_only_missing_result_falls_back(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import pytest

    from apps.vectorengine_case_materials import ref
    from packages.dataset_adapters.multimodal.tasks import load_intent
    from packages.grounding.models import digest
    from packages.multimodal_research.active_acquisition import install
    from packages.multimodal_research.active_requirements import load
    from packages.multimodal_research.campaign import MultimodalCampaign
    from tests.evaluation_preparation.test_active_acquisition import source_fixture

    prepared = prepare(
        "grape",
        grape(tmp_path / "source", with_images=True),
        "1:OD",
        cutoff_visit=1,
        semantic_profile="grape-v1",
    )
    output = tmp_path / "install"
    output.mkdir()
    sources = source_fixture()
    base = SimpleNamespace(
        output=output,
        prepared=prepared,
        manifest=prepared.manifest,
        task=load_intent("advanced_glaucoma", "configs/research/multimodal_propositions_v1.json")[
            0
        ],
        sources=SimpleNamespace(initial=sources, read=lambda: sources),
        resources=(ref("resource", "ASSET:INPUT", digest("input")),),
    )
    profile = dict(
        load("configs/research/multimodal_active_requirements_v7.json"),
        input_semantics="grape-v1",
        native_component_policy=dict(
            schema_version="bounded-native-components/1.0.0", max_components=2
        ),
    )
    calls, checked = [], []

    def native(current, target, *, before_dispatch, component_scoped):
        assert component_scoped is True
        before_dispatch()
        calls.append(target["id"])
        obtained = target["component"] == "visit:0:fundus"
        artifact = dict(asset_code="ASSET:MEASURE0", checksum=digest("measure"))
        return dict(
            schema_version="native-fundus-acceptance/1.1.0",
            plan=dict(target=target["id"], component=target["component"], tool="TOOL:FIXTURE"),
            status="obtained" if obtained else "unavailable",
            measurement=dict(
                values={"vertical_cdr": 0.5},
                quality=0.6,
                coordinate_system="pixel",
                radial_rim=None,
                radial_missing_reason="partial",
            ),
            artifact=artifact,
            artifacts=[artifact],
            prediction=dict(training_overlap="unknown"),
        )

    monkeypatch.setattr(
        "packages.multimodal_research.native_fundus.recheck",
        lambda current, record: checked.append(record["plan"]["target"]),
    )
    install(base, profile, native_runner=native)
    assert len(calls) == 2 and len(set(calls)) == 2 and checked == calls
    assert base.active_owner.native["schema_version"] == "native-fundus-collection/1.0.0"
    assert calls[0] in base.observations and calls[1] not in base.observations
    selected = MultimodalCampaign.select_targets(base, {"visit:0:fundus", "visit:1:fundus"})
    assert calls[0] not in selected and calls[1] in selected
    from packages.multimodal_research.active_acquisition import structure

    report = dict(
        requirements=[
            dict(id=r["id"], execution_binding={}, presentation_sections={}) for r in base.ledger
        ],
        report_production={},
        scores={"unchanged": 0.25},
    )
    structure(report, base)
    assert report["scores"] == {"unchanged": 0.25}
    assert report["active_acquisition"]["schema_version"] == "active-acquisition-evidence/1.1.0"
    for i, target in enumerate(calls):
        bound = next(r for r in report["requirements"] if r["id"] == target)["execution_binding"]
        assert bound["native_result_pointer"] == f"/active_acquisition/native/records/{i}"
        assert bound["native_record"]["plan"]["target"] == target
    base.active_owner.native["records"].pop()
    with pytest.raises(ValueError, match="active_native_collection_inventory"):
        base.active_owner.check()
