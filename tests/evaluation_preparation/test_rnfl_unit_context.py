"""Public-source bridge only; no clinical direction or patient reassessment."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from packages.dataset_adapters.multimodal import grape_semantics as original

candidate = original
EXPECTED_NAME = "cirrus_rnfl_report_guide_31_011_0002ii.pdf"


@pytest.fixture
def source():
    return Path("docs/sources") / EXPECTED_NAME


@pytest.mark.parametrize(
    "concept",
    ["rnfl_mean", "rnfl_superior", "rnfl_nasal", "rnfl_inferior", "rnfl_temporal", "iop", "cct"],
)
def test_original_profile_exact_without_new_source(concept):
    expected = json.loads(Path("tests/fixtures/grape_method_legacy_v1.json").read_text())[concept]
    assert candidate.method_context(concept) == expected["method"]
    assert json.loads(json.dumps(candidate.fields(concept))) == expected["fields"]
    assert json.loads(json.dumps(candidate.context(concept))) == expected["context"]


@pytest.mark.parametrize(
    "concept", ["rnfl_mean", "rnfl_superior", "rnfl_nasal", "rnfl_inferior", "rnfl_temporal"]
)
def test_new_profile_uses_both_actual_public_sources_without_rewriting_release(source, concept):
    before = deepcopy(original.method_context(concept))
    value = candidate.method_context(concept, profile=candidate.RNFL_UNIT_PROFILE)
    bridge = value["unit_bridge"]
    assert value["source_checksum"] == before["source_checksum"]
    assert value["excerpt"] == before["excerpt"]
    assert "CIRRUS HD-OCT 5000" in value["excerpt"]
    assert bridge["exact_unit_text"] == "µm" and bridge["proposed_unit"] == "um"
    assert bridge["locator"]["page_1_based"] == 12
    assert (
        value["unit_qualification"]
        == before["unit_qualification"]
        == "release_unit_remains_unverified"
    )
    assert (
        bridge["relation_kind"]
        == "documented_device_protocol_inference_not_release_field_assertion"
    )
    assert bridge["source_measurement_changed"] is bridge["clinical_qualification"] is False
    assert not ({"e", "threshold", "normative_percentile", "score"} & bridge.keys())
    assert candidate.method_context(concept) == before


@pytest.mark.parametrize("concept", ["iop", "cct", "vf", "age", "rnfl_unknown"])
def test_non_rnfl_cannot_inherit_manufacturer_unit(concept):
    with pytest.raises(ValueError, match="grape_rnfl_unit_concept"):
        candidate.method_context(concept, profile=candidate.RNFL_UNIT_PROFILE)


@pytest.mark.parametrize("fault", ["pdf", "paper", "symlink", "profile"])
def test_changed_source_or_profile_rejected(source, monkeypatch, fault):
    read = Path.read_bytes
    if fault in {"pdf", "paper"}:
        name = EXPECTED_NAME if fault == "pdf" else "grape-author-2023-PMC10404253.xml"
        monkeypatch.setattr(Path, "read_bytes", lambda p: b"changed" if p.name == name else read(p))
    if fault == "symlink":
        old = Path.is_symlink
        monkeypatch.setattr(
            Path, "is_symlink", lambda p: True if p.name == EXPECTED_NAME else old(p)
        )
    error = {
        "pdf": "grape_rnfl_manufacturer_source_binding",
        "paper": "grape_method_source_binding",
        "symlink": "grape_rnfl_manufacturer_source_binding",
        "profile": "grape_method_profile",
    }[fault]
    with pytest.raises(ValueError, match=error):
        candidate.method_context(
            "rnfl_mean", profile="unknown" if fault == "profile" else candidate.RNFL_UNIT_PROFILE
        )


def staged_review_context(monkeypatch):
    from packages.multimodal_research import review_context

    return review_context


def test_original_component_projection_and_provider_wire_get_explicit_inference(
    source, monkeypatch
):
    from packages.evidence_details.acceptance_models import DomainState
    from packages.evidence_details.review_tasks import project_review_tasks
    from packages.grounding.models import digest, validate
    from packages.multimodal_research import context_review
    from packages.multimodal_research.campaign import configuration
    from tests.agent_provider.batch_support import generate
    from tests.agent_provider.test_qualified_citation_wire import body
    from tests.evaluation_preparation.test_directional_review import fixture
    from tests.evaluation_preparation.test_grape_method_context import sample

    module = staged_review_context(monkeypatch)
    base, target, _ = sample()
    before = base.observations[target].model_dump(mode="json")
    contexts = module.collect(base, {}, [target], metadata_profile=candidate.RNFL_UNIT_PROFILE)
    assert contexts[target]["numeric"]["unit"] == "source_units_unqualified"
    assert base.observations[target].model_dump(mode="json") == before
    f = fixture()
    old = f.owner.state
    raw = old.model_dump(mode="json")
    raw["material"]["observations"].extend(
        o.model_dump(mode="json") for o in module.supplementary(contexts)
    )
    raw["material"]["scope"].pop("material_checksum")
    raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
    state = validate(DomainState, raw)
    assert state.policy == old.policy
    assert len(project_review_tasks(state).tasks) == len(project_review_tasks(old).tasks)
    config = context_review.configuration(
        configuration(directional=True, qualified_observations=True)
    )
    wire = str(body(config, state))
    assert "documented_device_protocol_inference_not_release_field_assertion" in wire
    assert "source_units_unqualified" in wire
    assert "cirrus_how_to_read" in wire
    f.owner.state = state
    f.acceptance.submit(generate(f), key="unit-bridge-material-only")
    assert f.acceptance.accept(key="unit-bridge-material-only").status == "accepted"
    scores = f.acceptance.read(key="unit-bridge-material-only")[2]["scores"]
    assert scores["evidence_counts"]["requirements"] == len(old.policy.requirements)
    assert not any("DISPLAY_CONTEXT" in r["id"] for r in scores["reviews"])


@pytest.mark.parametrize("fault", ["value", "scope"])
def test_bridge_cannot_fix_changed_patient_value_or_scope(source, monkeypatch, fault):
    from tests.evaluation_preparation.test_grape_method_context import sample

    module = staged_review_context(monkeypatch)
    base, target, component = sample()
    if fault == "value":
        component.value = 10000
    else:
        base.manifest.dataset = "gamma"
    with pytest.raises(ValueError, match="review_method_context"):
        module.collect(base, {}, [target], metadata_profile=candidate.RNFL_UNIT_PROFILE)
