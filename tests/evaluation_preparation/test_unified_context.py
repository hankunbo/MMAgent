"""Context -> original batch review -> original weighted scorer, injected transport."""

from types import SimpleNamespace

import pytest

from packages.evaluation_preparation.dynamic_stages import rebuild
from packages.evaluation_preparation.partial_observation import schedule
from packages.evaluation_preparation.quality_materials import upgrade
from packages.evaluation_preparation.unified_requirements import add_context, catalog
from packages.grounding.models import digest, validate
from tests.agent_provider.batch_support import generate
from tests.evaluation_preparation.test_dynamic_evidence import dynamic_fixture


@pytest.mark.parametrize("age", [0.0, 86.0, None])
def test_context_is_independent_of_masks_and_zero_is_not_missing(monkeypatch, age):
    f = dynamic_fixture(1, empty=True)
    seed = upgrade(f.owner.current())
    raw = seed.policy.model_dump(mode="json")
    raw["requirements"][0].update(id="criterion:AGE", input_modalities=["age"])
    raw["groups"] = [
        dict(
            id="age",
            slots=[["criterion:AGE"]],
            importance=dict(slo_image_research=0.0, comprehensive_research=1.0),
        )
    ]
    seed = rebuild(seed, validate(type(seed.policy), raw), [])
    context = dict(age=dict(value=age), projection_policy="fairvlmed-public-age/1.0.0")
    base = SimpleNamespace(
        runtime=None,
        objects=None,
        visible_value=None,
        input_context=context,
        context_artifact=SimpleNamespace(asset_code="ASSET:AGE", checksum=digest("age")),
    )
    monkeypatch.setattr("packages.dataset_adapters.public_context.read", lambda *args: context)
    result = add_context(base, seed)
    if age is None:
        assert not result.policy.items
        assert schedule(result.policy.requirements)[0]["action"] == "record_input_gap"
        return
    item = result.policy.items[0]
    assert item.role == "risk" and item.quality.value == 1
    assert item.dependency_paths == (("ASSET:AGE",),)
    assert item.correlation_group == "demographic_age"
    assert (
        schedule(result.policy.requirements, measured={item.id}, modalities=("age",))[0]["action"]
        == "review_measurement"
    )
    f.owner.state = result
    submission = generate(f)
    f.acceptance.submit(submission, key="age")
    assert f.acceptance.accept(key="age").status == "accepted"
    row = f.acceptance.read(key="age")[2]["scores"]["reviews"][0]
    assert row["id"] == "criterion:AGE" and row["role"] == "risk"


def test_new_bindings_use_exact_sources_and_do_not_duplicate_visuals():
    from apps.active_evidence_materials import PILOTS
    from packages.evaluation_preparation.evidence_inventory import inventory

    value = catalog()
    selected = {k for k, v in value["tasks"].items() if v["selected_source_ids"]} | set(PILOTS)
    ledger = inventory(value, selected, mappings=value["merges"])
    assert len(selected) == 23
    assert ledger["candidate_count"] == 45
    assert sum(r["disposition"] == "merged" for r in ledger["rows"]) == 12
    assert value["tasks"]["criterion:FIELD"]["input_modalities"] == ["visual_field"]
    assert value["tasks"]["criterion:RNFL_DEFECT"]["selected_source_ids"] == ["aao_signs"]
    assert len(value["sources"]) == 14


def test_declarative_profile_can_add_requirement_but_not_unbound_source(tmp_path):
    import json

    from packages.evaluation_preparation.unified_requirements import CONFIG

    profile = json.loads(CONFIG.read_bytes())
    entry = dict(profile["bindings"]["criterion:FIELD"])
    profile["bindings"]["criterion:SYNTHETIC_NEW"] = entry
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    assert "criterion:SYNTHETIC_NEW" in catalog(path)["tasks"]
    entry["selected_source_ids"] = ["unregistered"]
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="source_or_modality"):
        catalog(path)


def test_nei_admission_is_exact_page_and_pipeline():
    from packages.knowledge_archive.approved_web import admitted

    url = "https://www.nei.nih.gov/eye-health-information/eye-conditions-and-diseases/glaucoma"
    d = SimpleNamespace(
        provider_code="SOURCE:PROVIDER:NEI:GLAUCOMA_INFO",
        provider_version="1.0.0",
        provider_record_id="NEI-GLAUCOMA",
        config_checksum="sha256:df81464a9fa107635df44713c4cc0e0b49370c9e42886e42ccdc114673d5a4c2",
    )
    request = SimpleNamespace(
        pipeline_version="1.2.0",
        document=SimpleNamespace(adapter="web_fallback", locator=SimpleNamespace(url=url)),
        bibliography=SimpleNamespace(canonical_url=url),
    )
    assert admitted(SimpleNamespace(discovery=d), request)
    request.document.locator.url = "https://www.nei.nih.gov/other"
    assert not admitted(SimpleNamespace(discovery=d), request)
    request.document.locator.url = url
    request.pipeline_version = "1.0.0"
    assert not admitted(SimpleNamespace(discovery=d), request)
