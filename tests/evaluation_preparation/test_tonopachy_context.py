"""Versioned unit metadata through real projection, wire and domain acceptance."""

import copy
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from packages.dataset_adapters.multimodal import grape_semantics as g
from packages.multimodal_research import context_review as c
from packages.multimodal_research import review_context as r
from tests.evaluation_preparation.test_grape_method_context import sample
from tests.evaluation_preparation.test_null_review_context import persisted_origin, planning


@pytest.mark.parametrize("concept,unit", [("iop", "mmHg"), ("cct", "um")])
def test_original_author_and_device_units_project_without_rewriting_observation(concept, unit):
    base, target, component = sample(concept)
    old = copy.deepcopy(base.observations[target])
    context = r.collect(base, {}, [target], metadata_profile=g.TONOPACHY_UNIT_PROFILE)[target]
    method = context["author_method"]
    bridge = method["unit_bridge"]
    assert method["schema_version"] == g.TONOPACHY_UNIT_PROFILE
    assert "NT-530P" in method["excerpt"]
    assert bridge["proposed_unit"] == unit
    assert bridge["source_measurement_changed"] is False
    assert bridge["iop_compensation_status"] == "unknown_in_dataset_release"
    assert bridge["locator"]["page_1_based"] == 4
    assert context["numeric"]["unit"] == "source_units_unqualified"
    assert base.observations[target] == old and component.unit is None
    assert "e" not in bridge and "threshold" not in bridge


@pytest.mark.parametrize("concept", ["age", "vf", "rnfl_mean"])
def test_other_measurement_cannot_inherit_tonopachy_unit(concept):
    with pytest.raises(ValueError, match="grape_tonopachy_unit_concept"):
        g.method_context(concept, profile=g.TONOPACHY_UNIT_PROFILE)


@pytest.mark.parametrize("fault", ["value", "missing", "dataset", "pdf", "symlink"])
def test_current_owner_and_material_rejection(fault, monkeypatch):
    base, target, component = sample("iop")
    if fault == "value":
        component.value = 99
    elif fault == "missing":
        component.status = "missing"
    elif fault == "dataset":
        base.manifest.dataset = "gamma"
    elif fault == "pdf":
        read = Path.read_bytes
        monkeypatch.setattr(
            Path,
            "read_bytes",
            lambda p: b"changed" if p.name == "nidek_tonopachy_nt530p.pdf" else read(p),
        )
    else:
        link = Path.is_symlink
        monkeypatch.setattr(
            Path,
            "is_symlink",
            lambda p: True if p.name == "nidek_tonopachy_nt530p.pdf" else link(p),
        )
    with pytest.raises(ValueError, match="review_method_context|grape_tonopachy_manufacturer"):
        r.collect(base, {}, [target], metadata_profile=g.TONOPACHY_UNIT_PROFILE)


def test_exact_legacy_profile():
    legacy = json.loads(Path("tests/fixtures/grape_method_legacy_v1.json").read_text())
    for concept in ("iop", "cct"):
        assert g.method_context(concept) == legacy[concept]["method"]
        assert json.loads(json.dumps(g.fields(concept))) == legacy[concept]["fields"]


def test_real_wire_acceptance_adds_material_not_votes():
    from packages.evidence_details.acceptance_models import DomainState
    from packages.evidence_details.review_tasks import project_review_tasks
    from packages.grounding.models import digest, validate
    from packages.multimodal_research.campaign import configuration
    from tests.agent_provider.batch_support import generate
    from tests.agent_provider.test_qualified_citation_wire import body
    from tests.evaluation_preparation.test_directional_review import fixture

    base, target, _ = sample("iop")
    contexts = r.collect(base, {}, [target], metadata_profile=g.TONOPACHY_UNIT_PROFILE)
    f = fixture()
    old = f.owner.state
    raw = old.model_dump(mode="json")
    raw["material"]["observations"].extend(
        o.model_dump(mode="json") for o in r.supplementary(contexts)
    )
    raw["material"]["scope"].pop("material_checksum")
    raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
    current = validate(DomainState, raw)
    assert current.policy == old.policy
    assert len(project_review_tasks(current).tasks) == len(project_review_tasks(old).tasks)
    config = c.configuration(configuration(directional=True, qualified_observations=True))
    wire = str(body(config, current))
    for text in ("NT-530P", "mmHg", "unknown_in_dataset_release", "source_units_unqualified"):
        assert text in wire
    f.owner.state = current
    f.acceptance.submit(generate(f), key="tonopachy-context-only")
    assert f.acceptance.accept(key="tonopachy-context-only").status == "accepted"
    scores = f.acceptance.read(key="tonopachy-context-only")[2]["scores"]
    assert scores["evidence_counts"]["requirements"] == len(old.policy.requirements)
    assert not any("DISPLAY_CONTEXT" in row["id"] for row in scores["reviews"])


def test_new_aggregate_profile_requires_current_null_and_preserves_budget(monkeypatch):
    f = planning(count=2, prior=28)
    f[0].aggregation_baseline = object()
    monkeypatch.setattr(c, "freeze", lambda: {})
    args = dict(
        output="new",
        authorization="synthetic",
        started=datetime.now(UTC),
        metadata_profile=g.TONOPACHY_UNIT_PROFILE,
    )
    p = c.plan_for(f[0], {}, f[1], f[2], f[3], f[4], **args)
    assert p["schema_version"] == "multimodal-campaign-renewal/1.72.0"
    assert p["max_new_requests"] == 2 and p["max_model_dispatches"] == 1
    assert p["previous_attempts"] == f[1]["previous_attempts"]
    assert p["reuse_assessment_ids"] == [] and p["new_vision_requests"] == 0
    f[3]["null_ids"] = []
    with pytest.raises(ValueError, match="only_accepted_null"):
        c.plan_for(f[0], {}, f[1], f[2], f[3], f[4], **args)


def test_current_original_aggregate_proof_and_old_profile_version():
    f, parent, result, tip = persisted_origin()
    proof = c.origin_proof(parent, result, tip, metadata_profile=g.TONOPACHY_UNIT_PROFILE)
    assert f.target in proof["null_ids"]
    assert c.method_version(g.RNFL_UNIT_PROFILE) == "multimodal-campaign-renewal/1.69.0"
    assert c.method_version(g.TONOPACHY_UNIT_PROFILE) == "multimodal-campaign-renewal/1.72.0"


def test_original_cli_exposes_explicit_new_profile(monkeypatch, capsys):
    from apps.multimodal_case import main

    calls = []
    monkeypatch.setattr(c, "prepare", lambda **kw: calls.append(kw) or {"new_api": 0})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "case",
            "context-review-prepare",
            "--previous",
            "old",
            "--output",
            "new",
            "--assessment-id",
            "check:iop",
            "--authorization",
            "synthetic",
            "--metadata-profile",
            g.TONOPACHY_UNIT_PROFILE,
        ],
    )
    assert main() == 0
    assert calls[0]["metadata_profile"] == g.TONOPACHY_UNIT_PROFILE
    assert '"new_api": 0' in capsys.readouterr().out


def test_single_review_parent_does_not_require_invented_aggregation(monkeypatch):
    f = planning(count=2, prior=24)
    monkeypatch.setattr(c, "freeze", lambda: {})
    monkeypatch.setattr(c, "method_reuse_ids", lambda *args: ["old-reviewed"])
    p = c.plan_for(
        f[0],
        {},
        f[1],
        f[2],
        {},
        f[4],
        output="new",
        authorization="synthetic",
        started=datetime.now(UTC),
        metadata_profile=g.TONOPACHY_UNIT_PROFILE,
    )
    assert p["schema_version"] == c.TONOPACHY_METHOD_VERSION
    assert p["reuse_assessment_ids"] == ["old-reviewed"]
    assert p["review_target_ids"] == sorted([*f[2], "old-reviewed"])
    assert p["max_new_requests"] == 3 and p["seconds"] == 2700
    assert not c.aggregate_profile(f[0], g.TONOPACHY_UNIT_PROFILE)


def test_public_renewal_assembly_routes_new_version_to_original_parent_owner(tmp_path, monkeypatch):
    from packages.multimodal_research import campaign_renewal

    class OriginalParentReached(Exception):
        pass

    def parent(*args, **kwargs):
        raise OriginalParentReached()

    monkeypatch.setattr(c, "completed_parent", parent)
    (tmp_path / "protocol.json").write_text(
        json.dumps(
            dict(
                schema_version=c.TONOPACHY_METHOD_VERSION,
                metadata_profile=g.TONOPACHY_UNIT_PROFILE,
                output=str(tmp_path.absolute()),
                facts="existing-parent",
            )
        )
    )
    with pytest.raises(OriginalParentReached):
        campaign_renewal.assemble(tmp_path)
