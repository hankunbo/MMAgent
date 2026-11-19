"""Context reaches original review wire without changing facts, Q or votes."""

import copy
import json
import struct
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from apps.vectorengine_case_materials import ref
from packages.evidence_details.acceptance_models import DomainState
from packages.evidence_details.review_tasks import project_review_tasks
from packages.grounding.models import digest, validate
from packages.llm_provider.models import ImageBinding
from packages.multimodal_research.campaign import configuration as original_config
from packages.multimodal_research.context_review import configuration, plan_for
from packages.multimodal_research.material import observation
from packages.multimodal_research.review_context import array_summary, collect, supplementary
from packages.object_store import checksum
from tests.agent_provider.test_qualified_citation_wire import body
from tests.dataset_adapters.support import npy
from tests.evaluation_preparation.test_directional_review import fixture


def sample():
    raw = npy((2, 3), "<f8", struct.pack("<6d", -2, 0, 0, 40, 80, float("nan")))
    c = SimpleNamespace(
        id="map:r",
        concept="rnflt",
        unit="um",
        shape=(2, 3),
        kind="map",
        status="obtained",
        payload_ids=("p",),
        unit_basis="https://example.invalid/synthetic",
        semantic_limits=("unregistered",),
    )
    view = dict(
        transform=dict(
            kind="global_finite_minmax_display",
            minimum=0.0,
            maximum=80.0,
            invalid_display="magenta",
            unit="um",
            scalar_scale="linear_valid_thickness_not_normative_color_code",
        )
    )
    image = ImageBinding(
        reference=ref("image", "ASSET:TEST", digest("image")),
        media_type="image/png",
        width=3,
        height=2,
        coordinate_system="pixel_xy_top_left",
        location=(0, 0, 3, 2),
        preprocessing=ref("preprocessing", "POLICY:DISPLAY", digest(view)),
    )
    obs = observation("map-check", "original visual description", image=image)
    base = SimpleNamespace(
        read_dependency_input=lambda: None,
        ledger=[dict(id="map-check", component=c.id)],
        prepared=SimpleNamespace(payload_bytes=(("p", raw),)),
        manifest=SimpleNamespace(
            checksum=digest("input"),
            components=(c,),
            payloads=(SimpleNamespace(id="p", checksum=checksum(raw), byte_size=len(raw)),),
        ),
        view=lambda _: (image, b"image", view),
    )
    return base, obs, c, raw, view


def test_zero_negative_nonfinite_are_distinct_without_imputing_medical_validity():
    _, _, c, raw, view = sample()
    result = array_summary(raw, c, view)
    assert (result["negative"], result["zero"], result["positive"], result["nonfinite"]) == (
        1,
        2,
        2,
        1,
    )
    assert result["quantized_black_retained"] == 2
    assert result["retained_mean"] == 30 and result["positive_mean"] == 60
    assert "not_tissue_QC" in result["subset_semantics"]
    assert "e" not in result and "quality" not in result


def test_context_rebuild_exact_references_and_original_unchanged():
    base, obs, _, _, _ = sample()
    before = obs.model_dump(mode="json")
    contexts = collect(base, {"map-check": obs}, ["map-check"])
    extra = supplementary(contexts)
    assert obs.model_dump(mode="json") == before
    assert len(extra) == 5 and all(len(o.value) <= 1000 for o in extra)
    assert contexts["map-check"]["observation"] == before["reference"]
    assert all(o.method == "owner_display_context_not_disease_evidence" for o in extra)


@pytest.mark.parametrize(
    "change,error",
    [
        ("payload", "payload_changed"),
        ("image", "image_binding"),
        ("transform", "display_scale_changed"),
        ("missing", "confirmed_image"),
    ],
)
def test_context_mismatch_rejected(change, error):
    base, obs, _, raw, view = sample()
    if change == "payload":
        base.prepared.payload_bytes = (("p", raw + b"bad"),)
    elif change == "image":
        obs = obs.model_copy(update=dict(image=obs.image.model_copy(update=dict(width=4))))
    elif change == "transform":
        view["transform"]["maximum"] = 90
    else:
        obs = obs.model_copy(update=dict(status="missing"))
    with pytest.raises(ValueError, match=error):
        collect(base, {"map-check": obs}, ["map-check"])


def test_real_wire_has_context_and_per_task_table_without_extra_votes():
    base, obs, _, _, _ = sample()
    extras = supplementary(collect(base, {"map-check": obs}, ["map-check"]))
    f = fixture()
    old = f.owner.state
    raw = old.model_dump(mode="json")
    raw["material"]["observations"].extend(o.model_dump(mode="json") for o in extras)
    raw["material"]["scope"].pop("material_checksum")
    raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
    current = validate(DomainState, raw)
    assert current.policy == old.policy
    assert len(project_review_tasks(current).tasks) == len(project_review_tasks(old).tasks)
    config = configuration(original_config(directional=True, qualified_observations=True))
    wire = body(config, current)
    texts = [c["text"] for c in wire["messages"][1]["content"]]
    assert any("quantized_black_retained" in t for t in texts)
    table = json.loads(
        next(t.split("\n", 1)[1] for t in texts if t.startswith("REVIEW_CITATION_CODES:"))
    )
    assert not any("DISPLAY_CONTEXT" in str(t) for t in table["tasks"])
    assert "No minimum magnitude" in wire["messages"][0]["content"]
    assert wire["response_format"] == {"type": "json_object"}
    assert config.transport.max_retries == 0


def test_new_plan_bounds_and_old_results_immutable(monkeypatch):
    base, obs, _, _, _ = sample()
    base.task = SimpleNamespace(checksum=digest("task"))
    parent = SimpleNamespace(
        base=base,
        protocol=dict(deadline="2020-01-01T00:00:00+00:00"),
        accepted_observations=lambda: {"map-check": obs},
    )
    contexts = collect(base, {"map-check": obs}, ["map-check"])
    tip = dict(path="old", previous_attempts=[dict(requests=20)])
    before = copy.deepcopy(tip)
    monkeypatch.setattr("packages.multimodal_research.context_review.freeze", lambda: {})
    config = configuration(original_config(directional=True, qualified_observations=True))
    kwargs = dict(output="new", authorization="operator", started=datetime.now(UTC))
    plan = plan_for(parent, {}, tip, contexts, {}, config, **kwargs)
    assert tip == before
    assert plan["max_model_dispatches"] == 1 and plan["max_new_requests"] == 2
    assert (
        plan["new_vision_requests"] == 0 and plan["prior_deadline"] == parent.protocol["deadline"]
    )
    tip["previous_attempts"] = [dict(requests=39)]
    with pytest.raises(ValueError, match="scope_or_budget"):
        plan_for(parent, {}, tip, contexts, {}, config, **kwargs)


def test_supplementary_context_survives_original_acceptance_without_extra_scores():
    from tests.agent_provider.batch_support import generate

    base, obs, _, _, _ = sample()
    f = fixture()
    before = f.owner.state
    raw = before.model_dump(mode="json")
    extra = supplementary(collect(base, {"map-check": obs}, ["map-check"]))
    raw["material"]["observations"].extend(o.model_dump(mode="json") for o in extra)
    raw["material"]["scope"].pop("material_checksum")
    raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
    f.owner.state = validate(DomainState, raw)
    candidate = generate(f)
    f.acceptance.submit(candidate, key="display-context")
    assert f.acceptance.accept(key="display-context").status == "accepted"
    saved = f.acceptance.read(key="display-context")
    assert saved[2]["scores"]["evidence_counts"]["requirements"] == len(before.policy.requirements)
    assert not any("DISPLAY_CONTEXT" in r["id"] for r in saved[2]["scores"]["reviews"])
    assert candidate.observation_checksums == tuple(
        o.reference for o in f.owner.state.material.observations
    )


def test_nullable_parent_keeps_its_original_alignment_semantics():
    from packages.llm_provider.config import review_format_upgrade

    old = review_format_upgrade(original_config(directional=True, qualified_observations=True))
    before = old.model_dump(mode="json")
    new = configuration(old)
    assert old.model_dump(mode="json") == before
    assert new.schema_version == old.schema_version == "agent-provider-config/9.0.0"
    assert new.review_response_policy == old.review_response_policy
    assert new.format_correction_budget == old.format_correction_budget
    assert new.transport.max_retries == 0
    assert (
        next(r.prompt_version for r in new.roles if r.role == "review") == "review-direction-2.4.0"
    )
