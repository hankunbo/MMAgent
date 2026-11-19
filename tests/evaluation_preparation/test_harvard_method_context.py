"""Pinned author metadata corrects context, not values/units or medical scores."""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.dataset_adapters.multimodal import harvard_semantics as author
from packages.dataset_adapters.multimodal.adapters import prepare
from packages.grounding.models import digest
from packages.multimodal_research import context_review
from packages.multimodal_research.campaign import configuration
from packages.multimodal_research.material import numeric, observation
from packages.multimodal_research.review_context import collect, supplementary
from tests.dataset_adapters.test_multimodal import gdp


def sample(tmp_path):
    prepared = prepare(
        "harvard_gdp", gdp(tmp_path / "source"), "data_0001.npz", semantic_profile="harvard-gdp-v1"
    )
    component = next(c for c in prepared.manifest.components if c.concept == "rnflt")
    target = "check:" + component.id
    measured = numeric(component, dict(prepared.payload_bytes))
    base = SimpleNamespace(
        manifest=prepared.manifest,
        prepared=prepared,
        observations={target: observation(target, measured, unit="um")},
        ledger=[dict(id=target, component=component.id)],
        read_dependency_input=lambda: None,
        task=SimpleNamespace(checksum=digest("synthetic-target")),
    )
    return base, target


def test_exact_author_pdf_reaches_context_without_shape_conversion(tmp_path):
    base, target = sample(tmp_path)
    original = digest(base.observations)
    context = collect(base, {}, [target], metadata_profile=author.METHOD_PROFILE)[target]
    assert context["numeric"]["mean"] == 80 and context["numeric"]["unit"] == "um"
    assert context["author_method"]["paper_shape"] == [224, 224]
    assert context["author_method"]["release_readme"]["shape"] == [225, 225]
    assert context["numeric"]["shape"] == [225, 225]
    assert "peripapillary" in context["author_method"]["excerpt"]
    assert context["author_method"]["source_checksum"] == author.METHOD_CHECKSUM
    extras = supplementary({target: context})
    assert all(o.method == "owner_display_context_not_disease_evidence" for o in extras)
    assert all("99123" not in o.value for o in extras)  # hidden synthetic diagnosis never read
    assert digest(base.observations) == original


@pytest.mark.parametrize("change", ["source", "observation", "payload", "dataset", "count"])
def test_current_owner_and_source_binding(tmp_path, monkeypatch, change):
    base, target = sample(tmp_path)
    selected = [target]
    if change == "source":
        read = Path.read_bytes
        monkeypatch.setattr(
            Path, "read_bytes", lambda p: b"changed" if str(p) == author.METHOD_SOURCE else read(p)
        )
    elif change == "observation":
        base.observations[target] = base.observations[target].model_copy(
            update={"value": "changed"}
        )
    elif change == "payload":
        raw = list(base.prepared.payload_bytes)
        raw = [
            (k, b"changed" if k in base.manifest.components[0].payload_ids else v) for k, v in raw
        ]
        component = next(c for c in base.manifest.components if c.concept == "rnflt")
        raw = [(k, b"changed" if k in component.payload_ids else v) for k, v in raw]
        base.prepared = SimpleNamespace(payload_bytes=raw)
    elif change == "dataset":
        base.manifest = base.manifest.model_copy(update={"dataset": "gamma"})
    else:
        selected += [target]
    with pytest.raises(ValueError, match="method_(context|source)"):
        collect(base, {}, selected, metadata_profile=author.METHOD_PROFILE)


def test_new_version_freezes_context_scope_and_cumulative_budget(tmp_path, monkeypatch):
    base, target = sample(tmp_path)
    contexts = collect(base, {}, [target], metadata_profile=author.METHOD_PROFILE)
    config = configuration(directional=True, qualified_observations=True)
    parent = SimpleNamespace(
        base=base, config=config, protocol={"deadline": "old"}, accepted_observations=lambda: {}
    )
    tip = dict(path="parent", previous_attempts=[dict(requests=12)])
    monkeypatch.setattr(context_review, "freeze", lambda: {})
    monkeypatch.setattr(context_review, "method_reuse_ids", lambda *a: ["old-confirmed"])
    plan = context_review.plan_for(
        parent,
        {},
        tip,
        contexts,
        {},
        config,
        output="new",
        authorization="test",
        started=datetime.now(UTC),
        metadata_profile=author.METHOD_PROFILE,
    )
    assert plan["schema_version"] == context_review.HARVARD_METHOD_VERSION
    assert plan["review_context_scope"] == "current-owner-prior-visible/1.0.0"
    assert plan["previous_attempts"] == tip["previous_attempts"]
    assert plan["provider"] == digest(config) and plan["max_new_requests"] == 2
    assert plan["reassessment_ids"] == [target] and plan["reuse_assessment_ids"] == [
        "old-confirmed"
    ]
    assert plan["new_vision_requests"] == 0 and not plan["new_public_search_allowed"]
