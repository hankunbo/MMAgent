"""Exact author metadata reaches Review, without invented units or extra votes."""

import copy
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.dataset_adapters.multimodal import grape_semantics as grape
from packages.evidence_details.acceptance_models import DomainState
from packages.evidence_details.review_tasks import project_review_tasks
from packages.grounding.models import digest, validate
from packages.multimodal_research import context_review
from packages.multimodal_research.campaign import configuration
from packages.multimodal_research.material import numeric, observation
from packages.multimodal_research.review_context import collect, supplementary
from tests.agent_provider.test_qualified_citation_wire import body
from tests.evaluation_preparation.test_directional_review import fixture


def sample(concept="rnfl_mean"):
    fields = grape.fields(concept)
    c = SimpleNamespace(
        id="visit:0:" + concept,
        concept=concept,
        kind="scalar",
        status="obtained",
        unit=None,
        unit_basis=None,
        value=87.0,
        source_locator="baseline/row:3/column:L",
        semantic_limits=fields["semantic_limits"],
    )
    target = "check:" + c.id
    obs = observation(target, numeric(c, {}))
    base = SimpleNamespace(
        read_dependency_input=lambda: None,
        manifest=SimpleNamespace(
            dataset="grape",
            adapter_version=grape.ADAPTER,
            checksum=digest("synthetic-input"),
            components=(c,),
        ),
        ledger=[dict(id=target, component=c.id)],
        observations={target: obs},
        prepared=SimpleNamespace(payload_bytes=()),
        task=SimpleNamespace(checksum=digest("task")),
    )
    return base, target, c


@pytest.mark.parametrize(
    "concept",
    [
        "rnfl_mean",
        "rnfl_superior",
        "rnfl_nasal",
        "rnfl_inferior",
        "rnfl_temporal",
        "iop",
        "cct",
    ],
)
def test_exact_release_method_and_original_scalar_unchanged(concept):
    base, target, c = sample(concept)
    before = base.observations[target].model_dump(mode="json")
    context = collect(base, {}, [target], metadata_profile=grape.METHOD_PROFILE)[target]
    assert context["numeric"]["value"] == c.value
    assert context["numeric"]["unit"] == "source_units_unqualified"
    assert context["author_method"]["source_checksum"] == grape.SOURCE_CHECKSUM
    assert context["author_method"]["unit_qualification"] == "release_unit_remains_unverified"
    assert base.observations[target].model_dump(mode="json") == before
    assert "e" not in context and "quality" not in context


@pytest.mark.parametrize(
    "change", ["profile", "dataset", "source", "missing", "value", "duplicate"]
)
def test_owner_material_change_rejected(change):
    base, target, c = sample()
    profile, ids = grape.METHOD_PROFILE, [target]
    if change == "profile":
        profile = "unqualified"
    elif change == "dataset":
        base.manifest.dataset = "gamma"
    elif change == "source":
        c.semantic_limits = ("semantic_profile:grape-v1",)
    elif change == "missing":
        c.status = "missing"
    elif change == "value":
        c.value = 99.0
    else:
        ids.append(target)
    with pytest.raises(ValueError, match="review_method_context"):
        collect(base, {}, ids, metadata_profile=profile)


def test_public_source_tamper_rejected_without_modifying_resource(monkeypatch):
    original = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda p: b"changed" if p.name == "grape-author-2023-PMC10404253.xml" else original(p),
    )
    with pytest.raises(ValueError, match="grape_method_source_binding"):
        grape.method_context("rnfl_mean")
    with pytest.raises(ValueError, match="grape_method_concept"):
        grape.method_context("age")


def test_actual_review_wire_and_acceptance_preserve_policy_counts():
    from tests.agent_provider.batch_support import generate

    base, target, _ = sample()
    contexts = collect(base, {}, [target], metadata_profile=grape.METHOD_PROFILE)
    extras = supplementary(contexts)
    f = fixture()
    old = f.owner.state
    raw = old.model_dump(mode="json")
    raw["material"]["observations"].extend(o.model_dump(mode="json") for o in extras)
    raw["material"]["scope"].pop("material_checksum")
    raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
    current = validate(DomainState, raw)
    assert current.policy == old.policy
    assert len(project_review_tasks(current).tasks) == len(project_review_tasks(old).tasks)
    config = context_review.configuration(
        configuration(directional=True, qualified_observations=True)
    )
    wire = body(config, current)
    text = str(wire)
    assert "CIRRUS" in text and "author_method" in text
    assert "source_units_unqualified" in text
    assert all(o.method == "owner_display_context_not_disease_evidence" for o in extras)
    f.owner.state = current
    candidate = generate(f)
    f.acceptance.submit(candidate, key="author-method")
    assert f.acceptance.accept(key="author-method").status == "accepted"
    scores = f.acceptance.read(key="author-method")[2]["scores"]
    assert scores["evidence_counts"]["requirements"] == len(old.policy.requirements)
    assert not any("DISPLAY_CONTEXT" in r["id"] for r in scores["reviews"])


def test_new_plan_keeps_config_cumulative_budget_and_old_version(monkeypatch):
    base, target, _ = sample()
    contexts = collect(base, {}, [target], metadata_profile=grape.METHOD_PROFILE)
    config = configuration(directional=True, qualified_observations=True)
    parent = SimpleNamespace(
        base=base, config=config, protocol=dict(deadline="old"), accepted_observations=lambda: {}
    )
    tip = dict(path="old", previous_attempts=[dict(requests=19)])
    before = copy.deepcopy(tip)
    monkeypatch.setattr(context_review, "freeze", lambda: {})
    monkeypatch.setattr(context_review, "method_reuse_ids", lambda *a: ["old-confirmed"])
    kwargs = dict(output="new", authorization="test", started=datetime.now(UTC))
    new = context_review.plan_for(
        parent, {}, tip, contexts, {}, config, metadata_profile=grape.METHOD_PROFILE, **kwargs
    )
    assert new["schema_version"] == context_review.METHOD_VERSION
    assert new["provider"] == digest(config) and new["seconds"] == 2700
    assert new["prior_deadline"] == "old" and tip == before
    assert new["max_new_requests"] == 2 and new["max_model_dispatches"] == 1
    assert new["new_vision_requests"] == 0 and new["new_public_search_allowed"] is False
    assert new["review_target_ids"] == sorted([target, "old-confirmed"])
    assert new["reassessment_ids"] == [target] and new["reuse_assessment_ids"] == ["old-confirmed"]
    legacy = context_review.plan_for(parent, {}, tip, contexts, {}, config, **kwargs)
    assert legacy["schema_version"] == context_review.VERSION and legacy["seconds"] == 1800
    assert "metadata_profile" not in legacy and "execution_lease_profile" not in legacy
    tip["previous_attempts"][0]["requests"] = 39
    with pytest.raises(ValueError, match="scope_or_budget"):
        context_review.plan_for(
            parent, {}, tip, contexts, {}, config, metadata_profile=grape.METHOD_PROFILE, **kwargs
        )


def test_cli_profile_is_explicit(monkeypatch, capsys):
    import sys

    from apps.multimodal_case import main

    calls = []
    monkeypatch.setattr(context_review, "prepare", lambda **kw: calls.append(kw) or {"requests": 0})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "case",
            "context-review-prepare",
            "--previous",
            "p",
            "--output",
            "q",
            "--assessment-id",
            "a",
            "--authorization",
            "test",
            "--metadata-profile",
            grape.METHOD_PROFILE,
        ],
    )
    assert main() == 0 and calls[0]["metadata_profile"] == grape.METHOD_PROFILE
    assert '"requests": 0' in capsys.readouterr().out


def test_new_method_reuses_only_original_assessed_ids_and_rejects_replacement():
    records = {
        "dynamic-phase-result": ({"accepted": True, "target": None, "acceptance_key": "a"},),
        "accepted:a": (
            {
                "candidate": {
                    "items": [
                        {"assessment_id": "old", "status": "assessed"},
                        {"assessment_id": "new", "status": "reference_missing"},
                    ]
                }
            },
            {},
            {},
            {"status": "accepted"},
        ),
    }
    parent = SimpleNamespace(documents=lambda _: SimpleNamespace(read=records.get))
    tip = {"run_id": "original"}
    assert context_review.method_reuse_ids(parent, tip, ["new"]) == ["old"]
    with pytest.raises(ValueError, match="not_existing_score"):
        context_review.method_reuse_ids(parent, tip, ["old"])
    records["dynamic-phase-result"][0]["accepted"] = False
    with pytest.raises(ValueError, match="reuse_origin"):
        context_review.method_reuse_ids(parent, tip, ["new"])
