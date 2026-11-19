"""A selective Review must retain prior observed context without another vote.

Real Phase/material projection; no network, database, or medical inference.
The opt-in scope is for a future explicit renewal version, never legacy replay.
"""

from types import SimpleNamespace

import pytest

from packages.dataset_adapters.multimodal.adapters import prepare
from packages.dataset_adapters.multimodal.tasks import load_intent
from packages.grounding.models import digest
from packages.multimodal_research import material
from packages.multimodal_research.campaign import Phase, configuration
from tests.dataset_adapters.test_multimodal import gdp
from tests.evaluation_preparation.test_dynamic_evidence import dynamic_fixture
from tests.evaluation_preparation.test_multimodal_campaign import sources


def projection_fixture(tmp_path):
    prepared = prepare("harvard_gdp", gdp(tmp_path / "source"), "data_0001.npz")
    seed = dynamic_fixture(1).owner.current()
    task = load_intent("glaucoma_detection")[0]
    medical = sources()
    ledger = material.requirements(prepared.manifest, medical.initial, task)
    values = {
        c.id: material.numeric(c, dict(prepared.payload_bytes))
        for c in prepared.manifest.components
    }
    observations = {
        r["id"]: material.observation(r["id"], values[r["component"]])
        for r in ledger
        if r["component"] in values and values[r["component"]] is not None
    }
    base = SimpleNamespace(
        observations=observations,
        ledger=ledger,
        task=task,
        sources=medical,
        resources=seed.resources,
        visible_value=SimpleNamespace(
            case=SimpleNamespace(case_code=seed.material.scope.case_code, laterality="UNKNOWN"),
            checksum=seed.material.scope.visible_checksum,
            components=SimpleNamespace(checksum=seed.material.scope.input_checksum),
        ),
        root=SimpleNamespace(
            asset_code=seed.material.scope.graph.code,
            checksum=seed.material.scope.graph.checksum,
        ),
    )
    phase = Phase(base, observation_reader=lambda: {})
    phase.context = SimpleNamespace(
        run_id=seed.material.scope.run_id, request=SimpleNamespace(task=task)
    )
    phase.config = configuration()
    original = phase.current_projection()
    base.confirmed_review_reader = SimpleNamespace(state_reader=lambda: original)
    omitted = next(iter(observations))
    base.review_target_ids = tuple(k for k in observations if k != omitted)
    return phase, original, omitted


def test_legacy_selective_projection_preserves_historical_behavior(tmp_path):
    phase, original, omitted = projection_fixture(tmp_path)
    selected = phase.current_projection()
    missing = phase.base.observations[omitted].reference.code
    assert missing in {o.reference.code for o in original.material.observations}
    assert missing not in {o.reference.code for o in selected.material.observations}


def test_opt_in_retains_current_fact_as_context_without_scoring_item(tmp_path):
    phase, original, omitted = projection_fixture(tmp_path)
    before = digest(original)
    phase.base.review_context_scope = "current-owner-prior-visible/1.0.0"
    selected = phase.current_projection()
    fact = phase.base.observations[omitted]
    assert fact in selected.material.observations
    assert omitted not in {item.id for item in selected.policy.items}
    assert len(selected.policy.items) == len(original.policy.items) - 1
    assert len({o.reference.code for o in selected.material.observations}) == len(
        selected.material.observations
    )
    assert digest(original) == before


@pytest.mark.parametrize(
    "change", ["missing", "changed", "unknown_scope", "missing_owner", "ambiguous"]
)
def test_retained_context_needs_current_exact_owner_fact(tmp_path, change):
    phase, _, omitted = projection_fixture(tmp_path)
    phase.base.review_context_scope = "current-owner-prior-visible/1.0.0"
    if change == "missing":
        del phase.base.observations[omitted]
    elif change == "changed":
        old = phase.base.observations[omitted]
        phase.base.observations[omitted] = old.model_copy(update={"value": old.value + " changed"})
    elif change == "unknown_scope":
        phase.base.review_context_scope = "caller-trust-me"
    elif change == "missing_owner":
        phase.base.confirmed_review_reader = None
    else:
        old = phase.base.observations[omitted]
        phase.observation_reader = lambda: {
            "unrelated-key": old.model_copy(update={"value": old.value + " contradictory"})
        }
    with pytest.raises(ValueError, match="review_retained_context"):
        phase.current_projection()


def test_previous_display_context_is_rebuilt_from_current_owner_not_old_bytes(
    tmp_path, monkeypatch
):
    from packages.multimodal_research import review_context

    phase, _, omitted = projection_fixture(tmp_path)
    selected_ids = phase.base.review_target_ids
    del phase.base.review_target_ids
    phase.base.execution_profile = "synthetic_context_fixture"
    calls = []

    def collect(base, observations):
        calls.append(set(observations))
        return {
            key: dict(
                schema_version="synthetic-context/1.0.0",
                observation=value.reference.model_dump(mode="json"),
                unit_basis="synthetic",
                source_limits=[],
                transform=dict(kind="synthetic"),
                numeric=None,
                interpretation_limits=dict(scope="context only, not another vote"),
            )
            for key, value in observations.items()
        }

    monkeypatch.setattr(review_context, "collect_campaign", collect)
    original = phase.current_projection()
    phase.base.confirmed_review_reader = SimpleNamespace(state_reader=lambda: original)
    phase.base.review_target_ids = selected_ids
    phase.base.review_context_scope = "current-owner-prior-visible/1.0.0"
    result = phase.current_projection()
    assert set(calls[-2]) == set(selected_ids)
    assert set(calls[-1]) == set(phase.base.observations)
    assert all(o in result.material.observations for o in original.material.observations)
    assert omitted not in {p.id for p in result.policy.items}
