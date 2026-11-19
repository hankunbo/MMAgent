"""Source identities shortened before execution; old model failures stay failures."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from apps.active_evidence_materials import ActiveCaseEvidence
from apps.vectorengine_single_case import load_config
from packages.evidence_acquisition.source_aliases import alias_sources, prepare_source_aliases
from packages.evidence_details.acceptance_models import DomainState
from packages.evidence_details.review_batches import ReviewBatchPlan, check_batch, make_plan
from packages.evidence_details.review_tasks import project_review_tasks
from packages.grounding.models import digest, validate
from packages.llm_provider.public_batches import batch_wire
from packages.llm_provider.review_response import decode
from tests.agent_provider.test_review_citation_contract import owner, response, schema_from

CONFIG = "configs/agents/vectorengine_active_acquisition_r02.json"


def test_bijection_preserves_medical_content_and_original_bytes():
    state = owner(distinct=True).current()
    before = state.model_dump_json()
    prepared, manifest = prepare_source_aliases(state)
    assert state.model_dump_json() == before
    assert len(prepared.policy.items) == len(state.policy.items)
    assert prepared.resources == state.resources
    assert prepared.material.observations == state.material.observations
    for old, new in zip(state.material.sources, prepared.material.sources, strict=True):
        assert old.model_dump(exclude={"reference"}) == new.model_dump(exclude={"reference"})
        binding = next(x for x in manifest["bindings"] if x["alias"]["code"] == new.reference.code)
        assert binding["original"] == old.reference.model_dump(mode="json")
        assert binding["source_checksum"] == new.reference.checksum == digest(old)
    assert alias_sources(tuple(reversed(state.material.sources)))[1] == manifest
    tasks = project_review_tasks(prepared).tasks
    assert all(r.code.startswith("SRC:") for t in tasks for r in t.allowed_sources)
    assert prepared.material.scope.policy_checksum != state.material.scope.policy_checksum


def test_owner_reconstructs_current_alias_and_detects_change(monkeypatch):
    sources = owner(distinct=True).current().material.sources
    value = object.__new__(ActiveCaseEvidence)
    value.base = SimpleNamespace(read_sources=lambda: sources[:-1])
    monkeypatch.setattr(value, "packet", lambda: sources[-1])
    aliases, value.alias_manifest = alias_sources(sources)
    assert value.read_sources() == aliases
    monkeypatch.setattr(value, "packet", lambda: sources[-1].model_copy(update={"text": "changed"}))
    with pytest.raises(ValueError, match="binding_changed"):
        value.read_sources()

    def withdrawn():
        raise ValueError("source_withdrawn")

    value.base.read_sources = withdrawn
    with pytest.raises(ValueError, match="source_withdrawn"):
        value.read_sources()


@pytest.mark.parametrize("bad", ["foreign", "cross_task", "long_original", "duplicate"])
def test_alias_scope_is_still_enforced_by_original_owner(bad):
    original = owner(distinct=True).current()
    state, _ = prepare_source_aliases(original)
    config = load_config(CONFIG)
    plan = make_plan(state, config.review_batching, key="alias-wire")
    wire = batch_wire(config, state, plan, plan.batches[0])
    candidate = response(wire)
    assert check_batch(state, plan, plan.batches[0], candidate)
    branches = schema_from(wire)["properties"]["items"]["items"]["anyOf"]
    assert branches[0]["properties"]["reference_citations"]["items"]["enum"] == ["SRC:01"]
    references = {
        "foreign": ["SRC:99"],
        "cross_task": ["SRC:02"],
        "long_original": [original.policy.items[0].source_codes[0]],
        "duplicate": ["SRC:01", "SRC:01"],
    }
    candidate["items"][0]["reference_citations"] = references[bad]
    expected = "review_duplicate_citation" if bad == "duplicate" else "review_citation_scope"
    with pytest.raises(ValueError, match=expected):
        check_batch(state, plan, plan.batches[0], candidate)


def test_alias_end_to_end_original_acceptance_install_score_and_readback(monkeypatch, tmp_path):
    from tests.expert_agent import test_batch_assembly as helper

    original = helper.initial_state

    def initial(*args):
        state, image = original(*args)
        return prepare_source_aliases(state)[0], image

    monkeypatch.setattr(helper, "initial_state", initial)
    f = helper.assembly(monkeypatch, config_path=CONFIG)
    stages, _ = helper.stages(f, tmp_path)
    assert [s["status"] for s in stages] == ["accepted"] * 3
    assert f.ports.budget.used == 5
    records = f.review.acceptance.read(key=stages[-1]["acceptance_key"])[0]
    assert len(records["batch_results"]) == 3
    assert "SRC:01" in (tmp_path / "review-report.html").read_text()
    assert helper.stages(f, tmp_path)[0] == stages and f.ports.budget.used == 5


def test_real_r01_wrong_hash_remains_rejected_no_fuzzy_correction():
    root = Path("docs/acceptance/12-live-active-acquisition-r01-v1/readback")
    docs = json.loads((root / "original-case-documents.json").read_text())
    event = docs["live-stage-intent:review"][0]["event"]["code"]
    state = validate(DomainState, docs["review:" + event + ":intent"][0]["state"])
    plan = validate(ReviewBatchPlan, json.loads((root / "batch-progress.json").read_text())["plan"])
    records = json.loads((root / "actual-requests.json").read_text())
    failed = next(x["raw_response"][0] for x in records if x["record"]["error"] == "scope")
    candidate, _ = decode(
        failed["raw_model_text"],
        load_config("configs/agents/vectorengine_active_acquisition_r01.json"),
    )
    with pytest.raises(ValueError, match="review_citation_scope"):
        check_batch(state, plan, plan.batches[0], candidate)
    prepared, _ = prepare_source_aliases(state)
    old_fixed = [p for p in state.policy.items if p.fixed_model_row is not None]
    assert [p for p in prepared.policy.items if p.fixed_model_row is not None] == old_fixed
    assert len(prepared.policy.items) == 46
    assert len(project_review_tasks(prepared).tasks) == 45


def test_round_history_cannot_reset_previous_seventeen_requests():
    from apps.vectorengine_round import ActiveAcquisitionAliasRound, load_round

    r = load_round("configs/research/vectorengine_active_acquisition_r02_case.json")
    assert len(r.prior_round_configs) == 4 and r.max_requests == 40
    raw = r.model_dump(mode="json")
    raw["prior_round_configs"] = []
    with pytest.raises(ValueError, match="active_history_inventory"):
        validate(ActiveAcquisitionAliasRound, raw)
