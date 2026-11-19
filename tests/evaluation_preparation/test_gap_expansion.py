"""Adaptive decisions using real definition/projection owners; only HTTP and storage are doubles."""

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from packages.grounding.models import digest
from packages.llm_provider import DeterministicFakeProvider
from packages.multimodal_research.definition_expansion import (
    DefinitionExpansion,
    definition_config,
    read_expansion_policy,
)
from tests.evaluation_preparation.test_definition_expansion import fixture as seed_fixture
from tests.evaluation_preparation.test_owner_dependency_definition import confirmed


def fixture(
    tmp_path, *, first_adopts=False, second_adopts=True, same_material=False, documents=None
):
    owner, draft, item, _, _ = confirmed(tmp_path)
    seed = seed_fixture(tmp_path, documents=documents)
    base = item.base
    base.task.target_code = draft["target"] = "GOAL:ADVANCED_GLAUCOMA"
    base.read_dependency_input = lambda: base.manifest
    base.views = dict(views=[dict(component="image-a", status="obtained")])
    seed.campaign.base = base
    seed.campaign.protocol["deadline"] = (datetime.now(UTC) + timedelta(hours=2)).isoformat()
    seed.campaign.usage = lambda: seed.budget.used
    policy = json.loads(
        Path("configs/research/multimodal_definition_gap_loop_v1.json").read_bytes()
    )
    policy["query_requirements"] = {q: [item.target] for q in policy["public_queries"]}
    sources = owner.sources
    original = copy.deepcopy(sources.value)
    other = copy.deepcopy(original)
    if not same_material:
        other["code"] = "SOURCE:OTHER_SYNTHETIC"
        other["text"] += " Additional synthetic material, no established relation."
        other["checksum"] = digest(other["text"])
        other["material"].update(text=other["text"])
        other["material"]["reference"].update(code=other["code"], checksum=other["checksum"])
    materials = {original["code"]: original, other["code"]: other}
    sources.read = lambda code: copy.deepcopy(materials[code])
    sources.index_keys = ("synthetic-index",)
    sources.restore_indexes = lambda indexes: None
    calls = []

    def selected(query):
        return original if query == policy["public_queries"][1] or first_adopts else other

    def search(**kw):
        calls.append(kw["query"])
        return dict(
            schema_version="research-definition-search/1.1.0",
            materials=[selected(kw["query"])],
            acquired_indexes=[],
            backlog=[],
        )

    sources.search = search
    sources.reconcile_pool = lambda **kw: dict(
        schema_version="research-definition-pool/1.0.0",
        materials=[selected(kw["queries"][0])],
        backlog=[],
        retrieval={"synthetic": True},
    )

    def response(adopt):
        return dict(
            schema_version="research-definition-proposal/5.0.0",
            definitions=[draft] if adopt else [],
            no_addition_reason=None if adopt else "No defensible new relation in these passages.",
        )

    provider = DeterministicFakeProvider(
        definition_config(
            seed.provider.config, prompt_version=policy["prompt_version"], source_output_tokens=8192
        ),
        [response(first_adopts), response(second_adopts)],
    )
    expansion = DefinitionExpansion(
        seed.campaign,
        documents=seed.documents,
        sources=sources,
        provider=provider,
        budget=seed.budget,
        scope=seed.scope,
        policy=policy,
    )
    return expansion, calls, materials


def test_empty_first_direction_then_real_owner_adopts_second_and_cold_reads(tmp_path):
    e, calls, _ = fixture(tmp_path)
    old = copy.deepcopy(e.campaign.base.observations)
    protocol = digest(e.campaign.protocol)
    e.advance(until="sources")
    assert len(calls) == 1 and e.budget.used == 0
    e.advance(until="definitions")
    assert len(calls) == 2 and e.budget.used == 2
    revision = e.advance(until="requirements")
    assert len(revision["replaced"]) == 1 and revision["added"] == []
    assert revision["gap_expansion"]["stop_reason"] == "adopted_increment_ready"
    assert [a["action"] for a in revision["gap_expansion"]["attempts"]] == [
        "try_distinct_public_direction",
        "schedule_adopted_requirements",
    ]
    assert e.campaign.base.observations == old and e.campaign.targets == ()
    assert e.campaign.base.ledger[0]["definition"].interpretation_status == "installed"
    assert digest(e.campaign.protocol) == protocol
    counts = (len(calls), e.budget.used, len(e.documents.audit))
    assert e.advance() == revision
    assert (len(calls), e.budget.used, len(e.documents.audit)) == counts
    # Cold executor with exact original inputs; no fresh Source response needed.
    cold, cold_calls, _ = fixture(tmp_path / "cold", documents=e.documents)
    cold.campaign.protocol = copy.deepcopy(e.campaign.protocol)
    cold.binding = copy.deepcopy(e.binding)
    assert cold.advance() == revision
    assert cold_calls == [] and cold.budget.used == 0


def test_first_adoption_does_not_search_second_direction(tmp_path):
    e, calls, _ = fixture(tmp_path, first_adopts=True)
    assert e.advance()["gap_expansion"]["stop_reason"] == "adopted_increment_ready"
    assert len(calls) == e.budget.used == 1


@pytest.mark.parametrize("same_material,requests", [(False, 2), (True, 1)])
def test_no_increment_falls_back_without_cancelling_or_inventing_direction(
    tmp_path, same_material, requests
):
    e, calls, _ = fixture(tmp_path, second_adopts=False, same_material=same_material)
    original = copy.deepcopy(e.original_rows)
    result = e.advance()
    assert not result["added"] and not result["replaced"]
    assert result["denominator_before"] == result["denominator_after"] == 1
    assert digest(e.campaign.base.ledger) == digest(original)
    assert result["gap_expansion"]["next_action"] == "continue_existing_qualified_requirements"
    assert not (tmp_path / "cancel").exists()
    assert len(calls) == 2 and e.budget.used == requests
    if same_material:
        assert result["gap_expansion"]["attempts"][-1]["outcome"] == "unchanged_material"


def test_review_time_reserve_stops_expansion_without_spending_it(tmp_path):
    e, calls, _ = fixture(tmp_path)
    e.campaign.protocol["deadline"] = (datetime.now(UTC) + timedelta(minutes=30)).isoformat()
    result = e.advance()
    assert result["expansion_stop"] == "definition_review_time_reserve"
    assert calls == [] and e.budget.used == 0
    assert e.gap_attempt is None


def test_unknown_is_not_no_increment_or_permission_to_try_another_direction(tmp_path):
    e, calls, _ = fixture(tmp_path)

    def fail(**kwargs):
        calls.append("unknown")
        raise TimeoutError("synthetic transport unknown")

    e.provider.generate_definitions = fail
    with pytest.raises(TimeoutError, match="synthetic transport unknown"):
        e.advance()
    assert e.documents.read("definition:gap:final-candidate") is None
    with pytest.raises(ValueError, match="definition_response_unknown_no_repeat"):
        e.advance()
    assert calls.count("unknown") == 1
    assert e.gap_attempt is None


def test_cancel_does_not_dispatch_or_claim_normal_fallback(tmp_path):
    e, calls, _ = fixture(tmp_path)
    (tmp_path / "cancel").touch()
    with pytest.raises(ValueError, match="cancelled"):
        e.advance()
    assert calls == [] and e.budget.used == 0


def test_source_withdrawal_after_definition_checkpoint_refuses_current_adoption(tmp_path):
    e, _, materials = fixture(tmp_path, first_adopts=True)
    e.advance(until="definitions")
    next(iter(materials.values()))["state"] = "withdrawn"
    with pytest.raises(ValueError, match="definition_draft_exact_excerpt"):
        e.advance(until="requirements")


def test_config_rejects_renamed_duplicate_direction_and_legacy_stays_single_round():
    p = json.loads(Path("configs/research/multimodal_definition_gap_loop_v1.json").read_bytes())
    a, b = p["public_queries"]
    for field in ("public_topic_sets", "passage_plans", "query_requirements"):
        p[field][b] = p[field][a]
    with pytest.raises(ValueError, match="definition_equivalent_public_direction"):
        read_expansion_policy(p)
    old = read_expansion_policy(
        json.loads(Path("configs/research/multimodal_definition_gate_v22.json").read_bytes())
    )
    assert old.max_rounds == 1 and not hasattr(old, "reserve_review_seconds")


def test_new_request_budget_preserves_review_and_prior_budget(tmp_path):
    e, calls, _ = fixture(tmp_path)
    e.campaign.protocol["max_new_requests"] = 3
    result = e.advance()
    assert result["expansion_stop"] == "definition_review_reserve"
    assert e.budget.used == 0 and calls == []


def test_confirmed_operation_reuses_bytes_not_authorization(tmp_path):
    e, _, _ = fixture(tmp_path)
    checks = []
    e.campaign.guard = lambda **kw: checks.append("current")
    assert e.operation("definition:synthetic", "local", lambda: {"value": 1}) == {"value": 1}
    count = len(checks)
    assert count > 0
    assert e.operation("definition:synthetic", "local", lambda: pytest.fail("reissued")) == {
        "value": 1
    }
    assert len(checks) == count
    (tmp_path / "cancel").touch()
    with pytest.raises(ValueError, match="cancelled"):
        e.operation("definition:synthetic", "local", lambda: pytest.fail("reissued"))


def test_technical_backlog_is_distinct_from_normal_empty_material():
    from packages.multimodal_research.gap_expansion import classify

    candidate = dict(
        rows=[],
        added=[],
        replaced=[],
        merged=[],
        found={},
        backlog=[dict(reason="source_fetch_technical_failure")],
    )
    assert classify(candidate)["issues"] == ["retrieval_technical_error_not_empty_result"]


def test_public_prepare_and_cold_assembly_select_actual_policy(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from packages.multimodal_research import entry, entry_preflight
    from tests.dataset_adapters.test_multimodal import gamma

    def views(command, **kwargs):
        out = Path(command[4])
        out.mkdir(parents=True)
        (out / "views.json").write_text(json.dumps(dict(views=[])))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(entry.subprocess, "run", views)
    out = tmp_path / "entry"
    result = entry.prepare_entry(
        dataset="gamma",
        root=gamma(tmp_path / "source"),
        sample="0001",
        task="advanced_glaucoma",
        output=out,
        registry="configs/research/multimodal_propositions_v1.json",
        active_acquisition="configs/research/multimodal_active_requirements_v5.json",
        resilient_vision=True,
        view_profile="detail-v1",
    )
    assert result["api_requests"] == result["database_writes"] == 0
    spec = json.loads((out / "entry.json").read_bytes())
    assert spec["schema_version"] == "multimodal-live-entry/1.15.0"

    def boundary():
        raise RuntimeError("offline_original_owner_boundary")

    monkeypatch.setattr(entry, "owners", boundary)
    # This test isolates version dispatch; real resource readiness has CLI tests.
    monkeypatch.setattr(entry_preflight, "check_local_route", lambda *args: "synthetic_route")
    with pytest.raises(RuntimeError, match="offline_original_owner_boundary"):
        entry.assemble_entry(out)
    spec["schema_version"] = "multimodal-live-entry/1.13.0"
    (out / "entry.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="adaptive_gap_profile_entry_version"):
        entry.assemble_entry(out)


def test_current_revision_rejects_withdrawal_at_revision_boundary(tmp_path):
    e, _, materials = fixture(tmp_path)
    e.advance()
    next(iter(materials.values()))["state"] = "withdrawn"
    with pytest.raises(ValueError, match="definition_revision_source_changed"):
        e.current()


def test_decision_checkpoint_recovery_never_repeats_completed_directions(tmp_path):
    e, calls, _ = fixture(tmp_path)
    write = e.documents.write

    def fail_final(key, values):
        if key == "definition:gap:final-candidate":
            raise OSError("synthetic local checkpoint error")
        return write(key, values)

    e.documents.write = fail_final
    with pytest.raises(OSError, match="synthetic local checkpoint error"):
        e.advance()
    assert len(calls) == e.budget.used == 2
    e.documents.write = write
    result = e.advance()
    assert len(result["replaced"]) == 1
    assert len(calls) == e.budget.used == 2


def test_saved_decisions_survive_display_without_score_mutation(tmp_path):
    from packages.evaluation_preparation.evidence_tree_v3 import build, render
    from tests.evaluation_preparation.test_unified_view import report

    e, _, _ = fixture(tmp_path, second_adopts=False)
    revision = e.advance()
    value = report()
    scores = copy.deepcopy(value["scores"])
    value["definition_revision"] = revision
    packet = build(value, input_checksum=digest(value))
    assert packet["appendices"]["definition_revision"]["gap_expansion"] == revision["gap_expansion"]
    page = render(value, input_checksum=digest(value))
    assert "continue_existing_qualified_requirements" in page
    assert packet["scores"] == value["scores"] == scores
