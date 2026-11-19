"""Finite continuation: old terminal run unchanged, previous paid request not erased."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from apps.vectorengine_round import load_round
from packages.grounding.models import digest, validate
from packages.llm_provider.live_allocation import (
    StageAllocation,
    StageAllocationV2,
    prior_case_used,
    read_allocation,
)

CONFIG = "configs/research/vectorengine_batch_citations_case_quote_r01.json"


def saved_prior():
    from packages.planner_control.models import Control
    from tests.agent_provider.support import Documents

    package = Path("docs/acceptance/12-completion-cde-target-live-20261001")
    state = validate(Control, json.loads((package / "original-case-control.json").read_text()))
    events = json.loads((package / "original-case-journal.json").read_text())
    documents = Documents()
    documents.run_id = state.binding.run_id
    documents.sessions = SimpleNamespace(
        read=lambda _: state,
        origin=lambda _: (state.binding.started_at, state.binding.run_context_checksum),
    )
    for i, event in enumerate(events):
        documents.write("journal:" + str(i), (event,))
    return documents, state, events


def allocation():
    from apps.vectorengine_case_quote_gate import prior_case_usage

    new = load_round(CONFIG)
    prior = load_round(new.prior_case_config)
    public = load_round(new.public_config)
    docs, _, _ = saved_prior()
    usage = prior_case_usage(docs, prior)
    return StageAllocationV2(
        session_id=new.session_id,
        stage="case",
        public_run_id=public.run_id,
        public_used=2,
        public_journal_checksum=digest("public synthetic test binding"),
        configuration_checksums=(new.provider_checksum,),
        prior_cases=(usage,),
    )


def test_new_round_unique_identity_same_provider_and_no_budget_expansion():
    from apps.vectorengine_batch_prepare import preview
    from apps.vectorengine_round import CaseQuoteRepairRound

    new = load_round(CONFIG)
    old = load_round(new.prior_case_config)
    assert new.run_id != old.run_id and new.provider_checksum == old.provider_checksum
    assert new.max_requests == 40 and new.seconds == 2700
    assert preview("case", CONFIG)["round"]["source_quote_policy"] == new.source_quote_policy
    raw = new.model_dump(mode="json")
    for field, value in [
        ("run_id", old.run_id),
        ("max_requests", 41),
        ("source_quote_policy", "fuzzy"),
    ]:
        with pytest.raises(ValueError):
            validate(CaseQuoteRepairRound, dict(raw, **{field: value}))


def test_read_only_original_terminal_case_counts_failed_but_charged_request():
    from apps.vectorengine_case_quote_gate import prior_case_usage
    from packages.research_runtime.agent_records import RuntimeAgentJournal

    d, state, events = saved_prior()
    before = list(d.audit)
    result = prior_case_usage(d, load_round(load_round(CONFIG).prior_case_config))
    assert result.used == 1 and result.control_checksum == digest(state)
    assert RuntimeAgentJournal(d).read() == tuple(events)
    assert d.audit == before


@pytest.mark.parametrize(
    "mode", ["running", "pending", "empty", "unknown", "wrong_config", "missing_confirmation"]
)
def test_current_prior_unknown_or_nonterminal_cannot_be_treated_as_zero(mode):
    from apps.vectorengine_case_quote_gate import prior_case_usage

    docs, state, events = saved_prior()
    prior = load_round(load_round(CONFIG).prior_case_config)
    if mode == "running":
        docs.sessions.read = lambda _: state.model_copy(update={"phase": "initialize"})
    elif mode == "pending":
        docs.sessions.read = lambda _: state.model_copy(update={"pending": object()})
    elif mode == "empty":
        docs.saved.clear()
    elif mode == "unknown":
        docs.write("journal:" + str(len(events)), ({"phase": "reserved", "request_id": "unknown"},))
    elif mode == "wrong_config":
        prior = prior.model_copy(update={"provider_checksum": digest("other")})
    else:
        for value in docs.saved.values():
            if value[0].get("phase") == "response_confirmation":
                value[0]["confirmed"] = None
    with pytest.raises(ValueError, match="case_quote_prior"):
        prior_case_usage(docs, prior)


def test_allocation_versions_distinct_legacy_rejects_new_and_counts_not_relabelled():
    a = allocation()
    assert a.public_used == 2 and prior_case_used(a) == 1
    assert read_allocation(a) == a
    with pytest.raises(ValueError):
        validate(StageAllocation, a)
    old = a.model_dump(mode="json", exclude={"prior_cases"})
    old["schema_version"] = "bounded-live-allocation/1.0.0"
    assert prior_case_used(read_allocation(old)) == 0
    with pytest.raises(ValueError):
        validate(StageAllocationV2, a.model_copy(update={"prior_cases": a.prior_cases * 2}))


def test_shared_budget_recovery_and_hard_cap_include_prior_case(monkeypatch):
    from packages.llm_provider import MemoryJournal
    from packages.llm_provider.bounded_live import BoundedLiveBudget
    from packages.llm_provider.provider import ProviderError
    from tests.expert_agent.test_batch_assembly import assembly

    f = assembly(monkeypatch, config_path="configs/agents/vectorengine_batch_citations.json")
    journal = MemoryJournal()
    a = allocation()
    budget = BoundedLiveBudget(binding=f.binding, config=f.config, journal=journal, allocation=a)
    # This test exercises the durable allocation/reservation cap, not a remote provider.
    budget.check_control = lambda: None
    for _ in range(37):
        budget.reserve(f.initial.material.scope, f.binding.deadline, digest(f.config))
        # Only the cap is isolated here; completed records are covered by original budget tests.
        budget.summary = lambda: SimpleNamespace(pending_unknown=0, api_requests_unknown=0)
    assert budget.used + a.public_used + prior_case_used(a) == 40
    with pytest.raises(ProviderError) as error:
        budget.reserve(f.initial.material.scope, f.binding.deadline, digest(f.config))
    assert error.value.code == "budget"
    recovered = BoundedLiveBudget(binding=f.binding, config=f.config, journal=journal, allocation=a)
    assert recovered.used == 37
    with pytest.raises(ProviderError):
        BoundedLiveBudget(binding=f.binding, config=f.config, journal=journal)


def test_incremental_gate_rejects_extra_or_omitted_change_and_old_gate():
    from apps.vectorengine_case_quote_gate import BASELINE, verify_delta
    from apps.vectorengine_round import check_round_gate, implementation_files
    from packages.object_store import checksum

    before = json.loads((BASELINE / "implementation-before.json").read_text())
    # Positive proof belongs to its frozen revision; later implementation must not pass it.
    current = json.loads(
        Path(
            "docs/acceptance/12-completion-c-quote-integration-v1/implementation-freeze.json"
        ).read_text()
    )
    delta = {
        p: dict(before=before.get(p), after=current.get(p))
        for p in sorted(set(before) | set(current))
        if before.get(p) != current.get(p)
    }
    verify_delta(before, current, delta)
    with pytest.raises(ValueError, match="uncovered_delta"):
        verify_delta(before, current, {})
    with pytest.raises(ValueError, match="uncovered_delta"):
        verify_delta(before, dict(current, **{"apps/uncovered.py": digest("new")}), delta)
    live = {p: checksum(Path(p).read_bytes()) for p in implementation_files()}
    if live != current:
        with pytest.raises(ValueError, match="uncovered_delta"):
            verify_delta(before, live, delta)
    with pytest.raises(ValueError, match="round_regression_gate"):
        check_round_gate(load_round(load_round(CONFIG).prior_case_config))
