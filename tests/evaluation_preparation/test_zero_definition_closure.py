"""No-dispatch initialization remains recoverable; requests remain disallowed."""

import copy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation.generic_definition_execution import zero_review_journal
from packages.grounding.models import digest
from packages.llm_provider.bounded_live import BoundedLiveBudget
from packages.llm_provider.provider import MemoryJournal
from tests.agent_provider.batch_support import fixture as provider_fixture


@pytest.mark.parametrize("seconds", [2700, 900])
def test_original_budget_initialization_is_not_a_request(monkeypatch, seconds):
    from packages.evaluation_preparation import generic_definition_execution as m

    config = provider_fixture().config.model_copy(update=dict(mode="live"))
    binding = SimpleNamespace(
        run_id="RUN:ZERO",
        deadline=datetime.now(UTC) + timedelta(seconds=seconds),
        profile=SimpleNamespace(max_steps=20, max_no_progress=2, max_retries=1, seconds=seconds),
    )
    journal = MemoryJournal()
    budget = BoundedLiveBudget(
        binding=binding, config=config, journal=journal, bounded_phase=seconds < 2700
    )
    before = copy.deepcopy(journal.events)
    monkeypatch.setattr(
        m, "RuntimeAgentJournal", lambda docs: SimpleNamespace(read=lambda: tuple(journal.read()))
    )
    assert budget.used == 0 and len(before) == 2
    assert zero_review_journal(
        SimpleNamespace(config=config),
        SimpleNamespace(run_id=binding.run_id),
        SimpleNamespace(binding=binding),
    )
    assert journal.events == before


@pytest.mark.parametrize(
    "fault", ["unknown", "completed", "wrong_run", "config", "deadline", "unknown_metadata"]
)
def test_zero_review_journal_rejects_changed_or_dispatched_history(monkeypatch, fault):
    from packages.evaluation_preparation import generic_definition_execution as m

    config = provider_fixture().config.model_copy(update=dict(mode="live"))
    binding = SimpleNamespace(
        run_id="RUN:ZERO",
        deadline=datetime.now(UTC) + timedelta(seconds=2700),
        profile=SimpleNamespace(max_steps=20, max_no_progress=2, max_retries=1, seconds=2700),
    )
    journal = MemoryJournal()
    BoundedLiveBudget(binding=binding, config=config, journal=journal)
    if fault == "unknown":
        journal.append(
            dict(
                schema_version="agent-request-reservation/1.0.0",
                phase="reserved",
                request_id="unknown",
            )
        )
    elif fault == "completed":
        journal.append(dict(phase="completed", request_id="confirmed"))
    elif fault == "wrong_run":
        journal.events[0]["binding"]["run_id"] = "RUN:OTHER"
    elif fault == "config":
        journal.events[1]["config_checksum"] = digest("other config")
    elif fault == "deadline":
        journal.events[1]["deadline"] = (binding.deadline + timedelta(seconds=1)).isoformat()
    else:
        journal.append(dict(schema_version="unknown/1.0.0"))
    monkeypatch.setattr(
        m, "RuntimeAgentJournal", lambda docs: SimpleNamespace(read=lambda: tuple(journal.read()))
    )
    assert not zero_review_journal(
        SimpleNamespace(config=config),
        SimpleNamespace(run_id=binding.run_id),
        SimpleNamespace(binding=binding),
    )
