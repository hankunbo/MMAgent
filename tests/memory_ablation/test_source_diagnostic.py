"""Narrow owned-PG diagnosis of the actual 08 source admission, without task tools."""

import os
from datetime import timedelta

import pytest

from packages.evidence_details.current_rejection import CurrentResearchRejected
from tests.agent_provider.support import review_from_wire
from tests.counterfactual.support import candidate, fixture, run
from tests.expert_agent.pg_support import runtime_fixture
from tests.expert_agent.support import setup
from tests.expert_agent.test_pg_boundaries import database as database
from tests.memory_ablation.support import align_source_binding
from tests.research_task_binding_r01.observer import export


def test_counterfactual_source_admission(database, tmp_path, monkeypatch):
    if "MMAGENT_BINDING_DIAGNOSTIC_DIR" not in os.environ:
        diagnostics = tmp_path / "diagnostics"
        diagnostics.mkdir()
        monkeypatch.setenv("MMAGENT_BINDING_DIAGNOSTIC_DIR", str(diagnostics))
    pg = runtime_fixture(tmp_path)
    pg.binding = pg.binding.model_copy(
        update={
            "profile": pg.binding.profile.model_copy(update={"seconds": 3600}),
            "deadline": pg.binding.started_at + timedelta(seconds=3600),
        }
    )
    f = setup(
        [review_from_wire],
        binding=pg.binding,
        reader=pg.reader,
        documents=pg.documents,
        budget=pg.budget,
        config=pg.config,
    )
    with pytest.raises(CurrentResearchRejected, match="budget_changed"):
        f.service._check_current(f.context.frame, f.reader.current(), None)
    assert pg.budget.used == 0 and pg.budget.records() == ()
    budget, journal = pg.budget, pg.budget.journal.read()
    align_source_binding(pg)
    assert pg.budget.deadline == pg.binding.deadline
    assert pg.budget is budget and pg.budget.journal.read() == journal
    sim = fixture(pg=pg)
    report = run(sim, candidate(targets=("item:one",), id="remove"))["report"]
    assert sim.service.read(candidate_id="remove") == report
    with pytest.raises(ValueError, match="^source_review_budget_already_used$"):
        align_source_binding(pg)
    export(
        "source-admission",
        dict(
            result=sim.accepted,
            report=report,
            mismatch_rejection="budget_changed",
            deadline_consistent=True,
        ),
    )
