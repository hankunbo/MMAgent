"""Definitions must not inherit a Review-reader-only continuation context."""

from types import SimpleNamespace as NS

import pytest

from db.runtime.repository.case_pipeline import identity
from packages.multimodal_research import pending_review as p
from packages.multimodal_research.campaign_renewal import definition_phase_base
from packages.multimodal_research.review_context import retain_observations


def test_definition_projection_clears_coupled_context_without_changing_facts_or_parent():
    facts, reader = {"an_observation": object()}, object()
    parent = NS(
        observations=facts,
        confirmed_review_reader=reader,
        review_context_scope="current-owner-prior-visible/1.0.0",
        review_clarification="old",
    )
    broken = NS(**vars(parent))
    broken.confirmed_review_reader = None
    with pytest.raises(ValueError, match="review_retained_context_scope"):
        retain_observations(broken, {}, {}, ())
    current = definition_phase_base(parent)
    assert current is not parent and current.observations is facts
    assert current.confirmed_review_reader is None and current.review_clarification is None
    assert retain_observations(current, {}, {}, ("current_fact",)) == ("current_fact",)
    assert parent.confirmed_review_reader is reader
    assert parent.review_context_scope == "current-owner-prior-visible/1.0.0"
    assert parent.review_clarification == "old"


@pytest.mark.parametrize(
    "fault",
    [None, "operation", "revision", "source_intent", "journal", "control", "task", "visible"],
)
def test_definition_grant_requires_exact_zero_wire_owner(monkeypatch, tmp_path, fault):
    rid = identity("RUN", "task-research:definition-key")
    context = NS(
        run_id=rid,
        request=NS(
            input_checksum="input",
            task=NS(checksum="task"),
            visible_checksum="visible",
            scope=NS(case_code="case"),
        ),
    )
    occupied = {
        "operation": "definition:operation:0",
        "revision": "definition:revision",
        "source_intent": "live-stage-intent:source",
    }.get(fault)
    docs = NS(run_id=rid, read=lambda key: ({"reserved": True},) if key == occupied else None)
    monkeypatch.setattr(
        "packages.research_runtime.agent_records.RuntimeAgentJournal",
        lambda _: NS(read=lambda: [{"phase": "reserved"}] if fault == "journal" else []),
    )
    seen = []
    c = NS(
        request_key=lambda target: seen.append(target) or "definition-key",
        contexts=NS(read_research_run=lambda _: context),
        documents=lambda _: docs,
        sessions=NS(read=lambda _: {} if fault == "control" else None),
        protocol=dict(input="input", task="task"),
        base=NS(binding=dict(case_code="case"), visible_value=NS(checksum="visible")),
        phase_path=lambda _: tmp_path,
    )
    if fault == "task":
        context.request.task.checksum = "changed"
    if fault == "visible":
        context.request.visible_checksum = "changed"
    if fault is None:
        assert p.partition_projection_unstarted(c, definition=True) == (docs, context)
    else:
        with pytest.raises(ValueError, match="started_or_changed"):
            p.partition_projection_unstarted(c, definition=True)
    assert seen == ["definition-expansion"]


def test_new_definition_version_without_original_grant_cannot_resume(tmp_path):
    c = NS(output=tmp_path, protocol=dict(schema_version="multimodal-campaign-renewal/1.56.0"))
    assert p.partition_projection_repair_current(c) is False
