"""Root polarity is an owner-projected question, never an automatic score edit."""

from types import SimpleNamespace

import pytest

from agents.provider_agents.batched_review import run_batches
from packages.evidence_details.review_tasks import project_review_tasks
from packages.grounding.models import digest
from packages.multimodal_research.single_proposition import (
    clarification_spec,
    clarify_requirements,
    clarify_review_state,
)
from tests.agent_provider.test_accepted_review_reuse import accepted_fixture


def setup_clarification():
    f = accepted_fixture()
    state = f.owner.current()
    rows = [dict(id=p.id, definition=p.definition) for p in state.policy.requirements]
    task = SimpleNamespace(
        code="TASK:SYNTHETIC",
        target_code=state.material.goal_code,
        checksum=digest("synthetic-task"),
    )
    ids = f.owner.confirmed_review_reader.current(state).assessment_ids
    return f, rows, task, ids[0]


def test_only_changed_question_is_reviewed_and_saved_original_owner():
    f, rows, task, key = setup_clarification()
    before = digest(f.old.documents.saved)
    spec = clarification_spec(rows, task, [key])
    f.owner.state = clarify_review_state(f.owner.current(), task, spec)
    tasks = project_review_tasks(f.owner.current())
    target = next(t for t in tasks.tasks if t.assessment_id == key)
    assert target.goal in target.definition.interpretation
    assert "伪影/替代解释不自动支持疾病" in target.definition.interpretation
    assert (
        target.definition.source_refs
        == next(r for r in rows if r["id"] == key)["definition"].source_refs
    )
    sub = run_batches(
        f.agent, state=f.owner.current(), documents=f.documents, key="clarify", round_id="A"
    )
    assert key not in sub.historical.assessment_ids
    # This fixture also has two originally unassessed items; they are not
    # reusable judgments. The real one-item CLI additionally requires an
    # entirely assessed origin before claiming a one-item correction budget.
    expected = set(tasks.assessment_ids) - set(sub.historical.assessment_ids)
    assert {i.assessment_id for b in sub.batches for i in b.candidate.items} == expected
    assert key in expected and len(f.budget.records()) == len(expected)
    originals = {i.assessment_id: i for i in sub.historical.submission.candidate.items}
    assert all(
        i == originals[i.assessment_id]
        for i in sub.candidate.items
        if i.assessment_id in sub.historical.assessment_ids
    )
    f.acceptance.submit(sub, key="clarified")
    assert f.acceptance.accept(key="clarified").status == "accepted"
    from agents.expert_agent.reporting import read_research_report, render_research_html

    report = read_research_report(f.acceptance, key="clarified", parent=f.owner.current())
    assert "伪影/替代解释不自动支持疾病" in render_research_html(report)
    assert digest(f.old.documents.saved) == before


@pytest.mark.parametrize("change", ["task", "definition", "targets", "answer"])
def test_clarification_cannot_authorize_changed_binding_or_desired_answer(change):
    _, rows, task, key = setup_clarification()
    spec = clarification_spec(rows, task, [key])
    if change == "task":
        task.target_code = "GOAL:OTHER"
    elif change == "definition":
        spec["original_definitions"][key] = digest("different")
    elif change == "targets":
        spec["assessment_ids"] = ["not-present"]
    else:
        spec["desired_direction"] = "refute"
    with pytest.raises(ValueError, match="review_clarification"):
        clarify_requirements(rows, task, spec)


def test_actual_proposition_name_and_unchanged_definition_fields():
    from packages.dataset_adapters.multimodal.tasks import load_intent

    _, rows, _, key = setup_clarification()
    task = load_intent("advanced_glaucoma", "configs/research/multimodal_propositions_v1.json")[0]
    spec = clarification_spec(rows, task, [key])
    projected = clarify_requirements(rows, task, spec)
    for original, new in zip(rows, projected, strict=True):
        if original["id"] != key:
            assert new == original
        else:
            a, b = original["definition"].model_dump(), new["definition"].model_dump()
            assert "晚期青光眼" in b["interpretation"]
            assert b["interpretation"].endswith(a["interpretation"])
            for name in ("reference", "interpretation"):
                a.pop(name)
                b.pop(name)
            assert a == b
