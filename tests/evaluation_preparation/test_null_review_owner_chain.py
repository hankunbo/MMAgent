"""Original Provider, admission, saved-origin readers and scoring; memory storage.

This is a candidate integration gate, not a real patient or PostgreSQL run.
"""

from types import SimpleNamespace

import pytest

from agents.expert_agent.reporting import read_research_report, render_research_html
from packages.evaluation_preparation.dynamic_stages import rebuild
from packages.evidence_details import accepted_aggregation as a
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.acceptance_models import read_records
from packages.grounding.models import digest, validate
from packages.llm_provider import DeterministicFakeProvider
from packages.multimodal_research.review_context import supplementary
from packages.planner_control.models import Control
from packages.research_runtime.agent_records import RuntimeAgentJournal
from tests.agent_provider.batch_support import generate
from tests.agent_provider.support import budget_for
from tests.evaluation_preparation.test_directional_review import explanation, fixture, reply
from tests.expert_agent.test_batch_assembly import LiveDocuments
from tests.expert_agent.test_legacy_accepted_aggregation import move_state


def response_for(target, value):
    def response(body):
        raw = reply(body)
        for row in raw["items"]:
            if row["assessment_id"] != target:
                continue
            row["status"] = "method_mismatch" if value is None else "assessed"
            row["judgment"].update(
                value=value,
                lower=None if value is None else -0.1,
                upper=None if value is None else 0.1,
                method="not_assessed" if value is None else "manual_source_bound_review",
                missing_reason="Synthetic method context insufficient" if value is None else None,
            )
            row["direction_assessment"] = dict(
                explanation(value), schema_version="direction-assessment/1.0.0"
            )
        return raw

    return response


def terminal(f):
    f.documents.run_id = f.binding.run_id
    control = validate(Control, dict(binding=f.binding, graph=f.binding.graph, phase="stopped"))
    f.documents.sessions = SimpleNamespace(read=lambda _: control)


def accepted(f, key):
    sub = generate(f)
    assert f.acceptance.submit(sub, key=key)["status"] == "submitted_not_accepted"
    assert f.acceptance.accept(key=key).status == "accepted"
    terminal(f)
    return sub


def combined(value):
    old = fixture()
    target = next(
        p.id
        for p in old.owner.current().policy.items
        if p.definition.interpretation_status == "installed"
    )
    old.provider = old.agent.provider = DeterministicFakeProvider(
        old.config, [response_for(target, None)] * 10
    )
    old_sub = accepted(old, "before")
    context = supplementary(
        {
            target: dict(
                unit_basis=None,
                source_limits=["Synthetic method supplement"],
                transform={"kind": "unchanged"},
                numeric={"value": 87},
                interpretation_limits={"not_clinical": True},
                owner_method={"version": "synthetic/2"},
            )
        }
    )
    seed = old.owner.current()
    full = rebuild(seed, seed.policy, (*seed.material.observations, *context))
    new = fixture()
    raw = full.policy.model_dump(mode="json")
    raw["items"] = [i for i in raw["items"] if i["id"] == target]
    selected = rebuild(
        move_state(full, "RUN:CONTEXT_REVIEW"),
        validate(type(full.policy), raw),
        full.material.observations,
    )
    new.owner.state = selected
    new.binding = old.binding.model_copy(update={"run_id": selected.material.scope.run_id})
    new.budget = budget_for(new.binding, new.config)
    new.agent.budget = new.budget
    new.acceptance.requests = new.budget
    new.provider = new.agent.provider = DeterministicFakeProvider(
        new.config, [response_for(target, value)]
    )
    new_sub = accepted(new, "after")
    current = move_state(full, "RUN:CONTEXT_COMBINED")
    readers = [
        a.AcceptedOriginReader(
            documents=f.documents,
            requests=a.HistoricalRequests(f.budget.journal),
            state_reader=f.owner.current,
            acceptance_key=key,
        )
        for f, key in ((old, "before"), (new, "after"))
    ]
    proofs = [r.read(current)[0] for r in readers]
    old_item = next(i for i in old_sub.candidate.items if i.assessment_id == target)
    new_item = new_sub.candidate.items[0]
    revision = a.NullReviewRevision(
        assessment_id=target,
        previous_run_id=proofs[0].run_id,
        replacement_run_id=proofs[1].run_id,
        previous_submission_checksum=digest(old_sub),
        replacement_submission_checksum=digest(new_sub),
        previous_item_checksum=digest(old_item),
        replacement_item_checksum=digest(new_item),
        new_context_checksum=digest(a._target_context(new.owner.current(), target)),
    )
    source = a.AcceptedReviewSet(readers, new.config, null_revisions=(revision,))
    docs = LiveDocuments()
    docs.run_id = current.material.scope.run_id
    docs.root = SimpleNamespace(checksum=current.material.scope.graph.checksum)
    binding = old.binding.model_copy(
        update={
            "run_id": docs.run_id,
            "case_code": current.material.scope.case_code,
            "input_checksum": current.material.scope.input_checksum,
            "visible_checksum": current.material.scope.visible_checksum,
        }
    )
    control = validate(Control, dict(binding=binding, graph=binding.graph, phase="stopped"))
    docs.sessions = SimpleNamespace(read=lambda _: control)
    owner = SimpleNamespace(current=lambda: current, accepted_review_set=source)
    requests = a.AggregationRequests(source, owner, RuntimeAgentJournal(docs))
    return SimpleNamespace(
        old=old,
        new=new,
        target=target,
        source=source,
        owner=owner,
        docs=docs,
        requests=requests,
        old_sub=old_sub,
        new_sub=new_sub,
    )


@pytest.mark.parametrize("value", [0.03, 0.0, None])
def test_original_accepted_origins_select_once_score_save_and_render(value):
    f = combined(value)
    before = [(digest(x.documents.saved), digest(x.budget.journal.read())) for x in (f.old, f.new)]
    permission, sub = a.AggregationPermission.prepare(
        owner=f.owner, documents=f.docs, requests=f.requests, config=f.new.config
    )
    service = ResearchAcceptance(
        owner=f.owner, documents=f.docs, requests=f.requests, live_permission=permission
    )
    assert service.submit(sub, key=a.ACCEPTANCE_KEY)["status"] == "submitted_not_accepted"
    receipt = service.accept(key=a.ACCEPTANCE_KEY)
    assert receipt.status == "accepted"
    saved = read_records(service.read(key=a.ACCEPTANCE_KEY)[0])
    items = {i.assessment_id: i for i in saved.candidate.items}
    assert len(items) == len(f.old_sub.candidate.items)
    assert items[f.target] == f.new_sub.candidate.items[0]
    assert all(
        items[i.assessment_id] == i
        for i in f.old_sub.candidate.items
        if i.assessment_id != f.target
    )
    assert (
        next(i for i in f.old_sub.candidate.items if i.assessment_id == f.target).judgment.value
        is None
    )
    report = read_research_report(service, key=a.ACCEPTANCE_KEY, parent=f.owner.current())
    assert "零新模型请求" in render_research_html(report)
    audit = len(f.docs.audit)
    assert service.accept(key=a.ACCEPTANCE_KEY) == receipt
    assert len(f.docs.audit) == audit and not f.requests.journal.read()
    assert before == [
        (digest(x.documents.saved), digest(x.budget.journal.read())) for x in (f.old, f.new)
    ]


@pytest.mark.parametrize(
    "fault", ["missing_acceptance", "unconfirmed_request", "wrong_selection", "changed_context"]
)
def test_revision_never_bypasses_original_owner(fault):
    f = combined(0.03)
    if fault == "missing_acceptance":
        f.new.documents.saved.pop("accepted:after")
    elif fault == "unconfirmed_request":
        f.new.budget.journal.append(dict(phase="reserved", request_id="REQUEST:UNCONFIRMED"))
    elif fault == "wrong_selection":
        f.source.null_revisions = (
            f.source.null_revisions[0].model_copy(
                update={"replacement_item_checksum": digest("wrong")}
            ),
        )
    else:
        state = f.new.owner.current()
        f.new.owner.state = state.model_copy(update={"prompt_checksum": digest("changed")})
    with pytest.raises(ValueError):
        f.source.current(f.owner.current())
    assert not f.docs.saved
