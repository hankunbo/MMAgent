"""Original continued Review contributes only its new, fully verified batches."""

from types import SimpleNamespace

import pytest

from agents.provider_agents.batched_review import run_batches
from packages.evidence_details.acceptance_models import DomainState
from packages.evidence_details.accepted_aggregation import (
    AcceptedOriginReader,
    AcceptedReviewSet,
    HistoricalRequests,
)
from packages.grounding.models import digest, validate
from packages.planner_control.models import Control
from tests.agent_provider.test_accepted_review_reuse import accepted_fixture, state_from


def fixture(batch_size=1):
    f = accepted_fixture(new_observation=True, batch_size=batch_size)
    full = f.owner.current()
    retained = set(f.owner.confirmed_review_reader.current(full).assessment_ids)
    selected = retained | {"item:additional"}
    raw = full.model_dump(mode="json")
    raw["policy"]["items"] = [i for i in raw["policy"]["items"] if i["id"] in selected]
    f.owner.state = state_from(raw)
    sub = run_batches(
        f.agent, state=f.owner.current(), documents=f.documents, key="new", round_id="A"
    )
    f.acceptance.submit(sub, key="next")
    assert f.acceptance.accept(key="next").status == "accepted"
    state = f.owner.current()
    f.documents.run_id = state.material.scope.run_id
    binding = f.old.binding.model_copy(update={"run_id": f.documents.run_id})
    control = validate(Control, dict(binding=binding, graph=binding.graph, phase="stopped"))
    f.documents.sessions = SimpleNamespace(read=lambda _: control)
    raw = full.model_dump(mode="json")
    raw["material"]["scope"]["run_id"] = "RUN:WHOLE_AFTER_REUSE"
    current = state_from(raw)
    readers = [
        AcceptedOriginReader(
            documents=f.old.documents,
            requests=HistoricalRequests(f.old.budget.journal),
            state_reader=f.old.owner.current,
            acceptance_key="old-accepted",
        ),
        AcceptedOriginReader(
            documents=f.documents,
            requests=HistoricalRequests(f.budget.journal),
            state_reader=f.owner.current,
            acceptance_key="next",
            confirmed_review_reader=f.owner.confirmed_review_reader,
            reused_origin_new_items_only=True,
        ),
    ]
    return f, current, readers, sub


@pytest.mark.parametrize("batch_size", [1, 3])
def test_reused_origin_current_owner_union_no_duplicate_or_new_request(batch_size):
    f, current, readers, sub = fixture(batch_size)
    before = (
        digest(f.old.documents.saved),
        digest(f.documents.saved),
        digest(f.budget.journal.read()),
    )
    combined, requests = AcceptedReviewSet(readers, f.old.config).current(current)
    expected = {i.assessment_id: i for i in sub.historical.submission.candidate.items}
    expected.update({i.assessment_id: i for i in sub.candidate.items})
    assert {i.assessment_id: i for i in combined.candidate.items} == expected
    assert readers[1].read(current)[0].item_requests.keys() == {"item:additional"}
    assert len({r.request_id for r in requests}) == len(requests)
    assert sum(len(o.item_requests) for o in combined.origins) == len(expected)
    assert before == (
        digest(f.old.documents.saved),
        digest(f.documents.saved),
        digest(f.budget.journal.read()),
    )


def test_reused_aggregation_plan_is_explicit_and_legacy_cannot_enable_it():
    from packages.multimodal_research.accepted_aggregation import (
        RetainedOriginAggregationPlan,
        read_plan,
    )

    raw = dict(
        schema_version="accepted-review-aggregation-plan/1.5.0",
        facts="new-facts",
        origins=[dict(campaign="old"), dict(campaign="continued")],
    )
    assert isinstance(read_plan(raw), RetainedOriginAggregationPlan)
    with pytest.raises(ValueError):
        read_plan({**raw, "reuse_kind": "caller_selected_favorable_items"})
    with pytest.raises(ValueError):
        read_plan(
            {
                **raw,
                "schema_version": "accepted-review-aggregation-plan/1.3.0",
                "reuse_kind": "owner_verified_new_batches_only",
            }
        )


@pytest.mark.parametrize(
    "fault", ["legacy", "missing_owner", "source_changed", "old_origin_omitted", "duplicate"]
)
def test_reused_origin_requires_exact_current_ancestor_and_complete_union(fault):
    f, current, readers, _ = fixture()
    if fault == "legacy":
        readers[1].reused_origin_new_items_only = False
    elif fault == "missing_owner":
        readers[1].confirmed_review_reader = None
    elif fault == "source_changed":
        old = f.old.owner.current()
        f.old.owner.state = validate(
            DomainState, old.model_copy(update={"config_checksum": digest("changed")})
        )
    elif fault == "old_origin_omitted":
        readers = readers[1:]
    else:
        readers = [*readers, readers[1]]
    before = (
        digest(f.old.documents.saved),
        digest(f.documents.saved),
        digest(f.budget.journal.read()),
    )
    with pytest.raises(ValueError):
        AcceptedReviewSet(readers, f.old.config).current(current)
    assert before == (
        digest(f.old.documents.saved),
        digest(f.documents.saved),
        digest(f.budget.journal.read()),
    )
