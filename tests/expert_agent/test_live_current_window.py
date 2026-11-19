"""Original live owner reuse is synchronous; later boundaries must read again."""

from types import SimpleNamespace

import pytest

from agents.expert_agent.actions import ResearchActionSource
from apps.vectorengine_live_assembly import LiveSnapshots
from packages.planner_control.actions import SavedResearchActions
from packages.planner_control.models import Control, Phase, RankingPolicy
from tests.expert_agent.support import setup


def test_same_original_snapshot_and_domain_are_read_once_per_window():
    fixture = setup()
    calls = []

    def current():
        calls.append("current")
        return fixture.reader.current()

    owner = SimpleNamespace(current=current)
    source = ResearchActionSource(SavedResearchActions(LiveSnapshots(owner)), owner)
    state = Control(binding=fixture.binding, graph=fixture.binding.graph, phase=Phase.REVIEW)
    actions, domain = source.read_with_domain(state, RankingPolicy())
    assert domain == fixture.reader.current()
    assert any(a.branch == "research_review" for a in actions.actions)
    assert calls == ["current"], "same owner is reconstructed twice in one synchronous frame"
    source.read_with_domain(state, RankingPolicy())
    assert calls == ["current", "current"], "next invocation must revalidate, not reuse a cache"


def test_different_owner_keeps_both_reads_and_next_call_checks_binding():
    fixture = setup()
    calls = []
    first = SimpleNamespace(current=lambda: calls.append("snapshot") or fixture.reader.current())
    second = SimpleNamespace(current=lambda: calls.append("domain") or fixture.reader.current())
    source = ResearchActionSource(SavedResearchActions(LiveSnapshots(first)), second)
    state = Control(binding=fixture.binding, graph=fixture.binding.graph, phase=Phase.REVIEW)
    source.read_with_domain(state, RankingPolicy())
    assert calls == ["snapshot", "domain"]
    bad = fixture.binding.model_copy(update={"input_checksum": "sha256:" + "0" * 64})
    with pytest.raises(ValueError, match="live_snapshot_binding"):
        source.read_with_domain(state.model_copy(update={"binding": bad}), RankingPolicy())


def test_original_window_returns_checked_state_and_rejects_foreign_materials():
    from agents.expert_agent.service import ReviewCurrentWindow

    fixture = setup()
    state = fixture.reader.current()
    window = ReviewCurrentWindow(fixture.service, fixture.frame, state, None)
    assert window.current(fixture.ports.review, fixture.documents, state) == state
    with pytest.raises(ValueError, match="review_current_window_binding"):
        window.current(fixture.ports.vision, fixture.documents, state)
    with pytest.raises(ValueError, match="review_current_window_binding"):
        window.current(fixture.ports.review, object(), state)
    fixture.context.cancelled = True
    from packages.evidence_details.current_rejection import CurrentResearchRejected

    with pytest.raises(CurrentResearchRejected, match="cancelled"):
        window.current(fixture.ports.review, fixture.documents, state)


@pytest.mark.parametrize("boundary", ["validate_scope", "consumed"])
def test_document_reads_cannot_leave_a_stale_reusable_current_window(boundary):
    from agents.expert_agent.service import ReviewCurrentWindow
    from packages.evidence_details.current_rejection import CurrentResearchRejected
    from packages.grounding.models import digest

    fixture = setup()
    state = fixture.reader.current()
    window = ReviewCurrentWindow(fixture.service, fixture.frame, state, None)
    original = getattr(fixture.documents, boundary)

    def changed(*args):
        result = original(*args)
        fixture.reader.state = state.model_copy(update={"prompt_checksum": digest("changed")})
        return result

    setattr(fixture.documents, boundary, changed)
    with pytest.raises(CurrentResearchRejected, match="review_domain_changed"):
        window.current(fixture.ports.review, fixture.documents, state)


def test_next_window_revalidates_source_after_an_earlier_success():
    from agents.expert_agent.service import ReviewCurrentWindow
    from packages.evidence_details.current_rejection import CurrentResearchRejected
    from packages.grounding.models import digest

    fixture = setup()
    state = fixture.reader.current()
    window = ReviewCurrentWindow(fixture.service, fixture.frame, state, None)
    assert window.current(fixture.ports.review, fixture.documents, state) == state
    fixture.reader.state = state.model_copy(update={"prompt_checksum": digest("later")})
    with pytest.raises(CurrentResearchRejected, match="review_domain_changed"):
        window.current(fixture.ports.review, fixture.documents, state)


def test_untrusted_action_callback_cannot_change_owner_after_fallback_read():
    from agents.expert_agent.planner import PlannerReviewContext
    from agents.expert_agent.service import ReviewCurrentWindow
    from packages.evidence_details.current_rejection import CurrentResearchRejected
    from packages.grounding.models import digest

    f = setup()
    original = f.reader.current()
    event = f.frame.event.model_copy(update={"code": "review:" + f.frame.event.checksum[7:]})
    control = SimpleNamespace(
        phase=Phase.REVIEW,
        pending=SimpleNamespace(
            command=SimpleNamespace(action="review"), request_checksum=event.checksum
        ),
        cancelled=False,
        no_progress=0,
    )
    armed = [False]

    class CallbackSource:
        def read(self, state, policy):
            if armed[0]:
                f.reader.state = original.model_copy(
                    update={"prompt_checksum": digest("changed-after-domain-read")}
                )
            return f.frame.actions

    planner = SimpleNamespace(
        read=lambda _: control,
        observe_research_control=lambda *args, **kwargs: control,
        research_review_limit=lambda _: None,
    )
    context = PlannerReviewContext(
        planner=planner,
        run_id=f.binding.run_id,
        domain_owner=f.reader,
        action_source=CallbackSource(),
        ranking_policy=RankingPolicy(),
        review_policy=f.frame.policy,
        facts=lambda _: f.frame.facts,
        install=lambda *args, **kwargs: None,
        documents=f.documents,
    )
    f.service.context = context
    frame, _ = context.read_with_domain(event)
    window = ReviewCurrentWindow(f.service, frame, original, None)
    armed[0] = True
    with pytest.raises(CurrentResearchRejected, match="review_domain_changed"):
        window.current(f.ports.review, f.documents, original)
