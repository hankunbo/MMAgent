import pytest

from agents.expert_agent.actions import ResearchActionSource
from agents.expert_agent.models import ReviewFacts, ReviewPolicy
from agents.planner.service import Planner
from apps.expert_agent_assembly import assemble_research_review
from packages.grounding.models import digest
from packages.planner_control.models import Outcome, RankingPolicy
from tests.agent_provider.support import ref, review_from_wire
from tests.expert_agent.support import protocol, setup
from tests.planner.support import FakeOwner, MemoryRuntime, advance, command


@pytest.mark.parametrize("lost", ["none", "checkpoint", "install"])
def test_original_planner_event_successor_checkpoint_and_next_set(lost):
    f = setup([review_from_wire])

    class Owner(FakeOwner):
        def current(self, binding):
            return f.reader.current().material.scope.graph

        def invoke(self, state, pending, *, cancel):
            if pending.command.action == "execute":
                return Outcome(
                    reason="synthetic_observe_no_tool",
                    refs=(
                        ref("material", "ASSET:READ", checksum=digest(f.reader.current().material)),
                    ),
                )
            return super().invoke(state, pending, cancel=cancel)

    owner, runtime = Owner(f.binding), MemoryRuntime()
    owner.choices = f.source.choices
    planner = Planner(runtime=runtime, owner=owner)
    planner.begin(f.binding)
    advance(planner, until="review")
    cmd = command(planner, "review")
    event = planner.begin_research_review(f.binding.run_id, cmd)
    source = ResearchActionSource(f.source, f.reader)
    service = assemble_research_review(
        planner=planner,
        run_id=f.binding.run_id,
        domain_owner=f.reader,
        documents=f.documents,
        ports=f.ports,
        action_source=source,
        ranking_policy=RankingPolicy(),
        review_policy=ReviewPolicy(comparison_protocol=digest(protocol())),
        facts=lambda state: ReviewFacts(),
        install=f.context.install,
        protocol=protocol(),
    )
    frame = service.context.read(event)
    action = next(a for a in frame.actions.actions if a.branch == "research_review")
    if lost == "checkpoint":
        runtime.fail = "lost_reply"
    if lost == "install":
        f.context.fail_install = True
    args = dict(event=event, proposed_action_id=action.id, role="review")
    if lost != "none":
        with pytest.raises((RuntimeError, OSError)):
            service.run(**args)
    result = service.run(**args)
    assert result["status"] == "accepted"
    assert planner.read(f.binding.run_id).phase == "retrieve"
    assert planner.read(f.binding.run_id).graph.code == result["graph"]["code"]
    assert planner.begin_research_review(f.binding.run_id, cmd) == event
    assert service.run(**args) == result
    assert f.budget.used == 1
    assert runtime.state.started == 0
    assert runtime.state.reserved == 1  # Original observe prepare only; review reserves none.
    assert len(f.context.installs) == 1
    assert planner.replay(f.binding.run_id) == runtime.state
    for action_name in (
        "retrieve",
        "instantiate",
        "identify",
        "select",
        "prepare",
        "execute",
        "verify",
    ):
        planner.step(f.binding.run_id, command(planner, action_name))
    new_event = planner.begin_research_review(f.binding.run_id, command(planner, "review"))
    new = service.context.read(new_event)
    assert new.scope.graph.code == result["graph"]["code"]
    assert new.actions.graph == new.scope.graph
    assert f.reader.review_template(new.scope).scope_checksum == digest(new.scope)
