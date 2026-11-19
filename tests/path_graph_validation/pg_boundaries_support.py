"""Synthetic original Runtime/Planner assembly shared by the R01 PG probes."""

from agents.expert_agent.models import ReviewFacts
from packages.path_graph_validation import PathGraphValidation, ResearchPathReader
from tests.agent_provider.support import review_from_wire
from tests.expert_agent.support import expert_reply, setup
from tests.expert_agent.test_review_boundaries import integrated


def ready_pg(pg):
    f = setup(
        [review_from_wire, expert_reply()],
        binding=pg.binding,
        reader=pg.reader,
        documents=pg.documents,
        budget=pg.budget,
        config=pg.config,
    )
    parent = f.reader.state
    h = integrated(f, runtime=pg.documents.sessions)
    first = h.service.run(**h.args)
    reader = ResearchPathReader(
        acceptance=h.service.acceptance, parent=parent, key=first["acceptance_key"]
    )
    service = PathGraphValidation(reader=reader, documents=pg.documents)
    summary = service.inspect(key="summary")
    for name in ("retrieve", "instantiate", "identify", "select", "prepare", "execute", "verify"):
        h.planner.step(f.binding.run_id, h.command(name))
    event = h.planner.begin_research_review(f.binding.run_id, h.command("review"))
    h.service.context.facts = lambda _: ReviewFacts(required=True)
    action = next(a for a in h.service.context.read(event).actions.actions if a.kind == "observe")
    args = dict(key="summary", review=h.service, event=event, proposed_action_id=action.id)
    return f, h, service, summary, args
