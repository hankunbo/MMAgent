"""Synthetic fact runs through actual 08 acceptance and original Planner."""

from types import SimpleNamespace

from packages.counterfactual.facts import FactReader
from packages.counterfactual.models import Candidate, Intervention, SimulationPolicy
from packages.counterfactual.service import CounterfactualSimulation
from packages.grounding.models import digest
from packages.path_graph_validation import ResearchPathReader
from tests.agent_provider.support import ref, review_from_wire
from tests.expert_agent.support import setup
from tests.expert_agent.test_review_boundaries import integrated
from tests.path_graph_validation.support import configured, reply_for


def fixture(
    *,
    statuses=("obtained",),
    directions=None,
    same_source=False,
    shared_product=False,
    alternative_slot=False,
    pg=None,
    mutate=None,
    allow=None,
    policy_updates=None,
    profile_updates=None,
):
    if pg:
        f = setup(
            [review_from_wire],
            binding=pg.binding,
            reader=pg.reader,
            documents=pg.documents,
            budget=pg.budget,
            config=pg.config,
        )
    else:
        reader, binding = configured(
            statuses=statuses,
            same_source=same_source,
            shared_product=shared_product,
            alternative_slot=alternative_slot,
        )
        if mutate:
            reader, binding = mutate(reader, binding)
        if profile_updates:
            binding = binding.model_copy(
                update=dict(profile=binding.profile.model_copy(update=profile_updates))
            )
        f = setup([reply_for(statuses, directions)], reader=reader, binding=binding)
    parent = f.reader.state
    h = integrated(f, runtime=pg.documents.sessions if pg else None)
    result = h.service.run(**h.args)
    assert result["status"] == "accepted"
    path = ResearchPathReader(
        acceptance=h.service.acceptance, parent=parent, key=result["acceptance_key"]
    )
    reader = FactReader(path_reader=path, planner=h.planner, run_id=f.binding.run_id)
    fact = reader()
    policy = SimulationPolicy(
        source=ref("counterfactual_policy", "POLICY:SYNTHETIC:CF"),
        synthetic=True,
        case_code=f.binding.case_code,
        run_id=f.binding.run_id,
        fact_checksum=fact.checksum,
        rule=next(r for r in f.binding.locks if r.role == "rule"),
        allowed=allow
        or dict(
            remove_evidence=tuple(i.id for i in parent.policy.items),
            remove_resource=tuple(r.code for r in parent.resources),
        ),
        max_candidates=64,
        max_search=64,
        tolerance=0.001,
        coverage_retention=0.8,
        preserve_roles=True,
        metrics=("score", "support", "refutation", "conflict", "coverage"),
    )
    if policy_updates:
        policy = policy.model_copy(update=policy_updates)
    service = CounterfactualSimulation(reader=reader, documents=f.documents, policy=policy)
    return SimpleNamespace(
        f=f,
        h=h,
        reader=reader,
        fact=fact,
        parent=parent,
        policy=policy,
        service=service,
        accepted=result,
    )


def candidate(kind="remove_evidence", targets=("item:0",), id="one", alternative=None):
    return Candidate(
        id=id, interventions=(Intervention(kind=kind, targets=targets, alternative=alternative),)
    )


def run(x, c=None):
    c = c or candidate()
    x.service.declare((c,))
    return x.service.run(c)


def protection(x):
    return digest(
        dict(
            fact=x.reader().summary.original_report,
            state=x.f.reader.current(),
            graph=x.h.planner.read(x.f.binding.run_id).graph,
            requests=x.f.budget.records(),
            calls=x.h.planner.owner.calls,
        )
    )
