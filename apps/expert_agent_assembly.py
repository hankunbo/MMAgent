"""Explicit 08D offline factory, sharing 08C ports and original Planner/Runtime."""

from agents.expert_agent import ResearchReview
from agents.expert_agent.planner import PlannerReviewContext
from packages.evidence_details.acceptance import ResearchAcceptance


def assemble_research_review(
    *,
    planner,
    run_id,
    domain_owner,
    documents,
    ports,
    action_source,
    ranking_policy,
    review_policy,
    facts,
    install,
    protocol,
    live_permission=None,
):
    context = PlannerReviewContext(
        planner=planner,
        run_id=run_id,
        domain_owner=domain_owner,
        action_source=action_source,
        ranking_policy=ranking_policy,
        review_policy=review_policy,
        facts=facts,
        install=install,
        documents=documents,
    )
    acceptance = ResearchAcceptance(
        owner=domain_owner,
        documents=documents,
        requests=ports.budget,
        live_permission=live_permission,
    )
    return ResearchReview(acceptance=acceptance, ports=ports, context=context, protocol=protocol)
