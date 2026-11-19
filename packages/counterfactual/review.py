"""Bounded Expert review of declared candidates via original Provider/RequestBudget."""

from agents.provider_agents.service import provider_locks
from packages.evidence_details.current_rejection import CurrentResearchRejected, current_rejection
from packages.grounding.models import digest, validate
from packages.llm_provider.models import ExpertProposal, SimulationExpertInput
from packages.llm_provider.provider import ProviderError

from .service import observe_cancellation


@observe_cancellation
def review(service, candidate, *, ports, cancel=None):
    result = service.run(candidate, cancel=cancel)
    if result["status"] != "candidate":
        return result
    key = "review:" + candidate.id
    saved = service._read(key)
    if saved:
        try:
            service._current(result["report"], cancel=cancel)
        except ValueError as error:
            if current_rejection(error) or str(error).startswith("simulation_control:"):
                return dict(status="safe_stop", reason=str(error), saved_review=saved[0])
            raise
        return saved[0]
    plan = service._read("plan")[0]
    fact = service.reader()
    provider = ports.expert.provider
    locks = {r.role: r for r in fact.binding.locks}
    if (
        ports.budget.run_id != service.run_id
        or ports.budget is not service.reader.path_reader.acceptance.requests
        or ports.budget.deadline != fact.binding.deadline
    ):
        raise ValueError("simulation_review_budget_scope")
    if provider.config.mode != "fake" or not provider.transport.simulated:
        raise ValueError("simulation_live_review_not_enabled")
    if any(locks.get(r.role) != r for r in provider_locks(provider.config)):
        raise ValueError("simulation_review_provider_lock")
    claim = dict(
        report_checksum=result["report"]["checksum"],
        plan_checksum=digest(plan),
        policy_checksum=digest(service.policy),
    )
    try:
        service._current(result["report"], cancel=cancel)
    except ValueError as error:
        if current_rejection(error) or str(error).startswith("simulation_control:"):
            return dict(status="safe_stop", reason=str(error), report=result["report"])
        raise
    if not service._claim("review-claim:" + candidate.id, claim):
        # Unconfirmed external return cannot be repaired by another Provider call.
        value = dict(
            **claim,
            status="safe_stop",
            reason="review_confirmation_unknown",
            suggestion=None,
            request_ids=[],
        )
    else:
        before = {r.request_id for r in ports.budget.records()}

        def recheck():
            try:
                service._current(result["report"], cancel=cancel)
            except ValueError as error:
                if str(error).startswith("simulation_control:") or current_rejection(error):
                    raise CurrentResearchRejected("simulation_review_stopped") from error
                raise

        def admit(proposal):
            allowed = {c["id"] for c in plan["candidates"]}
            sources = {s.reference.code for s in fact.state.material.sources}
            if (
                proposal.scope_checksum != digest(fact.state.material.scope)
                or not set(proposal.evidence) <= sources
                or len(proposal.evidence) != len(set(proposal.evidence))
                or (proposal.action == "accept" and proposal.action_id != candidate.id)
                or (proposal.action == "alternative" and proposal.action_id not in allowed)
                or (proposal.action in {"reject", "safe_stop"} and proposal.action_id is not None)
            ):
                raise ValueError("simulation_expert_out_of_policy")

        try:
            recheck()
            proposal = ports.expert.provider.generate_structured(
                role="expert",
                input_payload=dict(
                    role="expert",
                    material=fact.state.material.model_dump(mode="json"),
                    actions=[
                        dict(
                            id=c["id"],
                            kind="request_review",
                            target=fact.summary.graph,
                            required_inputs=[],
                            components=[],
                            expires_at=fact.binding.deadline,
                        )
                        for c in plan["candidates"]
                    ],
                    simulation_review=SimulationExpertInput(
                        policy_checksum=digest(service.policy),
                        report_checksum=result["report"]["checksum"],
                        proposed_candidate_id=candidate.id,
                        declared_candidates=tuple(plan["candidates"]),
                        comparisons=result["report"]["comparisons"],
                    ).model_dump(mode="json"),
                ),
                output_schema=ExpertProposal,
                scope=fact.state.material.scope,
                budget=ports.budget,
                deadline=min(fact.binding.deadline, ports.budget.deadline),
                admit=admit,
                before_request=recheck,
            )
            proposal = validate(ExpertProposal, proposal)
            recheck()
            value = dict(
                **claim,
                status="reviewed_candidate",
                reason="bounded_expert_advice",
                suggestion=proposal.model_dump(mode="json"),
            )
        except (CurrentResearchRejected, ProviderError):
            value = dict(
                **claim,
                status="safe_stop",
                reason="review_failed_or_current_control",
                suggestion=None,
            )
        value["request_ids"] = [
            r.request_id for r in ports.budget.records() if r.request_id not in before
        ]
    service._write(key, (value,))
    try:
        service._current(result["report"], cancel=cancel)
    except ValueError as error:
        if current_rejection(error) or str(error).startswith("simulation_control:"):
            return dict(status="safe_stop", reason=str(error), saved_review=value)
        raise
    return value
