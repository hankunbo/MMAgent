"""Research advisory mapping. Formal v1 continues through its original validators."""

from agents.provider_agents.service import action_projection
from packages.evidence_details.current_rejection import CurrentResearchRejected
from packages.grounding.models import digest, validate
from packages.llm_provider.models import ExpertProposal
from packages.llm_provider.provider import ProviderError

from .models import ExpertDecision, triggers


def advise(*, frame, material, proposed, proposal, ports, recheck):
    codes, required = triggers(frame, proposed)
    suggestion, request_ids = None, ()
    decision, selected, reason = "baseline", proposed.id, "ordinary_step"
    before = {r.request_id for r in ports.budget.records()}
    if "no_progress" in codes:
        decision, selected, reason = "safe_stop", None, "no_new_information"
    elif codes:
        try:
            recheck()
            payload = dict(
                role="expert",
                material=material.model_dump(mode="json"),
                actions=action_projection(frame.actions),
                expert_review=dict(
                    event=frame.event,
                    proposal_checksum=digest(proposal),
                    proposed_action_id=proposed.id,
                    action_set_checksum=frame.actions.checksum,
                    policy_checksum=digest(frame.policy),
                    triggers=codes,
                    candidate=proposal if hasattr(proposal, "scope_checksum") else None,
                ),
            )
            from packages.llm_provider.input_scope import (
                ScopeVisionProposal,
                ScopeVisionProposalV2,
                ScopeVisionProposalV3,
            )

            if isinstance(proposal, ScopeVisionProposal):
                version = (
                    "1.2.0"
                    if isinstance(proposal, ScopeVisionProposalV3)
                    else "1.1.0"
                    if isinstance(proposal, ScopeVisionProposalV2)
                    else "1.0.0"
                )
                payload["schema_version"] = "input-scope-expert-input/" + version
                payload["expert_review"]["schema_version"] = "input-scope-expert-review/" + version

            def admit(result):
                refs = {s.reference.code for s in material.sources} | {
                    o.reference.code for o in material.observations
                }
                allowed = {a.id for a in frame.actions.actions}
                if result.scope_checksum != digest(frame.scope) or not set(result.evidence) <= refs:
                    raise ValueError("expert_scope")
                if len(set(result.evidence)) != len(result.evidence):
                    raise ValueError("expert_duplicate_evidence")
                if result.action == "accept" and result.action_id != proposed.id:
                    raise ValueError("expert_accept_wrong_proposal")
                if result.action == "alternative" and result.action_id not in allowed:
                    raise ValueError("expert_illegal_alternative")
                if result.action in {"reject", "safe_stop"} and result.action_id is not None:
                    raise ValueError("expert_control_action")
                recheck()

            agent = ports.expert
            suggestion = agent.provider.generate_structured(
                role="expert",
                input_payload=payload,
                output_schema=ExpertProposal,
                scope=frame.scope,
                budget=ports.budget,
                deadline=min(frame.actions.expires_at, ports.budget.deadline),
                admit=admit,
                before_request=recheck,
            )
            suggestion = validate(ExpertProposal, suggestion)
            decision, selected, reason = suggestion.action, suggestion.action_id, "expert_advice"
        except (CurrentResearchRejected, ProviderError):
            decision = "safe_stop" if required else "baseline"
            selected = None if required else proposed.id
            reason = "required_review_failed" if required else "optional_review_failed"
        request_ids = tuple(
            r.request_id for r in ports.budget.records() if r.request_id not in before
        )
    # Fallback cannot bypass a changed source/graph, cancellation or expired action.
    try:
        recheck()
    except CurrentResearchRejected:
        decision, selected, reason = "safe_stop", None, "context_changed_or_cancelled"
    return ExpertDecision(
        event=frame.event,
        scope=frame.scope,
        proposal_checksum=digest(proposal),
        proposed_action_id=proposed.id,
        action_set_checksum=frame.actions.checksum,
        policy_checksum=digest(frame.policy),
        source_uses=tuple((s.reference, s.purpose) for s in material.sources),
        triggers=codes,
        required=required,
        request_ids=request_ids,
        suggestion=suggestion,
        decision=decision,
        selected_action_id=selected,
        reason=reason,
    )
