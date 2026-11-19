"""Explicit version dispatch; never upgrade a saved v1 score on read."""

from packages.evidence_scoring import continuous, engine, interpretation, participation


def build(base, version="1.0.0"):
    if version == "11.0.0":
        from packages.evidence_details.acceptance_models import DomainState, TaskPolicyV7
        from packages.evidence_scoring.accepted import build_accepted
        from packages.grounding.models import validate as validate_value
        from packages.llm_provider.models import ReasonedReviewProposal

        if set(base) not in ({"state", "candidate"}, {"state", "candidate", "reviewed_ids"}):
            raise ValueError("accepted_score_explicit_inputs_required")
        state = validate_value(DomainState, base["state"])
        if not isinstance(state.policy, TaskPolicyV7):
            raise ValueError("score_policy_version_mismatch")
        candidate = (
            validate_value(ReasonedReviewProposal, base["candidate"])
            if base["candidate"] is not None
            else None
        )
        return build_accepted(state, candidate, reviewed_ids=base.get("reviewed_ids"))
    if version in {"6.0.0", "6.1.0", "7.0.0", "8.0.0", "9.0.0", "10.0.0"}:
        from packages.evidence_details.acceptance_models import DomainState

        if version == "6.0.0":
            from packages.evidence_scoring.accepted_v6 import build_accepted
        else:
            from packages.evidence_scoring.accepted import build_accepted
        from packages.grounding.models import validate as validate_value
        from packages.llm_provider.models import ReviewProposal

        if set(base) != {"state", "candidate"} and not (
            version in {"7.0.0", "8.0.0", "9.0.0", "10.0.0"}
            and set(base) == {"state", "candidate", "reviewed_ids"}
        ):
            raise ValueError("accepted_score_explicit_inputs_required")
        state = validate_value(DomainState, base["state"])
        if getattr(state.policy, "direction_policy", None):
            raise ValueError("score_policy_version_mismatch")
        from packages.evidence_details.acceptance_models import (
            TaskPolicyV3,
            TaskPolicyV4,
            TaskPolicyV5,
            TaskPolicyV6,
        )

        if (
            (version in {"7.0.0", "8.0.0", "9.0.0", "10.0.0"})
            != isinstance(state.policy, TaskPolicyV3)
            or (version in {"8.0.0", "9.0.0", "10.0.0"}) != isinstance(state.policy, TaskPolicyV4)
            or (version in {"9.0.0", "10.0.0"}) != isinstance(state.policy, TaskPolicyV5)
            or (version == "10.0.0") != isinstance(state.policy, TaskPolicyV6)
        ):
            raise ValueError("score_policy_version_mismatch")
        candidate = (
            None
            if version in {"7.0.0", "8.0.0", "9.0.0", "10.0.0"} and base["candidate"] is None
            else validate_value(ReviewProposal, base["candidate"])
        )
        if version in {"7.0.0", "8.0.0", "9.0.0", "10.0.0"}:
            return build_accepted(state, candidate, reviewed_ids=base.get("reviewed_ids"))
        return build_accepted(state, candidate)
    if version == "5.0.0":
        from packages.evidence_scoring.current import build as build_current

        return build_current(base)
    if version == "1.0.0":
        return engine.build(base)
    if version == "2.0.0":
        return continuous.build(base)
    if version == "3.0.0":
        return participation.build(base)
    if version == "4.0.0":
        return interpretation.build(base)
    raise ValueError("research_score_unsupported_version")


def validate(base, scores):
    schema = scores.get("schema_version")
    if schema in {
        "case-research-scoring/6.0.0",
        "case-research-scoring/6.1.0",
        "case-research-scoring/7.0.0",
        "case-research-scoring/8.0.0",
        "case-research-scoring/9.0.0",
        "case-research-scoring/10.0.0",
        "case-research-scoring/11.0.0",
    }:
        expected = build(base, version=schema.split("/")[1])
        if expected != scores:
            raise ValueError("accepted_score_reconstruction")
        return expected
    if schema == "case-research-scoring/5.0.0":
        from packages.evidence_scoring.current import validate as validate_current

        return validate_current(base, scores)
    if schema == "case-research-scoring/1.0.0":
        return engine.validate(base, scores)
    if schema == "case-research-scoring/2.0.0":
        return continuous.validate(base, scores)
    if schema == "case-research-scoring/3.0.0":
        return participation.validate(base, scores)
    if schema == "case-research-scoring/4.0.0":
        return interpretation.validate(base, scores)
    raise ValueError("research_score_unsupported_schema")
