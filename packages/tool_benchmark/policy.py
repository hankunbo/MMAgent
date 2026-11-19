"""Readiness only. ToolOnboardingDecision remains the sole shared decision contract."""

from packages.contracts import ToolOnboardingDecision
from packages.grounding.models import validate

from .models import BenchmarkError, BenchmarkManifest, EvaluationPolicy


def readiness(policy, document):
    policy = validate(EvaluationPolicy, policy)
    document = validate(BenchmarkManifest, document)
    reasons = []
    if policy.baseline_only or None in (
        policy.min_dice,
        policy.max_failure_rate,
        policy.min_samples,
    ):
        reasons.append("policy_pending")
    reserved = [s for s in document.samples if s.split == "reserved"]
    if not reserved:
        reasons.append("frozen_evaluation_set_missing")
    if any(s.prior_exposure != "unseen_attested" for s in reserved):
        reasons.append("prior_exposure_unverified")
    if any(s.group_basis == "unlinked_image_only" for s in reserved):
        reasons.append("patient_grouping_unverified")
    reasons.extend(
        (
            "formal_evaluator_not_implemented",
            "confidence_interval_not_implemented",
            "gates_5_6_pending",
        )
    )
    return {
        "formal_evaluation_allowed": False,
        "reasons": tuple(reasons),
        "required_gates": policy.required_gates,
    }


def read_onboarding_decision(value):
    """Compatibility reader, never a grant or catalog writer.

    No real decision is fabricated when implementation/catalog identities or Gate
    evidence are unavailable. Activation still belongs to the later controlled owner.
    """
    decision = validate(ToolOnboardingDecision, value)
    if decision.visible_to_current_run:
        raise BenchmarkError("new_tool_cannot_enter_old_run")
    return decision
