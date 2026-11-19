"""Task-relative partial visibility and source-backed scheduling. No disease priors."""

import json

from packages.grounding.models import validate
from packages.llm_provider.models import PartialObservation


def evaluate(observation):
    """V is multiplied once by the original scorer. No invented unseen disease."""
    data = json.loads(observation.value)
    if data.get("schema_version") != "partial-visual-observation/1.0.0":
        raise ValueError("partial_observation_version")
    detail = validate(PartialObservation, {k: data[k] for k in PartialObservation.model_fields})
    available = observation.status == "obtained" and detail.coverage > 0
    # Local positive findings need local clarity; absence must also account for
    # how much of the target was inspected. Coverage is not applied a second time in O.
    factor = detail.clarity * (1.0 if detail.signal == "present" else detail.coverage)
    return dict(
        availability_score=float(available),
        inspected_fraction=detail.coverage,
        local_clarity=detail.clarity,
        effective_quality=factor if available else 0.0,
        gap_score=1.0 - detail.coverage * detail.clarity,
        additional_observation_priority=1.0 - detail.coverage * detail.clarity,
        basis=detail.basis,
        basis_detail=detail.basis_detail,
        coverage_origin="vlm_estimate_not_pixel_measurement",
        unseen_disease_probability=None,
        unseen_reason="no_patient_observation_or_calibrated_prior",
        signal=detail.signal,
        status=observation.status,
        clinical_calibration=False,
    )


def schedule(requirements, *, measured=(), attempted=(), modalities=("SLO",)):
    """Enumerate every requirement, not a fixed list of four Vision IDs."""
    plan = []
    for r in requirements:
        missing = sorted(set(r.input_modalities) - set(modalities))
        if missing:
            action, reason = "record_input_gap", "missing:" + ",".join(missing)
        elif r.id in measured:
            action, reason = "review_measurement", "confirmed_current_case_output"
        elif r.id in attempted:
            action, reason = "retain_attempt", "no_automatic_repeat"
        elif r.definition.interpretation_status != "installed":
            action, reason = "record_definition_gap", "no_installed_interpretation"
        elif r.id == "descriptive:DISC_SIZE_CONTEXT":
            action, reason = "record_input_gap", "physical_scale_not_provided"
        else:
            action, reason = "observe_partial", "source_backed_SLO_requirement"
        plan.append(
            dict(
                id=r.id,
                action=action,
                reason=reason,
                sources=[s.model_dump(mode="json") for s in r.definition.source_refs],
            )
        )
    return plan
