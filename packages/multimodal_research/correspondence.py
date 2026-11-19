"""Correspondence eligibility is metadata, not a third independent evidence vote."""


def describe(manifest):
    rows = []
    components = [c for c in manifest.components if c.status in {"obtained", "partial"}]
    for index, first in enumerate(components):
        for second in components[index + 1 :]:
            if first.modality == second.modality:
                continue
            reasons = []
            if first.laterality == "UNKNOWN" or second.laterality == "UNKNOWN":
                reasons.append("laterality_unbound")
            elif first.laterality != second.laterality:
                reasons.append("different_eye")
            if first.visit != second.visit:
                reasons.append("different_visit_not_simultaneous")
            rows.append(
                dict(
                    first=first.id,
                    second=second.id,
                    same_case_input_checksum=manifest.checksum,
                    same_eye_visit_eligible=not reasons,
                    reasons=reasons,
                    spatial_registration=False,
                    independent_contribution=False,
                    interpretation="same_exam_context_only_no_anatomical_agreement_vote",
                )
            )
    return dict(
        schema_version="multimodal-correspondence/1.0.0",
        pairs=rows,
        new_score_rows=0,
        clinical_validation=False,
    )
