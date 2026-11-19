"""Research hypothesis comparison, reusing the frozen within-target kernel."""

from packages.evidence_scoring.continuous import summarize


def summarize_hypotheses(rows, policy, **removal):
    result = summarize([])
    result["confidence"]["missing_reason"] = (
        "互斥命题分别计分，不将各命题贡献加成一个疾病方向分；请查看命题比较。"
    )
    result["hypotheses"] = [
        dict(
            hypothesis=h.model_dump(mode="json"),
            summary=summarize([r for r in rows if r.target == h.code], **removal),
        )
        for h in policy.hypotheses
    ]
    result["assessment_ids"] = sorted(r.id for r in rows)
    result["contributing_ids"] = sorted(
        key for h in result["hypotheses"] for key in h["summary"]["contributing_ids"]
    )
    result["distinct_evidence_units"] = sorted({policy.evidence_units[r.id] for r in rows})
    return result


def decision(summary):
    """Descriptive ordering, NOT a fitted classifier or probability normalization."""
    values = [(h["hypothesis"], h["summary"]["confidence"]) for h in summary["hypotheses"]]
    available = [(h, s) for h, s in values if s["value"] is not None]
    ordered = sorted(available, key=lambda x: (-x[1]["value"], x[0]["code"]))
    best = ordered[0][1]["value"] if ordered else None
    tied = [h for h, s in ordered if abs(s["value"] - best) <= 1e-12]
    leading = tied[0] if len(tied) == 1 else None
    # Bounds describe the kernel's sensitivity, not a statistical confidence interval.
    separated = (
        bool(leading)
        and all(ordered[0][1]["lower"] > other["upper"] for _, other in ordered[1:])
        and len(available) == len(values)
    )
    return dict(
        schema_version="research-hypothesis-comparison/1.0.0",
        status="unavailable" if not ordered else "ranked_sensitive" if leading else "tied",
        leading_hypothesis=leading["label"] if leading else None,
        leading_evaluation_label=leading["evaluation_label"] if leading else None,
        tied_hypotheses=[h["label"] for h in tied],
        sensitivity_ranges_separated=separated,
        ranking=[dict(label=h["label"], score=s["value"]) for h, s in ordered],
        class_label=None,
        calibrated_probabilities=None,
        clinical_grade=None,
        note="仅记录未校准研究支持分的排序；不以最高分或分数区间代替临床分级。",
    )
