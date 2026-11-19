"""Finite proposition grammar; differences are candidates, with no evidence voting."""

import re

from packages.knowledge_claims.models import ClinicalClaimCandidate

from .models import Comparison, SourceContext


def proposition(text):
    patterns = (
        (r"OCT result is (?:positive|negative)\.", "OCT result"),
        (r"OCT is (?:not )?recommended for (?:adults|children)\.", "OCT recommendation"),
        (
            r"OCT (?:supports|does not support|cannot support|may support|might support) "
            r"evaluation\.",
            "OCT evaluation",
        ),
        (
            r"OCT (?:does not support|cannot support|may support|might support) exclusion\.",
            "OCT exclusion",
        ),
        (r"Pressure .+\.", "Pressure measurement"),
        (r"OCT (dose|value|count) is .+\.", None),
    )
    for pattern, key in patterns:
        match = re.fullmatch(pattern, text)
        if match:
            return key or "OCT " + match.group(1)
    return text  # Exact equality only; no fuzzy similarity implies equivalence.


def compare(left, right, left_context=None, right_context=None):
    left = ClinicalClaimCandidate.model_validate(dict(vars(left)))
    right = ClinicalClaimCandidate.model_validate(dict(vars(right)))
    lc = SourceContext.model_validate(dict(vars(left_context or SourceContext())))
    rc = SourceContext.model_validate(dict(vars(right_context or SourceContext())))
    differences = []
    relations = []
    fields = (
        ("population", "POPULATION"),
        ("disease_stage", "STAGE"),
        ("laterality", "ANATOMY"),
        ("modality", "MODALITY"),
        ("exam", "METHOD"),
        ("device", "METHOD"),
        ("intervention", "METHOD"),
        ("condition", "POPULATION"),
        ("applicability", "POPULATION"),
        ("time", "TIME"),
        ("recommendation_strength", "STRENGTH"),
        ("outcome", "NON_EQUIVALENT"),
        ("claim_type", "UNDETERMINED"),
    )
    for name, relation in fields:
        if getattr(left, name) != getattr(right, name):
            differences.append(name)
            relations.append(relation)
    for name, relation in (("anatomy", "ANATOMY"), ("publication_date", "TIME")):
        if getattr(lc, name) != getattr(rc, name):
            differences.append(name)
            relations.append(relation)
    if tuple((n.value, n.comparator) for n in left.numbers) != tuple(
        (n.value, n.comparator) for n in right.numbers
    ):
        differences.append("threshold")
        relations.append("THRESHOLD")
    if tuple(n.unit for n in left.numbers) != tuple(n.unit for n in right.numbers):
        differences.append("unit")
        relations.append("UNIT")
    if left.polarity != right.polarity:
        differences.append("polarity")
    if left.uncertainty != right.uncertainty:
        differences.append("uncertainty")
    same = left.source_version == right.source_version
    lifecycle = next(
        (n for n in ("retraction", "correction", "update") if n in (lc.notice, rc.notice)), None
    )
    if lifecycle:
        relation = lifecycle.upper()
        differences.append("source_lifecycle")
        reason = (
            "Source notice requires offline reevaluation; it does not establish medical polarity."
        )
    elif left.subject != right.subject or "NON_EQUIVALENT" in relations:
        relation = "NON_EQUIVALENT"
        reason = (
            "Different subject or outcome; surface similarity does not establish one proposition."
        )
    elif proposition(left.target_proposition) != proposition(right.target_proposition):
        relation = "UNDETERMINED"
        differences.append("proposition")
        reason = "Equivalence is outside the finite grammar; offline review required."
    elif relations:
        relation = relations[0]
        reason = "Scope or measurement differs; coexistence needs conditions and review."
    elif "UNCERTAIN" in (left.polarity, right.polarity):
        relation = "UNDETERMINED"
        reason = "Source medical uncertainty is preserved; no automatic support or refutation."
    elif left.polarity != right.polarity:
        relation = "CONTRADICTION"
        reason = "Same finite proposition and scope, opposite polarity; candidate contradiction."
    else:
        relation = "SUPPORT"
        reason = (
            "Same finite proposition, polarity and scope; candidate support only, never a vote."
        )
    return Comparison(
        relation=relation,
        differences=tuple(dict.fromkeys(differences)),
        reason=reason,
        same_source_version=same,
    )
