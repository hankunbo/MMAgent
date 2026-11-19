"""Conservative proposal extraction, never medical adjudication or review approval."""

import re

from packages.contracts import StructuredObservation
from packages.object_store import checksum

from .models import ObservationCandidate, Projection, Span

PATTERN = re.compile(r"IOP (OD|OS) current: ([0-9]{1,2}(?:\.[0-9])?) mmHg\.?\s*")


def project_note(note):
    spans = []
    candidates = []
    offset = 0
    parts = note.splitlines(keepends=True)
    # Multiple measurements anywhere in the record have no established temporal alignment.
    measurement_count = sum("iop" in part.lower() for part in parts)
    # Retain every original character, including separators, in one contiguous mapping.
    for part in parts:
        end = offset + len(part)
        match = PATTERN.fullmatch(part)
        if measurement_count != 1:
            match = None
        lower = part.lower()
        assessment = any(
            word in lower for word in ("glaucoma", "diagnos", "assessment", "impression")
        )
        reason = "diagnosis_assessment" if assessment else "inseparable_or_ambiguous"
        category = "assessment" if assessment else "unclassified"
        lost = tuple(
            key
            for key, words in (
                ("IOP", ("iop", "pressure")),
                ("OCT", ("oct",)),
                ("RNFL", ("rnfl",)),
                ("VF", ("visual field", "hvf")),
                ("ANGLE", ("angle",)),
                ("CDR", ("cdr", "cup")),
            )
            if any(w in lower for w in words)
        )
        obs = None
        if match:
            reason = "review_pending"
            category = "objective_candidate"
            obs = StructuredObservation(
                concept_code="CONCEPT:IOP_RECORDED",
                observation="INCONCLUSIVE",
                value=float(match[2]),
                unit="mmHg",
                source_code="SOURCE:NOTE_CANDIDATE",
            )
        if lost:
            candidates.append(
                ObservationCandidate(
                    start=offset,
                    end=end,
                    laterality=({"OD": "RIGHT", "OS": "LEFT"}[match[1]] if match else "UNKNOWN"),
                    timepoint="current" if match else "unknown",
                    status="pending_review" if match else "ambiguous",
                    observation=obs,
                )
            )
        spans.append(
            Span(start=offset, end=end, category=category, reason=reason, lost_evidence=lost)
        )
        offset = end
    return Projection(
        raw_checksum=checksum(note.encode()),
        raw_length=len(note),
        spans=tuple(spans),
        candidates=tuple(candidates),
    )
