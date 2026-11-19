import re

from packages.knowledge_claims.evidence import NEGATIVE, NUMBER, STRENGTH, UNCERTAIN
from packages.knowledge_claims.models import DraftOutput, NumericEvidence, Span, StructuredOutput


def output(text, start=0, **updates):
    uncertain = tuple(m.group() for m in UNCERTAIN.finditer(text))
    numbers = []
    for match in NUMBER.finditer(text):
        tail = text[match.end() :]
        unit = re.match(r" *(mmHg|years|年|mg|%)", tail)
        end = match.end() + (unit.end() if unit else 0)
        left = text[: match.start()]
        comp = re.search(r"(<=|>=|<|>|=|≤|≥) *$", left)
        begin = comp.start() if comp else match.start()
        numbers.append(
            NumericEvidence(
                value=match.group(),
                unit=unit[1] if unit else None,
                comparator=comp[1] if comp else None,
                quote=text[begin:end],
                span=Span(start=start + begin, end=start + end),
            )
        )
    data = dict(
        claim_type="evidence_direction",
        statement=text,
        target_proposition=text,
        polarity="UNCERTAIN" if uncertain else "NEGATIVE" if NEGATIVE.search(text) else "POSITIVE",
        uncertainty=uncertain,
        recommendation_strength=tuple(m.group() for m in STRENGTH.finditer(text)),
        subject=text.split()[0],
        population=(text,) if re.search(r"adult|children|patient", text, re.I) else (),
        disease_stage=(),
        condition=(text,) if re.search(r"\bif\b|\bwhen\b|\bonly\b", text, re.I) else (),
        intervention=(),
        exam=(),
        modality=(),
        outcome=(),
        time=(),
        laterality=(),
        device=(),
        applicability=(),
        numbers=tuple(numbers),
        quote=text,
        span=Span(start=start, end=start + len(text)),
        confidence=0.8,
    )
    # Population is bounded to literal terms, not an entire long paragraph.
    data["population"] = tuple(
        m.group() for m in re.finditer(r"\b(?:adults?|children|patients)\b", text, re.I)
    )
    data["condition"] = tuple(
        m.group() for m in re.finditer(r"\b(?:if|when|only|unless)\b", text, re.I)
    )
    data.update(updates)
    return StructuredOutput(candidates=(DraftOutput(**data),)).model_dump_json().encode()
