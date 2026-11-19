"""Independent annotated regression oracles for the 04B5 policy repair."""

import pytest

from packages.knowledge_claims.evidence import Rejected, validate_output
from packages.knowledge_claims.models import StructuredOutput
from tests.knowledge_claims.oracle import annotated_output, cases
from tests.knowledge_claims.test_claims_04b5 import evidence


@pytest.mark.parametrize("case", cases(), ids=lambda c: c["id"])
def test_policy_red_abc(case):
    value = StructuredOutput.model_validate_json(annotated_output(case)).candidates[0]
    with pytest.raises(Rejected, match=case["reason"]):
        validate_output(value, evidence(case["text"]))


@pytest.mark.parametrize("case", cases("policy-expected.json"), ids=lambda c: c["id"])
def test_independent_policy_expected(case):
    value = StructuredOutput.model_validate_json(annotated_output(case)).candidates[0]
    if case["reason"]:
        with pytest.raises(Rejected, match="^" + case["reason"] + "$"):
            validate_output(value, evidence(case["text"]))
    else:
        assert validate_output(value, evidence(case["text"])) == value


@pytest.mark.parametrize(
    "case",
    [
        c
        for c in cases("policy-expected.json")
        if c["id"] in {"same-distinct", "same-different-units", "bare", "mu-g"}
    ],
    ids=lambda c: c["id"],
)
def test_occurrences_use_nonzero_canonical_offset(case):
    from dataclasses import replace
    from types import SimpleNamespace

    original = evidence(case["text"])
    position = original.block.position.model_copy(
        update={"span_start": 101, "span_end": 101 + len(case["text"])}
    )
    bound = replace(
        original,
        block=original.block.model_copy(update={"position": position}),
        chunk=SimpleNamespace(derivation=SimpleNamespace(position=position)),
    )
    value = StructuredOutput.model_validate_json(annotated_output(case, 101)).candidates[0]
    assert validate_output(value, bound) == value


def test_legacy_records_serialize_without_new_policy_markers():
    import json
    from pathlib import Path

    from db.knowledge.models import KnowledgeSnapshot, revalidate_record
    from packages.knowledge_index.models import canonical

    legacy = json.loads((Path(__file__).parent / "fixtures/legacy-claims-v1.json").read_text())
    for wire in legacy["records"]:
        snapshot = KnowledgeSnapshot.model_validate_json(json.dumps(wire))
        revalidate_record(snapshot.record)
        assert canonical(snapshot) == canonical(wire)
        assert "validation_policy" not in snapshot.model_dump_json()


@pytest.mark.parametrize("case", cases("policy-tables.json"), ids=lambda c: c["id"])
def test_independent_table_policy(case):
    from dataclasses import replace
    from types import SimpleNamespace

    from packages.knowledge_index.parsers import parse

    parsed = parse(("<article>" + case["xml"] + "</article>").encode(), "application/xml")
    block = parsed.document.blocks[0]
    assert block.text == case["text"]
    base = evidence(block.text)
    bound = replace(
        base,
        parsed=parsed.document,
        block=block,
        chunk=SimpleNamespace(derivation=SimpleNamespace(position=block.position)),
    )
    candidate = StructuredOutput.model_validate_json(annotated_output(case)).candidates[0]
    if case["reason"]:
        with pytest.raises(Rejected, match="^" + case["reason"] + "$"):
            validate_output(candidate, bound)
    else:
        assert validate_output(candidate, bound) == candidate


@pytest.mark.parametrize("label", ["POSITIVE", "NEGATIVE", "UNCERTAIN"])
def test_negated_uncertainty_has_no_automatic_label(label):
    case = {
        "text": "OCT may not support evaluation.",
        "polarity": label,
        "numbers": [],
        "fields": {"uncertainty": ["may"]},
    }
    value = StructuredOutput.model_validate_json(annotated_output(case)).candidates[0]
    with pytest.raises(Rejected, match="polarity_unresolved"):
        validate_output(value, evidence(case["text"]))


def test_validation_policy_manifest():
    import json
    from pathlib import Path

    from packages.knowledge_claims.policy import POLICY_CHECKSUM, SPEC
    from packages.knowledge_index.models import canonical, digest

    recorded = json.loads(Path("packages/knowledge_claims/validation_policy.json").read_text())
    assert canonical(recorded) == canonical(SPEC)
    assert digest(recorded) == POLICY_CHECKSUM


@pytest.mark.parametrize("comparator", ["<", "<=", ">", ">=", "=", "≤", "≥"])
def test_declared_comparator_surface_forms(comparator):
    quote = comparator + " 5 μg"
    case = {
        "text": "OCT dose is " + quote + ".",
        "polarity": "POSITIVE",
        "numbers": [
            {
                "value": "5",
                "unit": "μg",
                "comparator": comparator,
                "quote": quote,
                "span": {"start": 12, "end": 12 + len(quote)},
            }
        ],
    }
    value = StructuredOutput.model_validate_json(annotated_output(case)).candidates[0]
    assert validate_output(value, evidence(case["text"])) == value


@pytest.mark.parametrize("comparator", ["<<", "> =", "=<", "=>", "≈", "±", "less than"])
def test_unsupported_comparator_never_uses_partial_match(comparator):
    case = {"text": "OCT value is " + comparator + " 5.", "polarity": "POSITIVE", "numbers": []}
    value = StructuredOutput.model_validate_json(annotated_output(case)).candidates[0]
    with pytest.raises(Rejected, match="numeric_comparator_unsupported"):
        validate_output(value, evidence(case["text"]))


@pytest.mark.parametrize("word", ["and", "supports", "require", "for"])
def test_bare_number_delimiter_is_not_an_affirmative_unit(word):
    # Delimiters may separate quantities, but must not complete a measurement sentence.
    case = {
        "text": "OCT dose is 5 " + word + ".",
        "polarity": "POSITIVE",
        "numbers": [
            {
                "value": "5",
                "unit": None,
                "comparator": None,
                "quote": "5",
                "span": {"start": 12, "end": 13},
            }
        ],
    }
    value = StructuredOutput.model_validate_json(annotated_output(case)).candidates[0]
    with pytest.raises(Rejected, match="polarity_unresolved"):
        validate_output(value, evidence(case["text"]))
