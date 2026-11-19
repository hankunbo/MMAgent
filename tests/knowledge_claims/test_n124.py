"""07G exact N124 syntax tests, no medical approval or formal source-chain claim."""

import json
from dataclasses import replace

import pytest
from pydantic import ValidationError

from packages.knowledge_claims import build_n124_request, parse_candidate_request
from packages.knowledge_claims.evidence import Rejected, validate_output
from packages.knowledge_claims.ingestion import validate_candidate
from packages.knowledge_claims.models import DraftOutput
from packages.knowledge_claims.n124 import (
    BODY,
    POLICY_VERSION,
    QUOTE,
    RULE_CONFIG,
    RULE_CONFIG_CHECKSUM,
)
from packages.knowledge_claims.policy import PolicyRejected, occurrences, polarity
from packages.knowledge_index.models import canonical, digest
from tests.knowledge_claims.n124_support import SOURCES, unit_evidence


def test_new_version_full_fields_spans_and_determinism():
    e = unit_evidence()
    a = build_n124_request(e, request_key="unit-n124")
    assert a == build_n124_request(e, request_key="unit-n124")
    assert parse_candidate_request(canonical(a)) == a
    assert validate_candidate(a, e) == a.candidate
    c = a.candidate
    assert c.statement == c.target_proposition == c.quote == QUOTE
    assert c.metadata.semantic_text == BODY and c.numbers == () and c.population == ()
    full = "\n".join(b.text for b in e.parsed.blocks)
    for text, span in (
        (c.quote, c.span),
        (BODY, c.metadata.semantic_span),
        (c.metadata.update_text, c.metadata.update_span),
        (c.metadata.examples_text, c.metadata.examples_span),
        ("1.2.4", c.metadata.recommendation_span),
    ):
        assert full[span.start : span.end] == text
    scope = c.metadata.source_scope
    assert (
        "\n".join(b.text for b in e.source_scope.parsed.blocks)[scope.span.start : scope.span.end]
        == scope.quote
    )
    assert scope.age.span.start == 567 and scope.age.value == "18"
    assert scope.assessment == "pending_review"
    assert digest(RULE_CONFIG | {"version": "nice-ng81-n124/1.0.1"}) != RULE_CONFIG_CHECKSUM


@pytest.mark.parametrize("name", ["N124", "N123", "N111", "N121", "N131"])
def test_old_rejections_preserved(name):
    text = (SOURCES / "fragments" / ("F-P1A-" + name + ".txt")).read_text()
    with pytest.raises(PolicyRejected, match="polarity_unresolved"):
        polarity(text)
    with pytest.raises(PolicyRejected):
        occurrences(text)


def test_old_validator_cannot_silently_select_n124():
    e = unit_evidence()
    c = build_n124_request(e, request_key="old").candidate
    old = DraftOutput.model_validate({k: getattr(c, k) for k in DraftOutput.model_fields})
    with pytest.raises(Rejected):
        validate_output(old, e)
    with pytest.raises(ValidationError):
        validate_output(c, e)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_version",
        "wrong_version",
        "config",
        "year",
        "number_span",
        "semantic_span",
        "scope_span",
        "scope_age",
        "scope_removed",
        "approved",
        "extra",
        "encoded",
        "illegal",
    ],
)
def test_request_and_field_mutations_fail_closed(mutation):
    e = unit_evidence()
    data = json.loads(build_n124_request(e, request_key="bad").model_dump_json())
    c = data["candidate"]
    m = c["metadata"]
    if mutation == "missing_version":
        data.pop("schema_version")
    elif mutation == "wrong_version":
        data["rule_version"] = "nice-ng81-n124/1.0.1"
    elif mutation == "config":
        data["rule_config_checksum"] = digest(RULE_CONFIG | {"version": "caller"})
    elif mutation == "year":
        m["update_text"] = "[2009, amended 2026]"
    elif mutation == "number_span":
        m["recommendation_span"]["start"] += 1
    elif mutation == "semantic_span":
        m["semantic_span"]["end"] -= 1
    elif mutation == "scope_span":
        m["source_scope"]["span"]["start"] += 1
    elif mutation == "scope_age":
        m["source_scope"]["age"]["value"] = "17"
    elif mutation == "scope_removed":
        m.pop("source_scope")
    elif mutation == "approved":
        m["source_scope"]["assessment"] = "approved"
    elif mutation == "extra":
        data["approved"] = True
    elif mutation == "encoded":
        data["display_zh"] = "%25" * 20 + "4d%25" * 20 + "52"
    else:
        data["display_zh"] = "\x00invalid"
    with pytest.raises((Rejected, ValidationError)):
        validate_candidate(parse_candidate_request(canonical(data)), e)


@pytest.mark.parametrize(
    "old,new",
    [
        ("at diagnosis", "after diagnosis"),
        ("for baseline documentation", "for diagnosis"),
        ("Obtain", "Do not obtain"),
        ("(for example,", "("),
        (" or OCT", " and OCT"),
        ("image at", "image 21 mmHg at"),
        ("2009", "2010"),
    ],
)
def test_changed_source_text_never_inherits_positive(old, new):
    e = unit_evidence()
    altered = replace(e, original=e.original.replace(old.encode(), new.encode()))
    with pytest.raises(Rejected, match="n124_source_snapshot"):
        build_n124_request(altered, request_key="variant")


@pytest.mark.parametrize("field", ["condition", "disease_stage", "outcome", "time"])
def test_semantic_field_omission_rejected_even_with_rehash(field):
    e = unit_evidence()
    request = build_n124_request(e, request_key="omission")
    request = request.model_copy(
        update={"candidate": request.candidate.model_copy(update={field: ()})}
    )
    with pytest.raises(Rejected, match="n124_exact_field_binding"):
        validate_candidate(parse_candidate_request(request), e)


@pytest.mark.parametrize(
    "variant",
    ["source", "scope", "parsed", "scope_parsed", "missing_scope", "n123", "n111", "n121", "n131"],
)
def test_source_scope_and_actual_parser_binding(variant):
    e = unit_evidence()
    if variant == "source":
        e = replace(e, original=b"wrong source")
    elif variant == "scope":
        e = replace(
            e,
            source_scope=replace(
                e.source_scope, original=e.source_scope.original.replace(b"18", b"17")
            ),
        )
    elif variant == "parsed":
        e = replace(e, parsed=e.parsed.model_copy(update={"title": "rehashed fake"}))
    elif variant == "scope_parsed":
        e = replace(
            e,
            source_scope=replace(
                e.source_scope, parsed=e.source_scope.parsed.model_copy(update={"title": "fake"})
            ),
        )
    elif variant == "missing_scope":
        e = replace(e, source_scope=None)
    else:
        number = "1." + variant[2] + "." + variant[3]
        block = next(
            b
            for b in e.parsed.blocks
            if b.position.kind == "paragraph" and b.position.section_path[-1:] == (number,)
        )
        e = replace(e, block=block, text=block.text)
    with pytest.raises(Rejected):
        build_n124_request(e, request_key="unsupported")


def test_explicit_policy_version_required():
    e = unit_evidence()
    c = build_n124_request(e, request_key="policy").candidate
    assert validate_output(c, e, policy_version=POLICY_VERSION) == c
    with pytest.raises(Rejected, match="validation_policy_binding"):
        validate_output(c, e, policy_version="claim-validation/99.0.0")


def test_new_manifests_match_registered_spec_and_preserve_old_rule_hash():
    from pathlib import Path

    from packages.knowledge_claims.ingestion import RULE_CONFIG_CHECKSUM as legacy
    from packages.knowledge_claims.n124 import POLICY_CHECKSUM, SPEC

    root = Path("packages/knowledge_claims")
    assert canonical(json.loads((root / "n124_validation_policy.json").read_text())) == canonical(
        SPEC
    )
    assert digest(SPEC) == POLICY_CHECKSUM
    assert canonical(json.loads((root / "n124_rule_config.json").read_text())) == canonical(
        RULE_CONFIG
    )
    assert legacy != RULE_CONFIG_CHECKSUM
