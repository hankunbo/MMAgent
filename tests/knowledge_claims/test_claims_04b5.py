import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from packages.knowledge_claims.evidence import Evidence, Rejected, validate_grid, validate_output
from packages.knowledge_claims.models import BatchPolicy, ClinicalClaimCandidate, StructuredOutput
from packages.knowledge_claims.ports import Budget, ConfiguredStructuredExtractor, load_llm_config
from packages.knowledge_index.models import Block, Position
from tests.knowledge_claims.support import output


def evidence(text):
    from types import SimpleNamespace

    p = Position(
        section_path=("test",), kind="paragraph", ordinal=1, span_start=0, span_end=len(text)
    )
    return Evidence(
        SimpleNamespace(derivation=SimpleNamespace(position=p)),
        None,
        None,
        None,
        Block(text=text, position=p),
        text,
        {},
    )


@pytest.mark.parametrize(
    "text",
    [
        "OCT supports evaluation.",
        "OCT is not recommended for adults.",
        "If adults have symptoms, consider OCT.",
        "Pressure >= 21 mmHg supports evaluation.",
        "Children may benefit; insufficient evidence remains.",
        "Adults aged 40 years require device A.",
        "OCT may not support exclusion.",
        "OCT不推荐，证据不足。",
    ],
)
def test_extracts_faithful_claim(text):
    value = StructuredOutput.model_validate_json(output(text)).candidates[0]
    # These legacy fixtures contain mixed/unclear scope; v2 must now decline them.
    if text in {
        "Children may benefit; insufficient evidence remains.",
        "OCT may not support exclusion.",
        "OCT不推荐，证据不足。",
    }:
        with pytest.raises(Rejected, match="polarity_unresolved"):
            validate_output(value, evidence(text))
    else:
        assert validate_output(value, evidence(text)) == value


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("quote", "Invented source.", "quote_missing"),
        ("statement", "OCT diagnoses glaucoma.", "unsupported_fact"),
        ("polarity", "POSITIVE", "polarity_changed"),
        ("population", [], "scope_lost"),
        ("exam", ["MRI"], "unsupported_fact"),
        ("recommendation_strength", [], "strength_lost"),
        ("span", {"start": 0, "end": 500}, "quote_span"),
    ],
)
def test_rejects_added_or_lost_facts(field, value, reason):
    text = "OCT is not recommended for adults."
    data = json.loads(output(text))
    data["candidates"][0][field] = value
    parsed = StructuredOutput.model_validate_json(json.dumps(data)).candidates[0]
    with pytest.raises(Rejected, match=reason):
        validate_output(parsed, evidence(text))


@pytest.mark.parametrize(
    "field,value",
    [
        ("polarity", "MAYBE"),
        ("confidence", "0.8"),
        ("claim_type", "diagnosis"),
        ("hidden_reasoning", "x"),
        ("subject", 1),
        ("status", "published"),
    ],
)
def test_strict_schema(field, value):
    data = json.loads(output("OCT supports evaluation."))
    data["candidates"][0][field] = value
    with pytest.raises(ValidationError):
        StructuredOutput.model_validate_json(json.dumps(data))


def test_numbers_units_comparators_cannot_be_synthesized():
    text = "Pressure >= 21 mmHg."
    data = json.loads(output(text))
    for field, value in (("unit", "mg"), ("value", "25"), ("comparator", "<")):
        candidate = json.loads(json.dumps(data))
        candidate["candidates"][0]["numbers"][0][field] = value
        parsed = StructuredOutput.model_validate_json(json.dumps(candidate)).candidates[0]
        with pytest.raises(Rejected, match="numeric_fragment"):
            validate_output(parsed, evidence(text))


def test_budget_retries_are_charged_and_deadline_enforced():
    policy = BatchPolicy(
        max_requests=2, max_input_chars=20, max_output_tokens=256, total_seconds=10
    )
    now = [0]
    budget = Budget(policy, clock=lambda: now[0])
    assert budget.reserve(10, 120) == 10
    assert budget.reserve(10, 120) == 10
    with pytest.raises(Rejected, match="budget_exhausted"):
        budget.reserve(1, 120)
    now[0] = 11
    with pytest.raises(Rejected, match="budget_exhausted"):
        budget.reserve(0, 1)


def test_config_from_project_and_environment_without_secret_resolution():
    config = load_llm_config(environ={})
    assert config.api_key_env == "OPENAI_API_KEY"
    other = load_llm_config(environ={"MMAGENT_LLM_MODEL": "fixture-model"})
    assert other.model == "fixture-model"
    assert ConfiguredStructuredExtractor(config).spec.configuration_checksum
    with pytest.raises(ValidationError):
        load_llm_config(environ={"MMAGENT_LLM_BASE_URL": "https://user:pass@example.org"})


def test_schema_snapshots():
    for cls in (StructuredOutput, ClinicalClaimCandidate):
        path = Path("packages/knowledge_claims/schemas") / (cls.__name__ + ".schema.json")
        assert json.loads(path.read_text()) == cls.model_json_schema()


def test_grid_rejects_missing_and_invented_headers():
    from packages.knowledge_index.models import digest
    from packages.knowledge_index.parsers import parse

    parsed = parse(
        b"<html><body><table><tr><th>Test</th><th>Limit</th></tr>"
        b"<tr><td>OCT</td><td>21 mmHg</td></tr></table></body></html>",
        "text/html",
    )
    block = parsed.document.blocks[0]
    validate_grid(block)
    cells = list(block.table.cells)
    cells[-1] = cells[-1].model_copy(update={"headers": ((9, 9),)})
    grid = block.table.model_copy(update={"cells": tuple(cells)})
    cap = block.position.table_capability.model_copy(update={"checksum": digest(grid)})
    corrupt = block.model_copy(
        update={
            "table": grid,
            "position": block.position.model_copy(update={"table_capability": cap}),
        }
    )
    with pytest.raises(Rejected, match="header_coordinates"):
        validate_grid(corrupt)


@pytest.mark.parametrize("kind", ["unit", "comparator", "number", "uncertainty", "condition"])
def test_qualifier_omission_is_rejected(kind):
    text = "If adults may have pressure >= 21 mmHg, consider OCT."
    data = json.loads(output(text))
    c = data["candidates"][0]
    if kind == "unit":
        n = c["numbers"][0]
        n["unit"] = None
        n["quote"] = ">= 21"
        n["span"]["end"] -= 5
    elif kind == "comparator":
        n = c["numbers"][0]
        n["comparator"] = None
        n["quote"] = "21 mmHg"
        n["span"]["start"] += 3
    elif kind == "number":
        c["numbers"] = []
    elif kind == "uncertainty":
        c["uncertainty"] = []
    else:
        c["condition"] = []
    value = StructuredOutput.model_validate_json(json.dumps(data)).candidates[0]
    with pytest.raises(Rejected):
        validate_output(value, evidence(text))


def test_cross_paragraph_number_unit_rejected():
    text = "Pressure 21\nmmHg OCT."
    data = json.loads(output(text))
    n = data["candidates"][0]["numbers"][0]
    n.update(unit="mmHg", quote="21\nmmHg")
    n["span"]["end"] += 5
    value = StructuredOutput.model_validate_json(json.dumps(data)).candidates[0]
    with pytest.raises(Rejected, match="cross_fragment_number"):
        validate_output(value, evidence(text))


@pytest.mark.parametrize(
    "mutation", ["checksum", "rows", "coordinates", "span", "numeric", "reason"]
)
def test_grid_integrity_rejects_each_invalid_layer(mutation):
    from packages.knowledge_index.models import digest
    from packages.knowledge_index.parsers import parse

    block = parse(
        b"<article><table><tr><th>OCT</th><th>Limit</th></tr>"
        b"<tr><td>OCT</td><td>21 mmHg</td></tr></table></article>",
        "application/xml",
    ).document.blocks[0]
    grid, cap = block.table, block.position.table_capability
    if mutation == "checksum":
        cap = cap.model_copy(update={"checksum": "sha256:" + "0" * 64})
    elif mutation == "rows":
        cap = cap.model_copy(update={"rows": 3})
    elif mutation in ("coordinates", "span"):
        cells = list(grid.cells)
        cells[-1] = cells[-1].model_copy(
            update={"row": 5} if mutation == "coordinates" else {"span_start": 0}
        )
        grid = grid.model_copy(update={"cells": tuple(cells)})
        cap = cap.model_copy(update={"checksum": digest(grid)})
    elif mutation == "numeric":
        cap = cap.model_copy(update={"numeric_evidence": False})
    else:
        cap = cap.model_copy(update={"reason": "unsupported"})
    block = block.model_copy(
        update={
            "table": grid,
            "position": block.position.model_copy(update={"table_capability": cap}),
        }
    )
    with pytest.raises(Rejected):
        validate_grid(block)


@pytest.mark.parametrize(
    "kind",
    ["exam_requirement", "evidence_direction", "exclusion", "applicability", "validation", "stop"],
)
def test_all_internal_claim_types_have_explicit_schema(kind):
    text = "OCT supports evaluation."
    value = StructuredOutput.model_validate_json(output(text, claim_type=kind)).candidates[0]
    assert validate_output(value, evidence(text)).claim_type == kind


def test_configured_extractor_uses_hard_process_timeout(monkeypatch):
    import subprocess

    from packages.knowledge_claims.ports import ConfiguredStructuredExtractor

    def timed_out(*args, **kwargs):
        assert kwargs["timeout"] == 0.01
        raise subprocess.TimeoutExpired("synthetic worker", 0.01)

    monkeypatch.setattr(subprocess, "run", timed_out)
    extractor = ConfiguredStructuredExtractor(load_llm_config(environ={}))
    with pytest.raises(Rejected, match="extractor_timeout"):
        extractor.extract({}, timeout=0.01, max_output_tokens=256)


def test_http_worker_missing_key_fails_without_stderr(monkeypatch):
    import io
    import sys

    from packages.knowledge_claims import worker

    config = load_llm_config(environ={})
    monkeypatch.delenv(config.api_key_env, raising=False)
    request = json.dumps(
        {"config": config.model_dump(mode="json"), "request": {}, "max_output_tokens": 256}
    ).encode()
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(request)))
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    assert worker.main() == 1
    assert sys.stdout.getvalue() == sys.stderr.getvalue() == ""
