"""B02 request boundary: synthetic text only; no medical content approval."""

import json
from pathlib import Path

import pytest

from packages.knowledge_claims.evidence import Rejected
from packages.knowledge_claims.ingestion import RULE_CONFIG_CHECKSUM, parse_candidate_request
from packages.knowledge_claims.models import CandidateRequest, CandidateSubmission, RuleExecution
from packages.knowledge_claims.ports import RuleStructuredExtractor
from tests.knowledge_claims.support import output


def request_data():
    def ref(kind):
        return dict(
            kind=kind, code="SYNTHETIC:BOUND", version="1.0.0", checksum="sha256:" + "a" * 64
        )

    return dict(
        request_key="synthetic-only",
        rule_config_checksum=RULE_CONFIG_CHECKSUM,
        source=dict(
            index_key="sha256:" + "b" * 64,
            source=ref("source"),
            source_version=ref("source_version"),
            document=ref("source_document"),
            chunk=ref("document_chunk"),
            original_checksum="sha256:" + "c" * 64,
            parse_checksum="sha256:" + "d" * 64,
            chunk_checksum="sha256:" + "e" * 64,
            parse_uri="object://knowledge/objects/" + "0" * 32,
            fidelity_policy="text-fidelity/2.0.0",
            position=dict(section_path=[], kind="paragraph", ordinal=1, span_start=0, span_end=32),
        ),
        candidate=json.loads(output("OCT is not recommended for adults."))["candidates"][0],
        display_zh="合成软件测试说明，未审核。",
    )


def test_local_rule_is_implemented_not_fake_or_llm():
    request = parse_candidate_request(json.dumps(request_data()).encode())
    adapter = RuleStructuredExtractor(request, producer="offline-fixture-producer")
    assert adapter.spec.method == "RULE" and adapter.spec.model is None
    assert adapter.execution == RuleExecution() and adapter.live is False
    with pytest.raises(Rejected, match="candidate_source_binding"):
        adapter.extract(
            {"evidence": {"text": "wrong", "span_start": 0, "span_end": 1}},
            timeout=1,
            max_output_tokens=256,
        )


@pytest.mark.parametrize(
    "field", ["source_verified", "extraction_audit", "batch", "synthetic", "producer"]
)
def test_untrusted_receipts_and_flags_are_not_request_fields(field):
    data = request_data()
    data[field] = True
    with pytest.raises(Rejected, match="candidate_schema"):
        parse_candidate_request(json.dumps(data).encode())


@pytest.mark.parametrize(
    "text",
    [
        "This patient reports pain.",
        "患者姓名：虚构测试",
        "%25" * 20 + "4d%25" * 20 + "52",
        r"\u004d\u0052\u004e",
        "MRN: SYNTHETIC-ONLY",
        "x" * 2001,
    ],
)
def test_patient_encoded_and_oversized_display_rejected(text):
    data = request_data()
    data["display_zh"] = text
    with pytest.raises(Rejected):
        parse_candidate_request(json.dumps(data).encode())


@pytest.mark.parametrize(
    "raw",
    [
        b"x" * 32769,
        b"[" * 1500 + b"]" * 1500,
        b'{"request_key":"a","request_key":"b"}',
        b'{"x":' + b"[" * 16 + b"0" + b"]" * 16 + b"}",
    ],
)
def test_wire_limits_duplicate_keys_and_deep_json(raw):
    with pytest.raises(Rejected):
        parse_candidate_request(raw)


@pytest.mark.parametrize("mutation", ["version", "config", "extra", "type", "nested"])
def test_constructed_models_revalidated(mutation):
    request = parse_candidate_request(json.dumps(request_data()).encode())
    updates = {
        "version": {"rule_version": "python-eval/1.0.0"},
        "config": {"rule_config_checksum": "sha256:" + "0" * 64},
        "extra": {"source_verified": True},
        "type": {"request_key": 12},
        "nested": {"source": request.source.model_copy(update={"chunk_checksum": "wrong"})},
    }
    with pytest.raises(Rejected):
        parse_candidate_request(request.model_copy(update=updates[mutation]))


def test_schema_snapshots():
    for model in (CandidateRequest, CandidateSubmission, RuleExecution):
        path = Path("packages/knowledge_claims/schemas") / (model.__name__ + ".schema.json")
        assert json.loads(path.read_text()) == model.model_json_schema()
