"""Saved real text and injected wire only; never sends new HTTP or edits old evidence."""

import copy
import json
from pathlib import Path

import pytest

from agents.provider_agents.batched_review import read_progress
from apps.vectorengine_single_case import load_config
from packages.evidence_details.review_batches import (
    ReviewBatchProposal,
    read_response,
    response_candidate,
    validate_batch_receipts,
)
from packages.grounding.models import digest, validate
from packages.llm_provider.provider import ProviderError
from packages.llm_provider.review_response import decode
from tests.agent_provider.batch_support import batch_reply, fixture, generate
from tests.agent_provider.support import Documents

CONFIG = "configs/agents/vectorengine_batch_shape.json"
OLD = "configs/agents/vectorengine_batch_citations.json"
LIVE = Path("docs/acceptance/12-completion-cde-quote-live-20261001")


def actual():
    rows = json.loads((LIVE / "actual-requests.json").read_text())
    return rows[-1]["raw_response"][0]["raw_model_text"]


def redundant_reply(body):
    value = batch_reply(body)
    for row in value["items"]:
        if row["status"] in {"input_missing", "observation_ungradable"}:
            row["observation"] = dict(
                value=None,
                reason="Synthetic missing observation",
                missing_reason="Absent",
                method="not_assessed",
            )
    return value


def test_real_response_old_contract_still_rejects_new_exact_adapter_preserves_all_values():
    raw = actual()
    with pytest.raises(ValueError, match="extra_forbidden"):
        decode(raw, load_config(OLD))
    candidate, proof = decode(raw, load_config(CONFIG))
    assert proof.redundant_observation_method_indices == (0, 1, 2)
    expected = json.loads(raw)
    for row in expected["items"]:
        assert row["observation"].pop("method") == "not_assessed"
    assert candidate.model_dump(mode="json") == expected
    # Original business owner, on its original frozen material, also accepts these rows.
    from packages.evidence_details.acceptance_models import DomainState
    from packages.evidence_scoring.accepted import rows_from_candidate

    documents = json.loads((LIVE / "original-case-documents.json").read_text())
    intent = next(
        v[0]
        for k, v in documents.items()
        if k.startswith("review:")
        and k.endswith(":intent")
        and v[0]["state"]["material"]["scope"]["graph"]["code"]
        == "ASSET:OID_41CF3670AA8D3E87CF92C5034651C749FEDADC609A2C99C9"
    )
    state = validate(DomainState, intent["state"])
    rows_from_candidate(state, candidate, batch_ids=tuple(r.assessment_id for r in candidate.items))
    assert actual() == raw  # No rewrite of the real failed record.


@pytest.mark.parametrize(
    "mutation",
    [
        "method",
        "zero",
        "available",
        "status",
        "reason",
        "judgment",
        "unknown",
        "relation",
        "duplicate",
        "version",
    ],
)
def test_non_equivalent_or_unknown_fields_never_silently_removed(mutation):
    value = json.loads(actual())
    row = value["items"][0]
    if mutation == "method":
        row["observation"]["method"] = "fixed_model_margin"
    elif mutation == "zero":
        row["observation"]["value"] = 0
    elif mutation == "available":
        row["observation_available"] = True
    elif mutation == "status":
        row["status"] = "reference_missing"
    elif mutation == "reason":
        row["observation"]["missing_reason"] = ""
    elif mutation == "judgment":
        row["judgment"]["value"] = 0
    elif mutation == "unknown":
        row["observation"]["arbitrary"] = "not ignored"
    elif mutation == "relation":
        row["relation"]["method"] = "not_assessed"
    elif mutation == "duplicate":
        value["items"].append(copy.deepcopy(row))
    else:
        value["schema_version"] = "unknown"
    with pytest.raises(ValueError):
        decode(json.dumps(value), load_config(CONFIG))


def test_duplicate_json_key_and_old_schema_remain_strict():
    with pytest.raises(ValueError, match="duplicate_json_key"):
        decode('{"items":[],"items":[]}', load_config(CONFIG))
    with pytest.raises(ValueError):
        validate(ReviewBatchProposal, json.loads(actual()))


def test_provider_receipts_original_acceptance_score_and_recovery_are_consistent():
    f = fixture(replies=[redundant_reply] * 3, config_path=CONFIG)
    parent = f.owner.current()
    sub = generate(f)
    checked = validate_batch_receipts(parent, sub, f.documents, f.budget)
    assert len(checked) == f.budget.used == 3
    responses = [
        b["response"][0] for b in read_progress(f.documents, "synthetic-batches")["batches"]
    ]
    assert all(r["schema_version"] == "review-batch-response/1.1.0" for r in responses)
    assert any(r["alignment"]["redundant_observation_method_indices"] for r in responses)
    f.acceptance.submit(sub, key="aligned")
    assert f.acceptance.accept(key="aligned").status == "accepted"
    audit = list(f.documents.audit)
    assert generate(f) == sub and f.budget.used == 3
    assert f.acceptance.accept(key="aligned").status == "accepted"
    assert f.documents.audit == audit
    assert all(r.judgment.value is None for r in sub.candidate.items if not r.observation_available)


@pytest.mark.parametrize("suffix", [":attempt:", ":response", ":result", ":aggregate"])
def test_alignment_saved_confirmation_lost_does_not_resend(suffix):
    class Lost(Documents):
        once = True

        def write(self, key, values):
            super().write(key, values)
            if self.once and suffix in key:
                self.once = False
                raise OSError("saved_confirmation_lost")

    f = fixture(replies=[redundant_reply] * 3, documents=Lost(), config_path=CONFIG)
    with pytest.raises(OSError, match="saved_confirmation_lost"):
        generate(f)
    sub = generate(f)
    validate_batch_receipts(f.owner.current(), sub, f.documents, f.budget)
    assert f.budget.used == 3


@pytest.mark.parametrize("change", ["proof", "config", "raw", "score"])
def test_original_receipt_binding_rejects_tampered_alignment(change):
    f = fixture(replies=[redundant_reply] * 3, config_path=CONFIG)
    sub = generate(f)
    key = "review-batches:synthetic-batches:batch:000:response"
    raw = f.documents.saved[key][0]
    if change == "proof":
        raw["alignment"]["candidate_checksum"] = digest("wrong")
    elif change == "config":
        raw["provider_config"]["max_requests"] = 39
    elif change == "raw":
        raw["raw_model_text"] += " "
    else:
        raw["alignment"]["redundant_observation_method_indices"] = [4]
    with pytest.raises(ValueError, match="binding"):
        validate_batch_receipts(f.owner.current(), sub, f.documents, f.budget)


def test_bad_citation_still_rejected_after_format_alignment_no_retry():
    def wrong(body):
        r = redundant_reply(body)
        r["items"][0]["reference_citations"] = ["SOURCE:FOREIGN"]
        return r

    f = fixture(replies=[wrong], config_path=CONFIG)
    with pytest.raises(ProviderError, match="scope"):
        generate(f)
    assert f.budget.used == 1
    with pytest.raises(ProviderError, match="unknown_no_repeat"):
        generate(f)
    assert f.budget.used == 1


def test_injected_http_three_stage_end_to_end_and_wire_mapping(monkeypatch, tmp_path):
    from tests.expert_agent.batch_case_support import source_reply, vision_reply
    from tests.expert_agent.test_batch_assembly import assembly, stages

    f = assembly(
        monkeypatch,
        config_path=CONFIG,
        replies=[source_reply, vision_reply, *([redundant_reply] * 3)],
    )
    results, roles = stages(f, tmp_path)
    assert roles == ("source", "vision", "review")
    assert all(x["status"] == "accepted" for x in results)
    assert f.ports.budget.used == len(f.wire.calls) == 5
    assert "REVIEW_CITATION_CODES" in str(f.wire.calls[2:])
    assert "Rating shape contract" in str(f.wire.calls[2:])
    assert (tmp_path / "review-report.html").exists()


def test_response_requires_owner_configuration_not_caller_policy():
    f = fixture(replies=[redundant_reply] * 3, config_path=CONFIG)
    generate(f)
    response = read_response(
        read_progress(f.documents, "synthetic-batches")["batches"][0]["response"][0]
    )
    with pytest.raises(ValueError, match="configuration_binding"):
        response_candidate(
            response,
            config_checksum=digest(load_config(OLD)),
            prompt_checksum=f.owner.current().prompt_checksum,
        )


@pytest.mark.parametrize("value", [-0.4, 0.0])
def test_canonical_zero_and_negative_values_are_unchanged(value):
    def signed(body):
        r = redundant_reply(body)
        r["items"][0]["judgment"].update(value=value, lower=-0.6, upper=0.1)
        return r

    f = fixture(1, replies=[signed], config_path=CONFIG)
    sub = generate(f)
    assert sub.candidate.items[0].judgment.value == value
    validate_batch_receipts(f.owner.current(), sub, f.documents, f.budget)


@pytest.mark.parametrize("role", ["source", "vision"])
def test_source_vision_wire_is_byte_equivalent(role):
    from packages.llm_provider.config import prompt
    from packages.llm_provider.live_probes import PublicProbeMaterials
    from packages.llm_provider.models import SourceProposal, VisionProposal
    from packages.llm_provider.transport import wire_request

    old, new = load_config(OLD), load_config(CONFIG)
    material = PublicProbeMaterials("RUN:ANON", old).state.material
    payload = dict(role=role, material=material.model_dump(mode="json"))
    schema = (SourceProposal if role == "source" else VisionProposal).model_json_schema()
    bodies = []
    for config in (old, new):
        setting, text = prompt(config, role)
        bodies.append(
            wire_request(
                config.transport,
                setting,
                text,
                payload,
                schema,
                (),
                scope_binding=digest(material.scope),
                wire_policy=config.wire_policy,
                wire_mode=config.wire_mode,
            )
        )
    assert bodies[0] == bodies[1]
