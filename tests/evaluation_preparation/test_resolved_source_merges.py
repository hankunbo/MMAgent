"""Formal regression for a non-applied merged Source decision."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.evidence_reuse.definition_acceptance import requirement_rows
from packages.evidence_reuse.definitions import OwnerBoundDefinitionProposal
from packages.grounding.models import canonical, digest
from packages.multimodal_research.definition_expansion import read_expansion_policy
from packages.multimodal_research.discovery_continuation import ConfirmedDefinitionContinuation
from packages.multimodal_research.gap_expansion import classify
from tests.evaluation_preparation.test_owner_dependency_definition import raw
from tests.evaluation_preparation.test_routed_source_backlog import routed_parent

POLICY = "retain-resolved-definition-merges/1.0.0"


def merged_parent(tmp_path):
    c, ref, spec, plan, failure, records, writes = routed_parent(tmp_path)
    parent = Path(ref["parent"])
    owner, manifest, _, _, _, value = raw()
    value.update(
        schema_version="source-definition-draft/6.1.0", target=spec["intent"]["target_code"]
    )
    for excerpt in value["excerpts"]:
        excerpt.pop("start", None)
        excerpt.pop("end", None)
    owner.binding.update(campaign_id=parent.name, target=value["target"])
    spec["input_checksum"] = plan["input"] = failure["input_checksum"] = manifest.checksum
    c.base.manifest = manifest
    proposal = OwnerBoundDefinitionProposal.model_validate_json(
        canonical(
            dict(
                schema_version="research-definition-proposal/5.0.0",
                definitions=[value],
                no_addition_reason=None,
            )
        )
    )
    draft = proposal.definitions[0]
    key = digest(draft)[7:47]
    receipt = owner.accept(key, draft)
    row = requirement_rows(receipt, manifest, fact_reader=owner.fact_reader)[0]
    row["id"] = "active:existing"
    rows = json.loads(canonical([row]))
    candidate = dict(
        rows=rows,
        found={},
        accepted={key: receipt},
        added=[],
        replaced=[],
        backlog=[],
        merged=[
            dict(
                definition=key,
                existing=row["id"],
                reason="existing_examination_requires_no_duplicate_vote",
                disposition="existing_definition_kept_new_relation_not_applied",
                interpretation_changed=True,
            )
        ],
        material_fingerprint=digest([]),
        evaluation_fingerprint=digest([]),
    )
    binding = dict(
        campaign_id=parent.name,
        input_checksum=manifest.checksum,
        target=value["target"],
        protocol_checksum=digest(plan),
        parent_requirements_checksum=digest(rows),
        policy_checksum=digest(
            read_expansion_policy(
                json.loads(Path(spec["active_acquisition"]["path"]).read_bytes())[
                    "definition_expansion"
                ]
            )
        ),
    )
    decision = dict(
        schema_version="definition-gap-attempt/1.0.0",
        attempt=0,
        binding=binding,
        direction={"query": "synthetic public question"},
        candidate=candidate,
        decision=dict(classify(candidate), action="try_distinct_public_direction"),
    )
    record = records["journal:1"][0]["record"]
    record.update(error=None, output_checksum=digest(proposal))
    records["definition:response:" + record["request_id"]][0]["raw_text"] = (
        proposal.model_dump_json()
    )
    records["definition:gap:0:decision"] = (decision,)
    records["definition:" + key] = (receipt,)
    failure["protocol_checksum"] = digest(plan)
    for name, val in (("entry", spec), ("protocol", plan), ("stage-failure", failure)):
        (parent / (name + ".json")).write_text(json.dumps(val))
    ref.update(
        entry=digest(spec),
        protocol=digest(plan),
        failure=digest(failure),
        resolved_decisions_policy=POLICY,
    )
    c.base.ledger = [row]
    return c, ref, records, writes, decision


def test_completed_merge_has_explicit_history_not_new_relation(tmp_path):
    c, ref, records, writes, decision = merged_parent(tmp_path)
    before = copy.deepcopy(records)
    owner = ConfirmedDefinitionContinuation(c, ref)
    assert owner.resolved_merges[0]["decision_checksum"] == digest(decision)
    assert owner.resolved_merges[0]["new_relation_applied"] is False
    assert owner.check_attempts()[0]["requests"] == 3
    assert ConfirmedDefinitionContinuation(c, ref, read_only=True).claim == owner.claim
    assert ConfirmedDefinitionContinuation(c, ref).claim == owner.claim
    assert len(writes) == 1 and all(records[k] == v for k, v in before.items())


def test_old_reference_cannot_silently_enable_new_disposition(tmp_path):
    c, ref, _, writes, _ = merged_parent(tmp_path)
    ref.pop("resolved_decisions_policy")
    with pytest.raises(ValueError, match="definition_successor_uncommitted_increment"):
        ConfirmedDefinitionContinuation(c, ref)
    assert not writes


@pytest.mark.parametrize(
    "field", ["added", "replaced", "rows", "receipt", "disposition", "policy", "target"]
)
def test_discarding_applied_or_unbound_content_never_authorized(tmp_path, field):
    c, ref, records, writes, decision = merged_parent(tmp_path)
    candidate = decision["candidate"]
    if field in ("added", "replaced"):
        candidate[field] = ["new-check"]
    elif field == "rows":
        candidate["rows"].append(dict(id="injected"))
    elif field == "disposition":
        candidate["merged"][0]["disposition"] = "apply_new_relation"
    elif field == "policy":
        decision["binding"]["policy_checksum"] = digest("wrong")
    elif field == "target":
        decision["binding"]["target"] = "GOAL:WRONG"
    else:
        key = next(iter(candidate["accepted"]))
        records.pop("definition:" + key)
    with pytest.raises(ValueError):
        ConfirmedDefinitionContinuation(c, ref)
    assert not writes


def test_new_deferral_keeps_baseline_and_history_without_applying_relation(tmp_path):
    from packages.multimodal_research.gap_expansion import verify_summary

    c, ref, records, _, decision = merged_parent(tmp_path)
    owner = ConfirmedDefinitionContinuation(c, ref)
    before = copy.deepcopy(records)
    c.definition_continuation = owner
    expansion = SimpleNamespace(
        original_rows=decision["candidate"]["rows"],
        control=lambda: None,
        binding={"new": "binding"},
        campaign=c,
        sources=SimpleNamespace(
            restore=lambda values: None if not values else pytest.fail("source installed")
        ),
        commit_revision=lambda *args, **kwargs: (args, kwargs),
    )
    args, kwargs = owner.defer_expansion(expansion, None)
    assert args[0] == expansion.original_rows
    assert args[1:5] == ({}, {}, [], [])
    assert args[5][0]["original_resolved_decision"] == owner.resolved_merges[0]
    assert kwargs["replaced"] == []
    verify_summary(expansion, expansion.gap_expansion_summary)
    assert records == before
    bad = dict(expansion.gap_expansion_summary, resolved_nonadopted_definitions=[])
    with pytest.raises(ValueError, match="definition_gap_deferral_binding"):
        verify_summary(expansion, bad)
    expansion.original_rows = [dict(id="different")]
    with pytest.raises(ValueError, match="definition_resolved_merge_current_baseline_changed"):
        owner.defer_expansion(expansion, None)


def test_changed_history_receipt_cannot_authorize_later_dispatch(tmp_path):
    c, ref, records, _, decision = merged_parent(tmp_path)
    owner = ConfirmedDefinitionContinuation(c, ref)
    key = next(iter(decision["candidate"]["accepted"]))
    records.pop("definition:" + key)
    with pytest.raises(ValueError, match="definition_successor_history_changed"):
        owner.check_attempts()


def test_explicit_policy_and_cli_never_default_to_enabled(monkeypatch):
    from apps.multimodal_case import main
    from packages.multimodal_research.resolved_source_decisions import policy_reference

    assert policy_reference({}, routed=True) == {}
    for reference, routed in (
        ({"resolved_decisions_policy": None}, True),
        ({"resolved_decisions_policy": POLICY}, False),
    ):
        with pytest.raises(ValueError, match="definition_resolved_merge_policy"):
            policy_reference(reference, routed=routed)
    calls = []
    monkeypatch.setattr(
        "packages.multimodal_research.entry.prepare_entry",
        lambda **kwargs: calls.append(kwargs) or {},
    )
    argv = [
        "command",
        "live-prepare",
        "--dataset",
        "gamma",
        "--root",
        "/not-used",
        "--sample",
        "0028",
        "--task",
        "advanced_glaucoma",
        "--output",
        "/not-used",
    ]
    monkeypatch.setattr("sys.argv", argv)
    main()
    assert calls[-1]["retain_resolved_definition_merges"] is False
    monkeypatch.setattr("sys.argv", [*argv, "--retain-resolved-definition-merges"])
    main()
    assert calls[-1]["retain_resolved_definition_merges"] is True


def test_real_prepare_and_cold_preflight_keep_explicit_policy(tmp_path, monkeypatch):
    from packages.multimodal_research import entry
    from tests.evaluation_preparation.test_vision_feedback_entry import (
        test_actual_prepare_preflight_provider_and_cold_binding as exercise,
    )

    original = entry.prepare_entry

    def prepare(**kwargs):
        if kwargs.get("definition_failure_of"):
            kwargs["retain_resolved_definition_merges"] = True
        result = original(**kwargs)
        if kwargs.get("definition_failure_of"):
            saved = json.loads((Path(kwargs["output"]) / "entry.json").read_bytes())
            assert saved["definition_failure_of"]["resolved_decisions_policy"] == POLICY
        return result

    monkeypatch.setattr(entry, "prepare_entry", prepare)
    exercise(tmp_path, monkeypatch, review_contract="rating-presence/1.1.0")
