"""Current terminal owner is reread; saved plan alone cannot authorize survey reuse."""

import json
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.memory.scope_archive import survey_origin
from packages.multimodal_research import input_scope, scope_successor
from packages.multimodal_research.campaign_renewal import RenewedCampaign
from packages.multimodal_research.capability_window import CURRENT_VERSION, ROUTED_VERSION


@pytest.mark.parametrize("fault", [None, "result", "cancel", "pending", "task"])
def test_bound_parent_scope_rechecks_original_owner(tmp_path, monkeypatch, fault):
    entry = dict(target="input-scope:one", task={"component_id": "image"})
    fact = dict(
        phase=dict(target=entry["target"]), task=entry["task"], observation={"checked": True}
    )
    result = dict(phases=[dict(run_id="RUN:REVIEW")])
    (tmp_path / "result.json").write_text(json.dumps(result))
    control = SimpleNamespace(phase="stopped", pending=None, cancelled=False)
    protocol = {"input": digest("image")}
    saved = {"multimodal-case-result": (result,), "dynamic-campaign": (protocol,)}
    parent = SimpleNamespace(
        output=tmp_path,
        protocol=protocol,
        scope_targets=(entry["target"],),
        documents=lambda _: SimpleNamespace(read=lambda k: saved.get(k)),
        sessions=SimpleNamespace(read=lambda _: control),
        base=SimpleNamespace(input_scope_plan={"entries": [entry]}),
    )
    reads = []

    def read(owner, *, require_all):
        assert owner is parent and require_all
        reads.append(True)
        return [fact]

    monkeypatch.setattr(input_scope, "read_results", read)
    if fault == "result":
        saved["multimodal-case-result"] = ({"changed": True},)
    elif fault == "cancel":
        control.cancelled = True
    elif fault == "pending":
        control.pending = object()
    elif fault == "task":
        fact = dict(fact, task={"component_id": "other"})
    before = digest(protocol)
    if fault:
        with pytest.raises(ValueError, match="scope_current_reuse_"):
            scope_successor.current_proof(parent, {"entries": [entry]})
    else:
        proof = scope_successor.current_proof(parent, {"entries": [entry]})
        assert proof["reused"] == [
            dict(target=entry["target"], fact=fact, fact_checksum=digest(fact))
        ]
        assert reads == [True] and digest(protocol) == before
        changed = dict(entry, task={"component_id": "other"})
        assert scope_successor.current_proof(parent, {"entries": [changed]})["reused"] == []


@pytest.mark.parametrize("version", [CURRENT_VERSION, ROUTED_VERSION])
def test_current_survey_direct_redispatch_rejected(version):
    campaign = SimpleNamespace(
        protocol=dict(
            schema_version=version,
            input_scope=dict(targets=["input-scope:one"], dispatch_targets=[]),
        )
    )
    with pytest.raises(ValueError, match="reused_target_dispatched"):
        RenewedCampaign.execute_phase(campaign, "input-scope:one")


@pytest.mark.parametrize("fault", [None, "input", "result", "fact"])
@pytest.mark.parametrize("version", [CURRENT_VERSION, ROUTED_VERSION])
def test_archive_follows_exact_parent_protocol_not_new_case_alias(fault, version):
    entry = dict(target="input-scope:one", task={"component_id": "image"})
    phase = dict(run_id="RUN:SURVEY", target=entry["target"], accepted=True)
    fact = dict(phase=phase, disease_votes=0)
    parent = dict(
        input=digest("input"),
        task=digest("task"),
        input_scope={"provider_checksum": digest("provider")},
        capability_transition={"scope_plan": {"entries": [entry]}},
    )
    result = dict(input_scopes=[fact])
    proof = dict(
        parent_run_id="RUN:PARENT",
        parent_protocol_checksum=digest(parent),
        parent_result_checksum=digest(result),
        reused=[dict(target=entry["target"], fact=fact, fact_checksum=digest(fact))],
    )
    protocol = dict(
        schema_version=version,
        input=parent["input"],
        task=parent["task"],
        current_scope_reuse=proof,
    )
    if fault == "input":
        protocol["input"] = digest("wrong")
    elif fault == "result":
        result = dict(input_scopes=[])
    elif fault == "fact":
        proof["reused"][0]["fact_checksum"] = digest("wrong")

    def read(key, *, source_run):
        if key == "dynamic-campaign":
            return [parent]
        if key == "multimodal-case-result":
            return [result]
        if key == "dynamic-phase-result":
            return [phase]
        raise AssertionError(key)

    if fault:
        with pytest.raises(ValueError, match="memory_scope_current_reuse_binding"):
            survey_origin(read, protocol, entry, fact)
    else:
        assert (
            survey_origin(read, protocol, entry, fact) == parent["input_scope"]["provider_checksum"]
        )
