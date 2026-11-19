"""Sources-only successors preserve old facts, request accounting and receipts."""

import copy
import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from db.runtime.repository.case_pipeline import identity
from packages.grounding.models import canonical, digest
from packages.multimodal_research import campaign_renewal as renewal
from packages.multimodal_research import source_checkpoint as recovery
from tests.evaluation_preparation.test_gap_expansion import fixture


def saved(tmp_path, monkeypatch, documents=None, configure=None):
    e, calls, materials = fixture(tmp_path, first_adopts=True, documents=documents)
    e.policy = e.policy.model_copy(update={"reserve_review_requests": 2})
    e.binding["policy_checksum"] = digest(e.policy)
    c = e.campaign
    c.output.mkdir(parents=True, exist_ok=True)
    c.protocol.update(
        schema_version=recovery.ORIGIN,
        mode="definitions",
        definition_increment_only=True,
        expansion_policy=e.policy.model_dump(mode="json"),
        seconds=3600,
        max_new_requests=8,
        max_cumulative_requests=40,
        previous_attempts=[dict(directory="synthetic-prior", requests=32, checksum=digest("old"))],
        pending_definition_acquisition={"targets": []},
    )
    c.base.definition_expansion_policy = c.protocol["expansion_policy"]
    e.binding["protocol_checksum"] = digest(c.protocol)
    c.config = e.provider.config
    c.request_key = lambda t: "checkpoint-synthetic:" + (t or "review")
    rid = identity("RUN", "task-research:" + c.request_key("definition-expansion"))
    c.documents = lambda _: e.documents
    c.base.binding = {"case_code": "CASE:SYNTHETIC_CHECKPOINT"}
    context = SimpleNamespace(
        run_id=rid,
        request=SimpleNamespace(
            input_checksum=c.base.manifest.checksum,
            task=c.base.task,
            scope=SimpleNamespace(case_code=c.base.binding["case_code"]),
            policy_ref=SimpleNamespace(checksum=digest((c.base.ledger, c.config, None))),
        ),
    )
    c.contexts = SimpleNamespace(
        read_research_run=lambda run, **kw: context if run == rid else None
    )
    c.sessions = SimpleNamespace(read=lambda _: None)
    c.targets = ()
    if not hasattr(e.documents, "read_many"):
        e.documents.read_many = lambda keys: tuple(e.documents.read(k) for k in keys)
    c.definition_expansion = e
    if configure is not None:
        configure(c, e, rid)
    checkpoint = e.advance(until="sources")
    receipt = dict(
        schema_version="multimodal-stage-receipt/1.0.0",
        stage="sources",
        run_id=rid,
        binding=digest(c.protocol),
        payload=checkpoint,
    )
    e.documents.write("multimodal-stage-operation:sources", (receipt,))
    e.documents.write(
        "journal:0",
        (
            dict(
                schema_version="agent-budget/1.0.0",
                binding=dict(run_id=rid, deadline=c.protocol["deadline"], max_requests=40),
            ),
        ),
    )
    (c.output / "stages").mkdir(exist_ok=True)
    (c.output / "stages/sources.json").write_bytes(canonical(receipt))
    (c.output / "protocol.json").write_bytes(canonical(c.protocol))
    later = datetime.fromisoformat(c.protocol["deadline"]) + timedelta(seconds=1)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return later

    monkeypatch.setattr(recovery, "datetime", Clock)
    monkeypatch.setattr(renewal, "assemble", lambda path: c)
    return c, e, calls, materials, later


def test_readonly_proof_new_window_retains_32_and_original_bytes(tmp_path, monkeypatch):
    c, e, _, _, later = saved(tmp_path, monkeypatch)
    before = copy.deepcopy(e.documents.saved)
    proof, checkpoint = recovery.inspect_checkpoint(c)
    plan = recovery.plan_for(
        c.protocol,
        proof,
        previous=c.output,
        output=tmp_path / "next",
        authorization="explicit-test",
        now=later,
        implementation={},
    )
    assert plan["previous_attempts"][-1]["requests"] == 0
    assert sum(a["requests"] for a in plan["previous_attempts"]) == 32
    assert plan["max_new_requests"] == 8 and plan["max_cumulative_requests"] == 40
    assert (
        plan["prior_deadline"] == c.protocol["deadline"]
        and plan["deadline"] != c.protocol["deadline"]
    )
    assert plan["expansion_policy"] == c.protocol["expansion_policy"]
    assert proof["checkpoint"] == digest(checkpoint) and e.documents.saved == before


@pytest.mark.parametrize(
    "fault", ["cancel", "revision", "reservation", "unknown_operation", "binding", "local_receipt"]
)
def test_no_checkpoint_claim_for_cancelled_unknown_or_changed_work(tmp_path, monkeypatch, fault):
    c, e, _, _, _ = saved(tmp_path, monkeypatch)
    if fault == "cancel":
        (c.output / "cancel").touch()
    elif fault == "revision":
        e.documents.write("definition:revision", ({"binding": e.binding},))
    elif fault == "reservation":
        e.documents.write("journal:1", (dict(phase="reserved", request_id="unknown"),))
    elif fault == "unknown_operation":
        del e.documents.saved[e.documents.saved["definition:operation:0"][0]["key"] + ":result"]
    elif fault == "binding":
        e.documents.saved["definition:operation:0"][0]["binding"] = {}
    else:
        receipt = json.loads((c.output / "stages/sources.json").read_bytes())
        receipt["run_id"] = "OTHER"
        (c.output / "stages/sources.json").write_bytes(canonical(receipt))
    with pytest.raises(ValueError, match="source_checkpoint_"):
        recovery.inspect_checkpoint(c)


def test_claim_confirmation_loss_recovers_exact_plan_not_new_deadline(tmp_path, monkeypatch):
    c, e, _, _, _ = saved(tmp_path, monkeypatch)
    original = e.documents.write

    def lose(key, values):
        original(key, values)
        if key == "authorized-campaign-renewal":
            raise OSError("synthetic lost claim confirmation")

    e.documents.write = lose
    output = tmp_path / "next"
    with pytest.raises(OSError, match="lost claim confirmation"):
        recovery.prepare(previous=c.output, output=output, authorization="test")
    assert not output.exists()
    e.documents.write = original
    saved_claim = e.documents.read("authorized-campaign-renewal")
    result = recovery.prepare(previous=c.output, output=output, authorization="test")
    assert result["previous_requests"] == 32 and result["api_requests"] == 0
    assert json.loads((output / "protocol.json").read_bytes()) == saved_claim[0]["plan"]
    assert recovery.prepare(previous=c.output, output=output, authorization="test") == result
    assert e.documents.read("authorized-campaign-renewal") == saved_claim
    with pytest.raises(ValueError, match="renewal_tip_already_claimed"):
        recovery.prepare(previous=c.output, output=tmp_path / "other", authorization="test")


def test_same_material_reaches_original_definition_owner_without_research(tmp_path, monkeypatch):
    from packages.multimodal_research.definition_expansion import DefinitionExpansion
    from tests.agent_provider.support import Documents

    c, e, calls, _, _ = saved(tmp_path, monkeypatch)
    checkpoint = e.documents.read("definition:stage:sources")[0]
    child = copy.copy(c)
    child.protocol = dict(
        c.protocol,
        schema_version=recovery.VERSION,
        source_checkpoint={"checkpoint": digest(checkpoint)},
    )
    child.logical_campaign_id = e.binding["campaign_id"]
    child.output = tmp_path / "child"
    child.output.mkdir()
    child.base = copy.copy(c.base)
    child.usage = lambda: e.budget.used
    monkeypatch.setattr(recovery, "recheck", lambda _: (c, checkpoint))
    new = DefinitionExpansion(
        child,
        documents=Documents(),
        sources=e.sources,
        provider=e.provider,
        budget=e.budget,
        scope=e.scope,
        policy=e.policy.model_dump(mode="json"),
    )
    before = len(calls)
    result = new.advance(until="definitions")
    assert result["accepted"] and len(calls) == before and e.budget.used == 1
    assert any(r.get("reason") == "confirmed_source_checkpoint_reused" for r in result["rejected"])
    used = e.budget.used
    assert new.advance(until="definitions") == result and e.budget.used == used
