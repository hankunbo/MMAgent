import copy
import json
from datetime import UTC, datetime, timedelta

import pytest

from agents.expert_agent.reporting import read_research_report
from packages.evaluation_preparation import report_production as production
from packages.evaluation_preparation.evidence_tree_v3 import build, render
from packages.evidence_details import accepted_aggregation as a
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.grounding.models import digest
from packages.multimodal_research import campaign_renewal as renewal
from packages.multimodal_research import context_review as context
from tests.evaluation_preparation.test_null_review_owner_chain import combined


def saved_report(value):
    f = combined(value)
    permission, sub = a.AggregationPermission.prepare(
        owner=f.owner, documents=f.docs, requests=f.requests, config=f.new.config
    )
    service = ResearchAcceptance(
        owner=f.owner, documents=f.docs, requests=f.requests, live_permission=permission
    )
    service.submit(sub, key=a.ACCEPTANCE_KEY)
    assert service.accept(key=a.ACCEPTANCE_KEY).status == "accepted"
    state = f.owner.current()
    detail = read_research_report(service, key=a.ACCEPTANCE_KEY, parent=state)
    report = dict(
        schema_version="dynamic-case-report/1.0.0",
        result=dict(
            case_code=state.material.scope.case_code,
            sample="synthetic-null-revision",
            execution_kind="local_accepted_review_aggregation_not_new_model_review",
            review_revision_policy="new_context_accepted_null_revision/1.0.0",
            review_revisions=[r.model_dump(mode="json") for r in f.source.null_revisions],
        ),
        original_review=detail,
        scores=detail["scores"],
        requirements=[
            dict(
                id=r.id,
                examination=r.definition.examination,
                modality=list(r.input_modalities),
                status="source_backed",
                sources=[ref.model_dump(mode="json") for ref in r.definition.source_refs],
            )
            for r in state.policy.requirements
        ],
    )
    return f, report, state


@pytest.mark.parametrize("value", [0.03, 0.0, None])
def test_accepted_revision_saved_json_six_sections_and_second_html_conversion(value):
    f, raw, state = saved_report(value)
    before = copy.deepcopy(raw)
    produced = production.produce(raw, state)
    restored = json.loads(json.dumps(produced))
    assert raw == before and produced["scores"] == raw["scores"]
    p = restored["review_revision_provenance"][f.target]
    assert p["current_judgment"]["value"] == value
    assert p["added_evidence_votes"] == 0 and p["previous_records_preserved"]
    row = next(r for r in restored["requirements"] if r["id"] == f.target)
    assert row["presentation_sections"]["rationale"]["accepted_review_revision"] == p
    assert len(produced["requirements"]) == len(raw["requirements"])
    packet = build(restored, input_checksum=digest(restored))
    page = render(restored, input_checksum=digest(restored))
    assert "accepted_review_revision" in json.dumps(packet)
    assert f.source.null_revisions[0].replacement_run_id in page
    assert "补充方法上下文" in page
    assert f.old_sub.candidate.items != f.new_sub.candidate.items


@pytest.mark.parametrize("fault", ["item", "policy", "missing_acceptance", "duplicate"])
def test_display_does_not_claim_unbound_revision(fault):
    f, report, state = saved_report(0.03)
    if fault == "item":
        report["result"]["review_revisions"][0]["replacement_item_checksum"] = digest("wrong")
    elif fault == "policy":
        report["result"]["review_revision_policy"] = "unsupported"
    elif fault == "missing_acceptance":
        report["original_review"]["receipt"]["status"] = "rejected"
    else:
        report["result"]["review_revisions"] *= 2
    with pytest.raises(ValueError, match="report_review_revision_"):
        production.produce(report, state)


def test_runtime_dispatches_new_context_profile_and_keeps_closure_reserve(tmp_path, monkeypatch):
    out = tmp_path / "new"
    out.mkdir()
    protocol = dict(
        schema_version=context.AGGREGATE_METHOD_VERSION,
        deadline=(datetime.now(UTC) + timedelta(seconds=5400)).isoformat(),
    )
    (out / "protocol.json").write_text(json.dumps(protocol))
    seen = []
    token = object()
    monkeypatch.setattr(
        context, "assemble", lambda output, **kwargs: seen.append((output, kwargs)) or token
    )
    assert renewal.assemble(out) is token
    assert seen[0][0] == out
    c = object.__new__(renewal.RenewedCampaign)
    c.protocol = protocol
    c._final_review_active = True
    assert renewal.RenewedCampaign.work_deadline(c) < datetime.fromisoformat(protocol["deadline"])
    c.protocol["execution_lease_profile"] = "invalid"
    with pytest.raises(ValueError, match="context_review_execution_lease_profile"):
        renewal.RenewedCampaign.execution_session(c)


def assert_cold_context_assembly(tmp_path, monkeypatch, runtime_target):
    from tests.evaluation_preparation.test_null_review_context import persisted_origin, planning

    f, parent, result, tip = persisted_origin()
    synthetic, _, _, _, config = planning(count=1)
    target = f.target
    parent.base = synthetic.base
    parent.base.output = tmp_path / "old"
    parent.base.manifest.sample = "synthetic"
    parent.base.manifest.components = ()
    parent.base.inputs = object()
    parent.base.views = {"views": []}
    parent.base.ledger = []
    parent.base.review_context_scope = "old-retained-scope"
    parent.base.confirmed_review_reader = object()
    # Public synthetic transport fixture; no actual network or secret loader.
    parent.config = config
    parent.target = runtime_target
    parent.protocol = {"deadline": "old"}
    parent.accepted_observations = lambda: {}
    parent.documents = lambda _: f.docs
    tip.update(path=str(tmp_path / "old"), previous_attempts=[dict(requests=26)])
    contexts = {
        target: dict(
            unit_basis="synthetic method context",
            source_limits=["synthetic"],
            transform={"kind": "unchanged"},
            numeric={"value": 87},
            interpretation_limits=["no normative comparator"],
            owner_method={"version": "synthetic-new"},
        )
    }
    monkeypatch.setattr(context, "completed_parent", lambda *args, **kwargs: (parent, result))
    monkeypatch.setattr(context, "stopped_tip", lambda *args, **kwargs: tip)
    monkeypatch.setattr(context, "collect", lambda *args, **kwargs: contexts)
    monkeypatch.setattr(context, "freeze", lambda: {})
    proof = context.origin_proof(parent, result, tip, metadata_profile=context.RNFL_UNIT_PROFILE)
    out = tmp_path / "context-successor"
    out.mkdir()
    plan = context.plan_for(
        parent,
        result,
        tip,
        contexts,
        proof,
        config=parent.config,
        output=out,
        authorization="synthetic",
        started=datetime.now(UTC),
        metadata_profile=context.RNFL_UNIT_PROFILE,
    )
    f.docs.write(context.claim_key(plan), (dict(output=str(out), plan_checksum=digest(plan)),))
    (out / "protocol.json").write_text(json.dumps(plan))
    assembled = renewal.assemble(out)
    assert isinstance(assembled, renewal.RenewedCampaign)
    assert assembled.config == parent.config
    assert assembled.base.confirmed_review_reader is None
    assert assembled.base.review_context_scope is None
    assert assembled.base.review_target_ids == (target,)
    assert assembled.targets == () and assembled.reused_review_count == 0
    assert parent.base.review_context_scope == "old-retained-scope"
    cold = renewal.assemble(out)
    assert cold.protocol == assembled.protocol and cold.base.review_display_context == contexts
    wrong = copy.deepcopy(plan)
    wrong["max_new_requests"] += 1
    (out / "protocol.json").write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="context_review_owner_binding"):
        renewal.assemble(out)
