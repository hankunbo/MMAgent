import sys
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from packages.evidence_details import accepted_aggregation as a
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.grounding.models import digest
from packages.multimodal_research import context_review as c
from packages.multimodal_research.review_context import supplementary
from tests.evaluation_preparation.test_http_client_entry import public_config
from tests.evaluation_preparation.test_null_review_owner_chain import combined


def planning(count=5, prior=26):
    config = public_config()
    contexts = {
        "check:rnfl:" + str(i): dict(
            transform={"kind": "unchanged"},
            numeric={"value": 87},
            owner_method={"version": "synthetic-new-unit-context"},
            unit_basis="synthetic protocol documentation",
            source_limits=["No normative comparator"],
            interpretation_limits=["Unit context does not supply disease direction"],
        )
        for i in range(count)
    }
    parent = SimpleNamespace(
        base=SimpleNamespace(
            manifest=SimpleNamespace(checksum=digest("input")),
            task=SimpleNamespace(checksum=digest("task")),
        ),
        config=config,
        protocol={"deadline": "unchanged-prior"},
        accepted_observations=lambda: {},
    )
    tip = dict(path="original-aggregate", previous_attempts=[dict(requests=prior)])
    proof = dict(
        schema_version="aggregate-null-review-origin/1.0.0",
        null_ids=sorted(contexts),
        context_refs={k: [] for k in contexts},
    )
    return parent, tip, contexts, proof, config


def plan(f):
    parent, tip, contexts, proof, config = f
    return c.plan_for(
        parent,
        {},
        tip,
        contexts,
        proof,
        config,
        output="new",
        authorization="synthetic",
        started=datetime.now(UTC),
        metadata_profile=c.RNFL_UNIT_PROFILE,
    )


def test_bounded_selected_review_counts_batches_preserves_budget_config_and_old_facts():
    f = planning()
    p = plan(f)
    assert p["schema_version"] == c.AGGREGATE_METHOD_VERSION
    assert p["max_new_requests"] == 3 and p["max_model_dispatches"] == 2
    assert p["seconds"] == 5400 and p["max_cumulative_requests"] == 40
    assert p["previous_attempts"] == f[1]["previous_attempts"]
    assert p["review_target_ids"] == sorted(f[2]) and p["reuse_assessment_ids"] == []
    assert p["provider"] == digest(f[4]) and p["new_vision_requests"] == 0
    assert p["new_public_search_allowed"] is False
    assert p["old_accepted_items_not_replayed"] is True
    assert p["execution_lease_profile"] == "original-runtime-case-pipeline-session/1.0.0"


@pytest.mark.parametrize("fault", ["not_null", "same_context", "budget", "too_many", "empty"])
def test_no_dispatch_plan_for_unqualified_repeated_or_excess_scope(fault):
    f = planning(
        count=7 if fault == "too_many" else 0 if fault == "empty" else 5,
        prior=38 if fault == "budget" else 26,
    )
    if fault == "not_null":
        f[3]["null_ids"] = []
    if fault == "same_context":
        f[3]["context_refs"] = {
            t: [o.reference.model_dump(mode="json") for o in supplementary({t: v})]
            for t, v in f[2].items()
        }
    with pytest.raises(ValueError, match="context_review_"):
        plan(f)


def persisted_origin():
    f = combined(None)
    permission, sub = a.AggregationPermission.prepare(
        owner=f.owner, documents=f.docs, requests=f.requests, config=f.new.config
    )
    service = ResearchAcceptance(
        owner=f.owner, documents=f.docs, requests=f.requests, live_permission=permission
    )
    service.submit(sub, key=a.ACCEPTANCE_KEY)
    assert service.accept(key=a.ACCEPTANCE_KEY).status == "accepted"
    state = f.owner.current()
    manifest = dict(
        run_id=f.docs.run_id, state_checksum=digest(state), submission_checksum=digest(sub)
    )
    result = dict(
        schema_version="multimodal-accepted-aggregation-result/1.0.0",
        complete=True,
        phases=[
            dict(run_id=f.docs.run_id, acceptance_key=a.ACCEPTANCE_KEY, accepted=True, target=None)
        ],
    )
    f.docs.write("accepted-aggregation-manifest", (manifest,))
    f.docs.write("multimodal-case-result", (result,))
    f.docs.write("dynamic-phase-input", (state,))
    parent = SimpleNamespace(
        sessions=f.docs.sessions,
        aggregation_baseline=SimpleNamespace(
            documents=f.docs,
            state=state,
            submission=sub,
            manifest=manifest,
            acceptance_key=a.ACCEPTANCE_KEY,
        ),
    )
    tip = dict(
        run_id=f.docs.run_id,
        protocol_kind="accepted_aggregation_manifest",
        protocol=digest(manifest),
        result=digest(result),
    )
    return f, parent, result, tip


def test_original_accepted_aggregate_null_inventory_and_current_material_binding():
    f, parent, result, tip = persisted_origin()
    proof = c.origin_proof(parent, result, tip, metadata_profile=c.RNFL_UNIT_PROFILE)
    assert f.target in proof["null_ids"]
    assert proof["submission_checksum"] == digest(parent.aggregation_baseline.submission)
    assert proof["context_refs"][f.target]
    assert c.method_version(c.RNFL_UNIT_PROFILE) == c.AGGREGATE_METHOD_VERSION


@pytest.mark.parametrize("fault", ["tip", "acceptance", "cancelled"])
def test_aggregate_proof_never_uses_unbound_saved_success(fault):
    f, parent, result, tip = persisted_origin()
    if fault == "tip":
        tip["result"] = digest("wrong")
    elif fault == "acceptance":
        f.docs.saved.pop("accepted:" + a.ACCEPTANCE_KEY)
    else:
        control = parent.sessions.read(f.docs.run_id)
        parent.sessions = SimpleNamespace(
            read=lambda _: control.model_copy(update={"cancelled": True})
        )
    with pytest.raises(ValueError):
        c.origin_proof(parent, result, tip, metadata_profile=c.RNFL_UNIT_PROFILE)


def test_cli_accepts_only_explicit_new_metadata_profile(monkeypatch, capsys):
    from apps.multimodal_case import main

    calls = []
    monkeypatch.setattr(c, "prepare", lambda **kw: calls.append(kw) or {"new_api": 0})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "case",
            "context-review-prepare",
            "--previous",
            "old",
            "--output",
            "new",
            "--assessment-id",
            "check:rnfl:0",
            "--authorization",
            "synthetic",
            "--metadata-profile",
            c.RNFL_UNIT_PROFILE,
        ],
    )
    assert main() == 0 and calls[0]["metadata_profile"] == c.RNFL_UNIT_PROFILE
    assert '"new_api": 0' in capsys.readouterr().out
