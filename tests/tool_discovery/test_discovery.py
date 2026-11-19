"""New offline query/provenance boundaries, no database or actual models."""

import time

import pytest

from packages.grounding.models import digest, validate
from packages.tool_discovery import Discovery, DiscoveryPolicy, DiscoveryResult, Provenance
from packages.tool_discovery.discovery import inspect, query
from tests.tool_gap_queue.support import setup

from .support import record, source


@pytest.fixture
def h(tmp_path):
    return setup(tmp_path)


def evaluate(h, sources, policy=None):
    d = Discovery(gap_queue=h.service, repository=None, sources=sources, policy=policy)
    q = query(h.service.prepare_research(h.context.run_id, key="discovery"), d.policy)
    plan = dict(
        query=q.model_dump(mode="json"),
        sources=list(d.bindings()),
        policy=d.policy.model_dump(mode="json"),
    )
    return d._evaluate("TASK:SYNTHETIC:DISCOVERY", plan, deadline=time.monotonic() + 10)


def test_query_preserves_original_contract_and_no_patient_data(h):
    q = query(h.service.prepare_research(h.context.run_id, key="q"), DiscoveryPolicy())
    assert q.contracts[0].output == h.request.candidate.output.model_dump(mode="json")
    assert q.contracts[0].inputs[0].model_dump(mode="json") == h.request.candidate.inputs[
        0
    ].model_dump(mode="json")
    assert q.purpose == h.request.purpose
    assert "patient_hash" not in q.model_dump_json() and "asset_code" not in q.model_dump_json()


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"repository_owner": "impostor"}, "repository_owner_changed"),
        ({"revision_kind": "branch"}, "mutable_revision"),
        ({"code_checksum": digest("changed")}, "code_checksum_changed"),
        ({"weights_checksum": digest("changed")}, "weights_checksum_changed"),
        ({"uploads_patient_data": True}, "patient_upload_required"),
        ({"arbitrary_remote_code": True}, "arbitrary_remote_code_required"),
        ({"requires_gpu": True}, "resource_limit"),
        ({"identity_ref": None}, "official_identity_unverified"),
    ],
)
def test_hard_filters_are_metadata_reasons_not_machine_gate(h, changes, reason):
    p = record(h, **changes)
    result = evaluate(h, (source([p]),))
    assert result.status == "incomplete" and not result.selected
    assert reason in result.candidates[0].rejection_codes
    assert result.candidates[0].machine_gate is None


@pytest.mark.parametrize(
    "field", ["modality", "scope_terms", "coordinate_space", "laterality", "artifact_ref"]
)
def test_exact_output_semantics_no_implicit_mapping(h, field):
    p = record(h)
    semantics = p.semantics.model_dump(mode="json")
    semantics["output"][field] = "synthetic-incompatible"
    p = validate(Provenance, {**p.model_dump(mode="json"), "semantics": semantics})
    result = evaluate(h, (source([p]),))
    assert "task_contract_mismatch" in result.candidates[0].rejection_codes


def test_missing_metadata_never_fills_checksum_license_or_release(h):
    p = record(h, code_checksum=None, weights_checksum=None, code_license=None, revision=None)
    c = evaluate(h, (source([p]),)).candidates[0]
    assert c.disposition == "incomplete"
    assert {"code_checksum", "weights_checksum", "code_license", "revision"} <= set(c.missing)
    assert c.records[0].code_checksum is None and c.onboarding_job is None
    with pytest.raises(ValueError):
        record(h, code_checksum="unknown checksum")


def test_mirrors_deduplicate_only_complete_identity_with_all_provenance(h):
    records = [record(h, name="mirror-a"), record(h, name="mirror-b")]
    result = evaluate(h, (source(records),))
    assert result.status == "discovered" and len(result.candidates) == len(result.selected) == 1
    assert len(result.candidates[0].records) == 2 and result.records_examined == 2
    assert "approved_threshold_policy" in result.candidates[0].missing
    assert result.candidates[0].dimensions["public_evaluation"] is None
    assert validate(DiscoveryResult, result) == result


def test_fixed_priority_and_configurable_topk(h):
    local = record(h, name="local")
    official = record(h, kind="official_model", name="official", preprocessing="other provenance")
    # Different semantics cannot be conflated; differing immutable evidence is retained per mirror.
    result = evaluate(
        h,
        (source([official], name="official", kind="official_model"), source([local])),
        DiscoveryPolicy(top_k=1),
    )
    assert result.candidates[0].source_priority == 0
    assert result.candidates[0].records[0].source_kind == "local_contract"
    assert len(result.selected) == 1


def test_empty_technical_failure_and_budget_are_distinct(h):
    assert evaluate(h, ()).status == "no_candidates"
    broken = source([], technical_failure=True)
    assert evaluate(h, (broken,)).status == "technical_failure"
    records = [record(h, name="a"), record(h, name="b")]
    result = evaluate(h, (source(records),), DiscoveryPolicy(max_records=1))
    assert result.status == "budget_exhausted" and result.records_examined == 1


def test_lead_and_paper_identity_do_not_skip_missing_evidence(h):
    lead = record(h, kind="lead")
    assert (
        "controlled_source_required"
        in evaluate(h, (source([lead], kind="lead"),)).candidates[0].missing
    )
    paper = record(h, kind="paper_code")
    assert "paper_doi" in evaluate(h, (source([paper], kind="paper_code"),)).candidates[0].missing


def test_source_size_policy_bounds_and_uninstalled_transports(h):
    with pytest.raises(ValueError, match="discovery_offline_sources_required"):
        Discovery(gap_queue=h.service, repository=None, sources=(object(),))
    with pytest.raises(ValueError):
        DiscoveryPolicy(max_seconds=999)
    with pytest.raises(ValueError):
        DiscoveryPolicy(weights=(float("nan"),) * 7)
    p = record(h)
    q = query(h.service.prepare_research(h.context.run_id, key="q"), DiscoveryPolicy())
    c = inspect(p, source([p]), q, DiscoveryPolicy())
    assert c.disposition == "discovered" and c.onboarding_job is None


def test_missing_raw_inputs_and_unknown_producer_cannot_form_discovery_query(tmp_path):
    h = setup(tmp_path, missing=True)
    a = h.service.prepare_research(h.context.run_id, key="missing")
    assert not a.queueable
    with pytest.raises(ValueError, match="discovery_resolved_contract_required"):
        query(a, DiscoveryPolicy())


def test_05_query_uses_original_suggestions_and_rejects_withdrawal(tmp_path):
    from packages.tool_gap_queue import GapQueue
    from tests.grounding.query_support import updated
    from tests.tool_admission.support import fixture

    f = fixture(tmp_path)
    g = GapQueue(
        repository=None,
        grounding_owner=f.grounding,
        grounding_request_reader=lambda run_id, ref: f.locks.q,
    )
    request = updated(
        f.q,
        environment=[
            dict(e.model_dump(mode="json"), availability="unavailable") for e in f.q.environment
        ],
    )
    f.locks.q = request
    a = g.prepare_grounding(request, key="05")
    q = query(g.recheck(a), DiscoveryPolicy())
    assert q.contracts and q.requirement == a.requirement
    assert q.contracts[0].output in [s["constraint"] for s in a.material["result"]["suggestions"]]
    f.locks.q = updated(request, artifacts=[])
    with pytest.raises(ValueError, match="gap_grounding_request_changed"):
        g.recheck(a)


@pytest.mark.parametrize("mode", ["unpublished", "expired", "revoked", "technical"])
def test_optional_memory_preserves_original_publication_filters(h, mode):
    from packages.retrieval.memory import MemoryRetrieval
    from tests.memory_review.support import QueryOwner

    class Repo:
        contexts = h.service.contexts

        def publication_keys(self):
            return () if mode == "unpublished" else ("synthetic",)

        def read_publication(self, key):
            if mode == "technical":
                raise OSError("synthetic_memory_unavailable")
            raise ValueError("memory_qualification_" + mode)

    h.service.memory = MemoryRetrieval(
        repository=Repo(), query_reader=QueryOwner(h.context.run_id, h.request.scope.case_code)
    )
    d = Discovery(gap_queue=h.service, repository=None)
    diagnostic = d.memory_diagnostics(h.context.run_id)
    assert diagnostic["status"] == ("technical_failure" if mode == "technical" else "ok")
    assert not diagnostic["extension_authority"]
    assert not any(diagnostic["receipt"]["groups"].values())
    if mode in {"expired", "revoked"}:
        assert diagnostic["receipt"]["rejected"][0]["reason"] == "memory_qualification_" + mode
    elif mode == "unpublished":
        assert not diagnostic["receipt"]["rejected"]
    assert evaluate(h, ()).status == "no_candidates"


def test_control_projection_keeps_resolved_owner_contract_and_original_requirement(h):
    from packages.tool_gap_queue.models import Application

    a = h.service.prepare_research(h.context.run_id, key="resolved")
    for owner in ("08", "09"):
        wrapped = validate(
            Application,
            {
                **a.model_dump(mode="json"),
                "owner": owner,
                "purpose": "research_offline",
                "material": {"resolved": a.model_dump(mode="json")},
            },
        )
        q = query(wrapped, DiscoveryPolicy())
        assert q.contracts == query(a, DiscoveryPolicy()).contracts
        assert q.application_checksum == digest(wrapped) and q.requirement == a.requirement
    # Projection unit test only; authority remains GapQueue.prepare_control/recheck.


@pytest.mark.parametrize("license_value", ["", "unknown", "not_obtained"])
def test_unobtained_license_is_explicit_missing(h, license_value):
    p = record(h, code_license=license_value, weights_license=license_value)
    c = evaluate(h, (source([p]),)).candidates[0]
    assert c.disposition == "incomplete"
    assert {"code_license", "weights_license"} <= set(c.missing)


def test_plan_schema_and_bounded_preparation(h):
    from datetime import UTC, datetime, timedelta

    from packages.tool_discovery.models_v11 import DiscoveryPlan
    from packages.tool_gap_queue.models import Receipt

    a = h.service.prepare_research(h.context.run_id, key="plan")
    now = datetime.now(UTC)
    lease = Receipt(
        task_id="TASK:SYNTHETIC:PLAN",
        application=a,
        revision=1,
        state="leased",
        claims=1,
        created_at=now,
        expires_at=now + timedelta(days=1),
        lease_until=now + timedelta(seconds=60),
        lease_token="lease:fixture",
        worker="offline",
        reason="fixture",
    )
    d = Discovery(gap_queue=h.service, repository=None, sources=(source([record(h)]),))
    plan = d._prepare(lease, claim_key="claim", worker="offline")
    assert validate(DiscoveryPlan, plan).task_id == lease.task_id
    assert plan["ack_command_checksum"] == digest(
        dict(action="ack", parameters=dict(lease_token=lease.lease_token, worker="offline"))
    )
    result = d._evaluate(lease.task_id, plan, deadline=time.monotonic() - 1)
    assert result.status == "budget_exhausted" and result.records_examined == 0
