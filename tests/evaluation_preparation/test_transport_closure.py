"""Offline regression for incomplete visual requests and target-specific definition gaps."""

import copy
import errno
import json
import ssl
import urllib.error
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from tests.evaluation_preparation.test_definition_expansion import fixture


def test_safe_transport_diagnostic_preserves_reason_not_message():
    from packages.llm_provider import worker

    diagnostic = getattr(worker, "safe_transport_error", None)
    assert callable(diagnostic), "transport exceptions currently lose their reason"
    value = diagnostic(
        urllib.error.URLError(ConnectionResetError(errno.ECONNRESET, "secret")), "dispatch_boundary"
    )
    assert value["category"] == "connection_reset"
    assert value["stage"] == "dispatch_boundary"
    assert "secret" not in json.dumps(value)
    assert diagnostic(ssl.SSLError("secret"), "dispatch_boundary")["category"] == "tls_error"
    assert diagnostic(RuntimeError("secret"), "response_headers")["category"] == "unknown_transport"


def test_installed_general_definition_is_not_target_binding(tmp_path):
    from packages.multimodal_research.definition_expansion import DefinitionExpansion

    e = fixture(tmp_path)
    revision = e.advance()
    row = copy.deepcopy(e.campaign.base.ledger[0])
    # Keep an installed definition but remove its accepted receipt: existing catalog case.
    row.pop("definition_receipt_checksum", None)
    e.campaign.base.ledger = [row]
    policy = dict(
        schema_version="campaign-definition-expansion/1.1.0",
        index_keys=[],
        public_queries=["public texture relationship alternatives"],
        query_requirements={"public texture relationship alternatives": [row["id"]]},
        require_target_relation=True,
    )
    newer = DefinitionExpansion(
        e.campaign,
        documents=e.documents,
        sources=e.sources,
        provider=e.provider,
        budget=e.budget,
        scope=e.scope,
        policy=policy,
    )
    plan = newer.query_plan()[0]
    assert plan["selected"] and plan["target_relation_gap_ids"] == [row["id"]]
    assert plan["reason"] == "target_relation_unverified"
    assert revision["added"]  # Existing facts are not deleted to force discovery.


def test_new_profile_is_explicit_and_old_profile_unchanged():
    from packages.multimodal_research.active_requirements import load

    new = load("configs/research/multimodal_active_requirements_v3.json")
    old = load("configs/research/multimodal_active_requirements_v2.json")
    assert "observation_failure_policy" not in old
    assert new["observation_failure_policy"] == "confirmed-only-closure/1.0.0"
    assert new["definition_expansion"]["require_target_relation"]
    assert new["checks"] == old["checks"]


@pytest.mark.parametrize(
    "role,error",
    [
        ("review", "transport"),
        ("vision", "scope"),
        ("vision", "authentication"),
        ("vision", "cancelled"),
    ],
)
def test_non_observation_fault_never_authorizes_closure(role, error):
    from packages.multimodal_research.observation_closure import transport_claims

    events = [
        dict(phase="reserved", request_id="r"),
        dict(phase="completed", record=dict(request_id="r", role=role, error=error)),
    ]
    assert transport_claims(events) is None


def test_local_completion_does_not_confirm_remote_or_authorize_replay():
    from packages.multimodal_research.observation_closure import transport_claims

    events = [
        dict(phase="reserved", request_id="r"),
        dict(
            phase="completed",
            record=dict(request_id="r", role="vision", error="transport", api_requests=1),
        ),
    ]
    value = transport_claims(events)
    assert value[0]["response_confirmed"] is None and not value[0]["replay_authorized"]
    assert transport_claims(events + [dict(phase="reserved", request_id="unknown")]) is None
    assert transport_claims(events + [events[0]]) is None


def test_general_to_target_relation_rebind_preserves_identity_and_denominator(tmp_path):
    from packages.evidence_reuse.definition_acceptance import requirement_rows
    from packages.multimodal_research.definition_expansion import read_expansion_policy
    from tests.evaluation_preparation.test_definition_expansion import draft

    e = fixture(tmp_path)
    old = e.owner.build(draft(e.binding["target"]))
    row = requirement_rows(old, e.campaign.base.manifest)[0]
    row.pop("definition_receipt_checksum")
    e.original_rows = e.campaign.base.ledger = [row]
    e.binding["parent_requirements_checksum"] = digest([row])
    e.policy = read_expansion_policy(
        dict(
            schema_version="campaign-definition-expansion/1.1.0",
            index_keys=[],
            public_queries=["public texture relationship alternatives"],
            query_requirements={"public texture relationship alternatives": [row["id"]]},
            require_target_relation=True,
        )
    )
    e.binding["policy_checksum"] = digest(e.policy)
    before = digest(row)
    result = e.advance()
    assert result["denominator_before"] == result["denominator_after"] == 1
    assert not result["added"] and result["replaced"][0]["reason"] == "target_relation_rebound"
    assert digest(row) == before
    assert e.target_relation_bound(e.campaign.base.ledger[0])
    assert e.advance() == result and e.sources.calls == 1


def test_worker_reports_safe_error_without_network_or_secret(monkeypatch, capsys):
    import io

    from packages.llm_provider import worker
    from packages.multimodal_research.campaign import configuration

    config = configuration().transport
    monkeypatch.setenv(config.api_key_env, "synthetic-test-only")
    monkeypatch.setattr(
        worker.sys,
        "stdin",
        SimpleNamespace(
            buffer=io.BytesIO(
                json.dumps(
                    dict(
                        config=config.model_dump(mode="json"),
                        body={},
                        max_bytes=100,
                        timing_events=True,
                    )
                ).encode()
            )
        ),
    )

    def failed(*a, **k):
        raise urllib.error.URLError(
            ConnectionResetError(errno.ECONNRESET, "synthetic-test-only private input")
        )

    monkeypatch.setattr(
        worker.urllib.request, "build_opener", lambda *a: SimpleNamespace(open=failed)
    )
    assert worker.main() == 0
    captured = capsys.readouterr()
    assert "synthetic-test-only" not in captured.out + captured.err
    value = json.loads(captured.out.removeprefix("DISPATCH\n"))
    assert value["error"] == "transport"
    assert value["diagnostic"]["transport_failure"]["category"] == "connection_reset"


def test_closure_refuses_old_profile_before_reading_owners():
    from packages.multimodal_research.observation_closure import close_observation

    assert not close_observation(
        SimpleNamespace(base=SimpleNamespace()),
        dict(target="a", accepted=False, error="candidate_failed_or_context_changed"),
    )


def test_saved_closure_survives_cold_report_without_new_authorization():
    from packages.multimodal_research.observation_closure import POLICY, saved_closure

    phase = dict(target="observation:a", run_id="failed", error="transport")
    protocol = dict(campaign_id="c", deadline="past-historical-deadline")
    value = dict(
        schema_version=POLICY,
        campaign_id="c",
        input_checksum="input",
        protocol_checksum=digest(protocol),
        failed_phase_checksum=digest(phase),
        failed_target=phase["target"],
        stop_new_observations=True,
        next_action="review_confirmed_only",
        original_failure_preserved=True,
    )
    data = {"observation-closure": (value,), "dynamic-phase-result": (phase,)}
    campaign = SimpleNamespace(
        base=SimpleNamespace(
            observation_failure_policy=POLICY, manifest=SimpleNamespace(checksum="input")
        ),
        campaign_id="c",
        protocol=protocol,
        documents=lambda run_id: SimpleNamespace(read=data.get),
    )
    assert not hasattr(campaign, "observation_closure")
    assert saved_closure(campaign, [phase, dict(target=None)]) == value
    value["protocol_checksum"] = "wrong-campaign"
    with pytest.raises(ValueError, match="observation_closure_report_binding"):
        saved_closure(campaign, [phase])
    assert saved_closure(SimpleNamespace(base=SimpleNamespace()), [phase]) is None


def closure_fixture(tmp_path):
    from pydantic import BaseModel

    from tests.agent_provider.support import Documents
    from tests.evaluation_preparation.test_directional_review import fixture as state_fixture

    state = state_fixture().owner.current()
    phase = dict(
        target="failed-image",
        accepted=False,
        error="candidate_failed_or_context_changed",
        run_id=state.material.scope.run_id,
    )
    docs = Documents()
    docs.write("dynamic-phase-input", (state,))
    docs.write("journal:0", (dict(phase="reserved", request_id="r"),))
    docs.write(
        "journal:1",
        (
            dict(
                phase="completed",
                record=dict(role="vision", error="transport", request_id="r", api_requests=1),
            ),
        ),
    )

    class Current(BaseModel):
        phase: str
        cancelled: bool
        pending: object = None
        graph: object

    current = Current(
        phase="stopped", cancelled=False, pending=None, graph=state.material.scope.graph
    )
    phase["planner"] = dict(control_checksum=digest(current))
    docs.write("dynamic-phase-result", (phase,))
    campaign = SimpleNamespace(
        base=SimpleNamespace(
            observation_failure_policy="confirmed-only-closure/1.0.0",
            manifest=SimpleNamespace(checksum=digest("input")),
            observations={"confirmed": state.material.observations[0]},
        ),
        documents=lambda run: docs,
        guard=lambda: None,
        sessions=SimpleNamespace(read=lambda run: current),
        phase_class=lambda *a, **kw: SimpleNamespace(
            project=lambda **kw: None, verify=lambda initial: None
        ),
        accepted_observations=lambda: {},
        read_model=lambda: None,
        config={},
        campaign_id="closure-unit",
        protocol={},
    )
    return campaign, phase, current, docs


@pytest.mark.parametrize("failure", ["cancel", "pending", "running", "no_facts", "review"])
def test_closure_rejects_unsafe_or_empty_state(tmp_path, failure):
    from packages.multimodal_research.observation_closure import close_observation

    c, phase, state, docs = closure_fixture(tmp_path)
    if failure == "cancel":
        state.cancelled = True
    elif failure == "pending":
        state.pending = object()
    elif failure == "running":
        state.phase = "running"
    elif failure == "no_facts":
        c.base.observations.clear()
    else:
        phase["target"] = None
    assert not close_observation(c, phase)
    assert docs.read("observation-closure") is None


@pytest.mark.parametrize("failure", ["cancelled", "deadline", "budget", "scope"])
def test_current_owner_failure_cannot_become_partial_success(tmp_path, failure):
    from packages.llm_provider.provider import ProviderError
    from packages.multimodal_research.observation_closure import close_observation

    c, phase, state, docs = closure_fixture(tmp_path)

    def deny():
        raise ProviderError(failure)

    c.guard = deny
    with pytest.raises(ProviderError, match=failure):
        close_observation(c, phase)
    assert docs.read("observation-closure") is None


def test_closure_is_monotone_idempotent_and_preserves_failure(tmp_path):
    from packages.multimodal_research.observation_closure import close_observation

    c, phase, state, docs = closure_fixture(tmp_path)
    original = docs.read("dynamic-phase-result")
    assert close_observation(c, phase)
    before = digest(docs.saved)
    assert close_observation(c, phase) and digest(docs.saved) == before
    assert docs.read("dynamic-phase-result") == original
    assert c.observation_closure["requests"][0]["response_confirmed"] is None


@pytest.mark.parametrize("exhausted", [False, True])
def test_independent_observation_failure_never_replays_unknown_but_can_continue(
    tmp_path, exhausted
):
    from packages.multimodal_research.observation_closure import (
        close_observation,
        saved_independent_failures,
    )

    c, phase, state, docs = closure_fixture(tmp_path)
    c.protocol.update(
        independent_observation_policy="bounded-independent-observations/1.0.0",
        independent_observation_failure_limit=2,
    )
    c.acquisition_exhausted = lambda: exhausted
    original = digest(docs.saved)
    assert close_observation(c, phase)
    saved = docs.read("independent-observation-failure")[0]
    assert saved["stop_new_observations"] is exhausted
    assert saved["original_target_replay_authorized"] is False
    assert saved["requests"][0]["response_confirmed"] is None
    assert saved["next_action"] == (
        "review_confirmed_only" if exhausted else "continue_distinct_unattempted_target"
    )
    assert hasattr(c, "observation_closure") is exhausted
    assert docs.read("observation-closure") is None
    assert docs.read("dynamic-phase-result") == (phase,)
    after = digest(docs.saved)
    assert after != original
    assert close_observation(c, phase) and digest(docs.saved) == after
    assert saved_independent_failures(c, [phase]) == [saved]
    c.protocol["independent_observation_failure_limit"] = 3
    with pytest.raises(ValueError, match="independent_observation_report_binding"):
        saved_independent_failures(c, [phase])


def test_new_entry_selects_policy_and_rejects_version_mix(tmp_path, monkeypatch):
    from pathlib import Path

    from packages.multimodal_research import entry, entry_preflight
    from tests.dataset_adapters.test_multimodal import gamma

    def views(command, **kwargs):
        out = Path(command[4])
        out.mkdir(parents=True)
        (out / "views.json").write_text(json.dumps(dict(views=[])))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(entry.subprocess, "run", views)
    # Policy dispatch test; resource drift has its own command-level tests.
    monkeypatch.setattr(entry_preflight, "check_local_route", lambda *args: "synthetic_route")
    out = tmp_path / "prepared"
    entry.prepare_entry(
        dataset="gamma",
        root=gamma(tmp_path / "source"),
        sample="0001",
        task="advanced_glaucoma",
        output=out,
        registry="configs/research/multimodal_propositions_v1.json",
        active_acquisition="configs/research/multimodal_active_requirements_v3.json",
        resilient_vision=True,
        view_profile="detail-v1",
    )
    spec = json.loads((out / "entry.json").read_bytes())
    assert spec["schema_version"] == "multimodal-live-entry/1.11.0"
    assert spec["max_requests"] == 40 and spec["seconds"] == 5400

    def owner_boundary():
        raise RuntimeError("verified_entry_reached_owner_boundary")

    monkeypatch.setattr(entry, "owners", owner_boundary)
    with pytest.raises(RuntimeError, match="verified_entry_reached_owner_boundary"):
        entry.assemble_entry(out)
    spec["schema_version"] = "multimodal-live-entry/1.10.0"
    (out / "entry.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="observation_closure_entry_version"):
        entry.assemble_entry(out)


def test_changed_control_receipt_cannot_authorize_closure(tmp_path):
    from packages.multimodal_research.observation_closure import close_observation

    c, phase, state, docs = closure_fixture(tmp_path)
    state.graph = state.graph.model_copy(update=dict(checksum=digest("changed graph")))
    with pytest.raises(ValueError, match="observation_closure_control_changed"):
        close_observation(c, phase)
    assert docs.read("observation-closure") is None


def test_saved_closure_is_visible_without_rescoring_or_javascript():
    from packages.evaluation_preparation.evidence_tree_v3 import build, render
    from tests.evaluation_preparation.test_unified_view import report

    value = report()
    scores = digest(value["scores"])
    value["observation_closure"] = dict(
        schema_version="confirmed-only-closure/1.0.0",
        failed_target="a",
        requests=[dict(response_confirmed=None, replay_authorized=False)],
        stop_new_observations=True,
    )
    before = digest(value)
    packet = build(value, input_checksum=digest(value))
    assert packet["nodes"][0]["observation_closure"] == value["observation_closure"]
    html = render(value, input_checksum=digest(value))
    assert "仅已确认观察收尾（失败请求未重发）" in html.split('<script id="evidence-view"')[0]
    assert digest(value) == before and digest(packet["scores"]) == scores
