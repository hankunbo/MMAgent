"""New-window accounting and confirmed-material scope; not medical validation."""

import json
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import campaign_renewal as renewal
from packages.multimodal_research import interrupted_definition as recovery
from packages.multimodal_research import stage_failure
from tests.evaluation_preparation.test_definition_runtime import policy
from tests.evaluation_preparation.test_interrupted_definition_closure import fixture


class Config(dict):
    max_requests = 40


def history(tmp_path, monkeypatch):
    c, docs, values, context = fixture()
    c.base.task.checksum = digest("task")
    c.config = Config(config="unchanged")
    parent = NS(output=tmp_path / "facts", base=c.base, config=c.config)
    path = tmp_path / "stopped"
    path.mkdir()
    old = dict(
        schema_version=renewal.RUNTIME_DEFINITION_VERSION,
        mode="definitions",
        definition_increment_only=True,
        facts=str(parent.output),
        input=c.base.manifest.checksum,
        task=c.base.task.checksum,
        provider=digest(c.config),
        started_at="2020-01-01T00:00:00+00:00",
        deadline="2020-01-01T01:00:00+00:00",
        max_new_requests=8,
        expansion_policy=dict(max_operations=8),
        previous_attempts=[dict(requests=4)],
    )
    c.protocol = old
    c.contexts.read_research_run = lambda rid, **kw: context if rid == docs.run_id else None
    context.request.policy_ref.checksum = digest((c.base.ledger, c.config, None))
    c.documents = lambda rid: docs
    c.sessions = NS(read=lambda rid: None)
    c.targets = ()
    c.request_key = lambda target: "campaign:" + (target or "review")
    binding = values["definition:operation:0"][0]["binding"]
    binding["protocol_checksum"] = digest(old)
    values["definition:operation:1"][0]["wire_kind"] = "fetch"
    values["definition:wire:1:result"] = (
        dict(provider="SOURCE:PROVIDER:EUROPEPMC", reservation_only=True),
    )
    values["definition:wire:1:outcome"] = (
        dict(
            adapter="europe_pmc",
            provider_code="SOURCE:PROVIDER:EUROPEPMC",
            provider_version="2.0.0",
            config_checksum=digest("config"),
            http_status=200,
            operation="fetch",
            request_checksum=digest("request"),
            response_checksum=digest("body"),
            requested_at="2020-01-01T00:01:00Z",
        ),
    )
    values["definition:operation:2"] = (
        dict(key="definition:fetch:index:window:0", kind="local_index_window", binding=binding),
    )
    values["definition:fetch:index:window:0:result"] = (
        dict(key=digest("index"), status="indexed"),
    )
    values["journal:0"] = (
        dict(
            schema_version="agent-budget/1.0.0",
            binding=dict(run_id=docs.run_id, deadline=old["deadline"], max_requests=40),
        ),
    )
    failure = dict(
        error_code="execution_process_interrupted",
        stage="definitions",
        failure_kind="no_provider_reservation",
        requests=[],
        scores=None,
        definition_revision_saved=False,
        run_id=docs.run_id,
        new_requests=1,
    )
    (path / "protocol.json").write_text(json.dumps(old))
    monkeypatch.setattr(renewal, "RenewedCampaign", lambda *args: c)
    monkeypatch.setattr(stage_failure, "read_saved_failure", lambda campaign: failure)
    return parent, path, c, values, failure


def test_original_tuple_journal_is_unstarted_and_budget_is_preserved(tmp_path, monkeypatch):
    parent, path, c, values, failure = history(tmp_path, monkeypatch)
    before = deepcopy(values)
    proof = recovery.tip(parent, path)
    assert proof["previous_attempts"][-1]["requests"] == 1
    assert sum(a["requests"] for a in proof["previous_attempts"]) == 5
    assert proof["result_kind"] == recovery.KIND
    assert proof["previous_attempts"][-1]["kind"] == recovery.HISTORY_KIND
    assert proof["result"] == digest(failure) and values == before
    assert len(proof["interruption"]["indexes"]) == 1


def test_downstream_accounting_rechecks_original_interruption_owner(tmp_path, monkeypatch):
    parent, path, c, values, failure = history(tmp_path, monkeypatch)
    proof = recovery.tip(parent, path)
    successor = NS(parent=parent, protocol={"tip": proof})
    attempt = proof["previous_attempts"][-1]
    assert renewal.attempted_acquisition_targets(successor, [attempt]) == set()
    changed = dict(attempt, requests=2)
    with pytest.raises(ValueError, match="interrupted_definition_history_owner"):
        renewal.attempted_acquisition_targets(successor, [changed])


@pytest.mark.parametrize(
    "fault",
    [
        "unknown_wire",
        "wire_status",
        "source_reserved",
        "review_started",
        "index_missing",
        "cancel",
        "counter",
        "checkpoint",
        "wrong_fact_parent",
    ],
)
def test_incomplete_or_wrong_history_never_grants_successor(tmp_path, monkeypatch, fault):
    parent, path, c, values, failure = history(tmp_path, monkeypatch)
    if fault == "unknown_wire":
        values.pop("definition:wire:1:outcome")
    elif fault == "wire_status":
        values["definition:wire:1:outcome"][0]["http_status"] = 500
    elif fault == "source_reserved":
        values["journal:1"] = (dict(phase="reserved"),)
    elif fault == "review_started":
        original_read = c.contexts.read_research_run
        c.contexts.read_research_run = lambda *a, **kw: original_read(*a, **kw) or NS()
    elif fault == "index_missing":
        values.pop("definition:fetch:index:window:0:result")
    elif fault == "cancel":
        (path / "cancel").touch()
    elif fault == "counter":
        failure["new_requests"] = 0
    elif fault == "checkpoint":
        values["definition:stage:sources"] = (dict(binding={}),)
    else:
        parent.output = tmp_path / "other"
    with pytest.raises(ValueError):
        recovery.tip(parent, path)


def window(tmp_path):
    old = dict(
        expansion_policy=policy(),
        facts="same-facts",
        deadline="2020-01-01T01:00:00+00:00",
        max_new_requests=8,
        provider=digest("provider"),
        input=digest("input"),
        task=digest("task"),
        original_fact_result=digest("fact-result"),
    )
    (tmp_path / "protocol.json").write_text(json.dumps(old))
    proof = dict(
        path=str(tmp_path),
        result_kind=recovery.KIND,
        previous_attempts=[dict(requests=17), dict(requests=3)],
        interruption=dict(indexes=[dict(key=digest("new-index"))]),
    )
    return dict(
        schema_version=recovery.VERSION,
        interrupted_definitions=True,
        tip=proof,
        logical_campaign_id=tmp_path.name,
        new_public_search_allowed=False,
        facts=old["facts"],
        prior_deadline=old["deadline"],
        provider=old["provider"],
        input=old["input"],
        task=old["task"],
        original_fact_result=old["original_fact_result"],
        expansion_policy=recovery.policy_for(old, proof),
        previous_attempts=proof["previous_attempts"],
        started_at="2020-01-02T00:00:00+00:00",
        deadline="2020-01-02T01:00:00+00:00",
        seconds=3600,
        max_new_requests=5,
        max_cumulative_requests=40,
    )


def test_new_window_retains_remaining_budget_and_original_queries(tmp_path):
    p = window(tmp_path)
    recovery.check_plan(p)
    old = json.loads((tmp_path / "protocol.json").read_bytes())
    assert p["expansion_policy"]["public_queries"] == list(
        old["expansion_policy"]["public_queries"]
    )
    assert digest("new-index") in p["expansion_policy"]["index_keys"]
    assert (tmp_path / "result.json").exists() is False


@pytest.mark.parametrize(
    "fault", ["http", "budget", "clock", "provider", "queries", "scope", "count"]
)
def test_new_window_cannot_extend_scope_or_reset_history(tmp_path, fault):
    p = window(tmp_path)
    if fault == "http":
        p["new_public_search_allowed"] = True
    elif fault == "budget":
        p["max_new_requests"] = 6
    elif fault == "clock":
        p["seconds"] = 7200
    elif fault == "provider":
        p["provider"] = digest("other")
    elif fault == "queries":
        p["expansion_policy"]["public_queries"] = ["different"]
    elif fault == "scope":
        p["logical_campaign_id"] = "other"
    else:
        p["previous_attempts"] = []
    with pytest.raises(ValueError, match="interrupted_definition_successor_binding"):
        recovery.check_plan(p)
