"""Confirmed failed Source is accounted history, never permission to replay UNKNOWN."""

import json
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import campaign_renewal as renewal
from packages.multimodal_research import confirmed_definition_failure as recovery
from tests.evaluation_preparation.test_interrupted_definition_successor import history
from tests.evaluation_preparation.test_revision_acquisition import policy


def failed(tmp_path, monkeypatch):
    parent, path, campaign, values, failure = history(tmp_path, monkeypatch)
    parent.base.binding = {"case_code": "CASE:TEST"}
    old = campaign.protocol
    old.update(new_public_search_allowed=False, original_fact_result=digest("facts"))
    old["expansion_policy"] = renewal.read_expansion_policy(policy("1.34.0")).model_dump(
        mode="json"
    )
    campaign.base.definition_expansion_policy = old["expansion_policy"]
    # Restore this fixture's operation binding to the real current owner inputs.
    binding = values["definition:operation:0"][0]["binding"]
    binding["protocol_checksum"] = digest(old)
    binding["policy_checksum"] = digest(old["expansion_policy"])
    failure.update(
        error_code="schema",
        failure_kind="confirmed_response_or_domain_refusal",
        previous_attempts=old["previous_attempts"],
        new_requests=1,
    )
    for key in list(values):
        if key != "definition:operation:0" and key != "journal:0":
            del values[key]
    values["definition:query:0:result"] = ({"hits": []},)
    values["definition:operation:1"] = (
        {"kind": "llm", "key": "definition:proposal", "binding": binding},
    )
    scope = dict(run_id=failure["run_id"], case_code="CASE:TEST", input_checksum=old["input"])
    record = dict(
        request_id="r1",
        invocation_id="r1",
        role="source",
        error="schema",
        api_requests=1,
        simulated=False,
        retry_index=0,
        scope=scope,
        config_checksum=digest("source"),
        prompt_version=old["expansion_policy"]["prompt_version"],
        output_checksum=digest(json.dumps({"invalid": True}).encode().hex()),
    )
    values["journal:1"] = (
        dict(
            phase="reserved",
            request_id="r1",
            scope_checksum=digest(scope),
            config_checksum=record["config_checksum"],
        ),
    )
    values["journal:2"] = (dict(phase="completed", record=record),)
    values["journal:3"] = (dict(phase="response_confirmation", request_id="r1", confirmed=True),)
    values["definition:response:r1"] = (
        dict(record=record, raw_text=json.dumps({"invalid": True})),
    )
    (path / "protocol.json").write_text(json.dumps(old))
    (path / "stage-failure.json").write_text(json.dumps(failure))
    return parent, path, campaign, values, failure


def test_confirmed_failure_uses_original_binding_and_preserves_history(tmp_path, monkeypatch):
    parent, path, campaign, values, failure = failed(tmp_path, monkeypatch)
    before = deepcopy(values)
    proof = recovery.tip(parent, path)
    assert values == before
    assert proof["previous_attempts"][-1]["requests"] == 1
    assert proof["result"] == digest(failure)
    successor = NS(parent=parent, protocol={"tip": proof})
    assert (
        renewal.attempted_acquisition_targets(successor, proof["previous_attempts"][-1:]) == set()
    )
    with pytest.raises(ValueError, match="history_owner"):
        recovery.verify_history(successor, dict(proof["previous_attempts"][-1], requests=0))


def test_local_repaired_quote_failure_has_explicit_selection_transition(tmp_path, monkeypatch):
    parent, path, campaign, values, failure = failed(tmp_path, monkeypatch)
    old = campaign.protocol
    old["schema_version"] = "multimodal-campaign-renewal/1.62.0"
    old["expansion_policy"] = renewal.read_expansion_policy(
        dict(policy("1.36.0"), prompt_version="definition-draft-6.7.0")
    ).model_dump(mode="json")
    campaign.base.definition_expansion_policy = old["expansion_policy"]
    binding = values["definition:operation:0"][0]["binding"]
    binding.update(protocol_checksum=digest(old), policy_checksum=digest(old["expansion_policy"]))
    record = values["journal:2"][0]["record"]
    record["prompt_version"] = "definition-draft-6.7.0"
    values["definition:response:r1"][0]["record"] = record
    (path / "protocol.json").write_text(json.dumps(old))
    with pytest.raises(ValueError, match="passage_diagnostic"):
        recovery.tip(parent, path)
    values["journal:4"] = (
        dict(
            schema_version="source-definition-contract-diagnostic/1.0.0",
            request_id="r1",
            failures=[
                dict(
                    path=["definitions", 0, "excerpts", 0, "text"],
                    type="verbatim_request_quote_required",
                )
            ],
        ),
    )
    proof = recovery.tip(parent, path)
    assert proof["previous_attempts"][-1]["requests"] == 1
    new = recovery.policy_for(old)
    assert new["schema_version"] == "campaign-definition-expansion/1.37.0"
    assert new["prompt_version"] == "definition-draft-6.8.0"
    assert new["reserve_review_requests"] == 2
    renewal.validate_increment_window(new, seconds=3600, cap=3)
    with pytest.raises(ValueError, match="no_proposal_capacity"):
        renewal.validate_increment_window(new, seconds=3600, cap=2)
    assert new["index_keys"] == old["expansion_policy"]["index_keys"]
    with pytest.raises(ValueError, match="format_transition"):
        recovery.policy_for(dict(old, schema_version=recovery.PASSAGE_VERSION))


def test_selection_successor_keeps_exact_intervening_local_repair_owner(tmp_path, monkeypatch):
    from packages.multimodal_research import interrupted_definition as interrupted
    from tests.evaluation_preparation.test_confirmed_local_index_successor import repaired_history

    parent, path, _, _, _ = repaired_history(tmp_path, monkeypatch)
    original = interrupted.tip(parent, path, local_failure=True)
    failed_path = tmp_path / "failed-source"
    failed_path.mkdir()
    (failed_path / "protocol.json").write_text(
        json.dumps(
            dict(
                schema_version=interrupted.LOCAL_VERSION,
                tip=original,
            )
        )
    )
    proof = dict(path=str(failed_path), result_kind=recovery.KIND)
    monkeypatch.setattr(
        recovery,
        "tip",
        lambda p, where: proof if p is parent and where == str(failed_path) else None,
    )
    successor = NS(parent=parent, protocol=dict(schema_version=recovery.PASSAGE_VERSION, tip=proof))
    attempt = original["previous_attempts"][-1]
    assert renewal.attempted_acquisition_targets(successor, [attempt]) == set()
    tampered = deepcopy(original)
    tampered["interruption"]["indexes"][0]["checksum"] = digest("changed")
    (failed_path / "protocol.json").write_text(
        json.dumps(dict(schema_version=interrupted.LOCAL_VERSION, tip=tampered))
    )
    with pytest.raises(ValueError, match="history_owner"):
        renewal.attempted_acquisition_targets(successor, [attempt])


@pytest.mark.parametrize(
    "fault",
    [
        "unknown",
        "pending",
        "duplicate",
        "raw",
        "record",
        "scope",
        "provider",
        "counter",
        "wire",
        "hole",
        "proposal",
        "revision",
        "cancel",
        "parent",
        "result",
        "wrong_failure",
    ],
)
def test_wrong_or_unconfirmed_origin_never_authorizes_request(tmp_path, monkeypatch, fault):
    parent, path, campaign, values, failure = failed(tmp_path, monkeypatch)
    if fault == "unknown":
        values["journal:3"][0]["confirmed"] = None
    elif fault == "pending":
        del values["journal:2"]
    elif fault == "duplicate":
        values["journal:4"] = values["journal:1"]
    elif fault == "raw":
        values["definition:response:r1"][0]["raw_text"] = "{}"
    elif fault == "record":
        values["definition:response:r1"][0]["record"] = {}
    elif fault == "scope":
        values["journal:2"][0]["record"]["scope"]["input_checksum"] = digest("other")
    elif fault == "provider":
        values["journal:1"][0]["config_checksum"] = digest("other")
    elif fault == "counter":
        failure["new_requests"] = 0
    elif fault == "wire":
        values["definition:operation:0"][0]["kind"] = "wire"
    elif fault == "hole":
        values["definition:operation:2"] = values.pop("definition:operation:1")
    elif fault == "proposal":
        values["definition:proposal:result"] = ({"ok": True},)
    elif fault == "revision":
        values["definition:revision"] = ({},)
    elif fault == "cancel":
        (path / "cancel").touch()
    elif fault == "parent":
        parent.output = tmp_path / "wrong"
    elif fault == "result":
        (path / "result.json").write_text("{}")
    else:
        failure["error_code"] = "transport"
    with pytest.raises(ValueError):
        recovery.tip(parent, path)


def plan(tmp_path, monkeypatch):
    parent, path, campaign, values, failure = failed(tmp_path, monkeypatch)
    old = campaign.protocol
    proof = recovery.tip(parent, path)
    return dict(
        schema_version=recovery.VERSION,
        confirmed_definition_failure=True,
        tip=proof,
        logical_campaign_id=path.name,
        new_public_search_allowed=False,
        facts=old["facts"],
        provider=old["provider"],
        input=old["input"],
        task=old["task"],
        original_fact_result=old["original_fact_result"],
        prior_deadline=old["deadline"],
        previous_attempts=proof["previous_attempts"],
        expansion_policy=recovery.policy_for(old),
        started_at="2020-01-02T00:00:00+00:00",
        deadline="2020-01-02T01:00:00+00:00",
        seconds=3600,
        max_new_requests=4,
        max_cumulative_requests=40,
        execution_lease_profile="multimodal-runtime-serialization/1.0.0",
        mode="definitions",
        definition_increment_only=True,
    )


def test_only_format_changes_no_medical_or_search_scope_change(tmp_path, monkeypatch):
    p = plan(tmp_path, monkeypatch)
    recovery.check_plan(p)
    assert renewal.current_increment(p)
    assert p["expansion_policy"]["prompt_version"] == "definition-draft-6.7.0"
    from packages.multimodal_research.execution_lease import PROFILE

    p["execution_lease_profile"] = PROFILE
    renewal.check_local_history(p)


def test_passage_successor_preserves_original_logical_campaign_not_failed_window_name():
    old = dict(schema_version="multimodal-campaign-renewal/1.62.0", logical_campaign_id="root")
    assert recovery.logical_campaign(old, "local-failure-successor") == "root"
    with pytest.raises(ValueError, match="campaign_identity"):
        recovery.logical_campaign(dict(old, logical_campaign_id=None), "window")
    assert (
        recovery.logical_campaign(
            dict(schema_version="multimodal-campaign-renewal/1.56.0"), "old-window"
        )
        == "old-window"
    )


@pytest.mark.parametrize(
    "fault", ["budget", "cumulative", "clock", "scope", "policy", "history", "facts", "version"]
)
def test_successor_cannot_reset_or_expand(tmp_path, monkeypatch, fault):
    p = plan(tmp_path, monkeypatch)
    if fault == "budget":
        p["max_new_requests"] = 8
    elif fault == "cumulative":
        p["max_cumulative_requests"] = 100
    elif fault == "clock":
        p["seconds"] = 7200
    elif fault == "scope":
        p["new_public_search_allowed"] = True
    elif fault == "policy":
        p["expansion_policy"]["public_queries"] = ["unrelated"]
    elif fault == "history":
        p["previous_attempts"] = []
    elif fault == "facts":
        p["facts"] = "other"
    else:
        p["schema_version"] = renewal.RUNTIME_DEFINITION_VERSION
    with pytest.raises(ValueError):
        recovery.check_plan(p)


def test_production_gate_uses_current_successor_history_owner(tmp_path, monkeypatch):
    parent, path, failed_campaign, values, failure = failed(tmp_path, monkeypatch)
    proof = recovery.tip(parent, path)
    # Only this incoming schema failure needs the successor's owner. Historical
    # fact records still come from the unchanged completed parent.
    parent.accepted_observations = lambda: {}
    parent.protocol = {}
    c = NS(
        parent=parent,
        protocol=dict(
            schema_version=recovery.VERSION,
            tip=proof,
            expansion_policy=recovery.policy_for(failed_campaign.protocol),
            confirmed_observations_checksum=digest({}),
        ),
        previous_attempts=lambda: proof["previous_attempts"][-1:],
        base=NS(observations={}),
        targets=("check:new",),
    )
    revision = dict(
        added=[],
        replaced=[dict(id="check:new")],
        rows=[dict(id="check:new", dependency_assessment={}, review_allowed=True)],
    )
    assert renewal.gate_selection(c, revision) == ("check:new",)
    assert c.targets == ("check:new",)
