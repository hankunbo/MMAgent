"""A third attempt is an explicit bounded successor, never UNKNOWN replay."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation import generic_acquisition_successor as owner
from packages.grounding.models import digest
from tests.evaluation_preparation.test_generic_acquisition_chain import fixture


def client_failed_fixture(tmp_path, monkeypatch):
    checked, saved, journal, calls, state = fixture(tmp_path, monkeypatch)
    from packages.llm_provider.client_selection import load_selection

    original = checked.proof
    checked.proof = dict(
        original,
        schema_version=owner.CHAIN_VERSION,
        original=original,
        previous_requests=14,
        failed_successor=dict(key="first", request_id="first-request"),
        client_binding=load_selection("configs/agents/system_curl_client_v1.json"),
    )
    checked.config = checked.campaign.config
    selection = owner.selection_for(checked, "failed")
    (tmp_path / "first").mkdir()
    (tmp_path / "first/acquisition-successor.json").write_text(json.dumps(selection))
    (tmp_path / "failed/input.json").write_text(json.dumps(selection))
    result = saved["dynamic-case-result"][0]
    result["prior_graph_requests"] = 14
    (tmp_path / "failed/evidence/result.json").write_text(json.dumps(result))
    protocol = saved["dynamic-campaign"][0]
    protocol["observations"]["proof"] = checked.proof
    (tmp_path / "failed/evidence/protocol.json").write_text(json.dumps(protocol))
    saved["dynamic-campaign"] = (dict(protocol, **owner.client_implementation(checked.proof)),)
    report = saved["dynamic-case-report"][0]
    saved["dynamic-case-report"] = (report, dict(report_checksum=digest(report)))
    return checked, saved, journal, calls, state


def test_exact_failed_client_selection_uses_immediate_claim_and_original_request(
    tmp_path, monkeypatch
):
    checked, saved, journal, calls, state = client_failed_fixture(tmp_path, monkeypatch)
    before = digest(saved), digest(journal)
    value = owner.failed_successor(checked, "failed", "child-request")
    assert value["requests"] == 1 and calls == ["child-request"]
    assert before == (digest(saved), digest(journal))
    assert not (tmp_path / "failed/acquisition-successor.json").exists()


@pytest.mark.parametrize(
    "fault", ["claim", "cancel", "budget", "material", "protocol_extra", "protocol_hash"]
)
def test_failed_client_history_cannot_be_forged(tmp_path, monkeypatch, fault):
    checked, saved, journal, calls, state = client_failed_fixture(tmp_path, monkeypatch)
    if fault == "claim":
        (tmp_path / "first/acquisition-successor.json").write_text("{}")
    elif fault == "cancel":
        (tmp_path / "failed/cancel-request.json").write_text("{}")
    elif fault == "budget":
        journal.append(dict(phase="reserved", request_id="hidden"))
    elif fault == "protocol_extra":
        saved["dynamic-campaign"][0]["unexpected"] = True
    elif fault == "protocol_hash":
        saved["dynamic-campaign"][0]["packages/llm_provider/curl_client.py"] = digest("changed")
    else:
        saved["dynamic-phase-input"] = (
            state.model_copy(update={"config_checksum": digest("different")}).model_dump(
                mode="json"
            ),
        )
    with pytest.raises(ValueError, match="generic_acquisition_failed_child_"):
        owner.failed_successor(checked, "failed", "child-request")


def proof_tree():
    base = dict(
        schema_version=owner.VERSION,
        parent_key="root",
        target="check",
        request_id="old",
        previous_requests=13,
    )
    first = dict(
        base,
        schema_version=owner.CHAIN_VERSION,
        original=base,
        previous_requests=14,
        failed_successor=dict(key="first", request_id="r1"),
    )
    return dict(
        first,
        schema_version=owner.RECOVERY_VERSION,
        original=first,
        previous_requests=15,
        failed_successor=dict(key="second", request_id="r2"),
    )


def test_version_three_keeps_both_children_without_extending_bounds():
    proof = proof_tree()
    checked = SimpleNamespace(proof=proof)
    assert [r["key"] for r in owner.failed_lineage(proof)] == ["first", "second"]
    value = owner.selection_for(checked, "third")
    assert value["schema_version"] == owner.RECOVERY_VERSION
    assert value["profile"] == owner.RECOVERY_PROFILE
    assert value["max_new_requests"] == 2 and value["seconds"] == 2700
    assert value["proof"]["previous_requests"] == 15
    for key in ["root", "first", "second"]:
        with pytest.raises(ValueError, match="generic_acquisition_distinct_successor"):
            owner.selection_for(checked, key)
    extra = dict(proof, original=proof, failed_successor=dict(key="third", request_id="r3"))
    with pytest.raises(ValueError, match="generic_acquisition_failed_chain_bound"):
        owner.failed_lineage(extra)


def test_either_failed_child_cancellation_is_live(tmp_path, monkeypatch):
    from threading import Event

    monkeypatch.setattr(owner, "directory", lambda k: tmp_path / k)
    inherited = Event()
    monkeypatch.setattr(owner.GenericReview, "cancellation_signal", lambda _: inherited)
    c = owner.FailedAcquisitionSuccessor.__new__(owner.FailedAcquisitionSuccessor)
    c.checked = SimpleNamespace(proof=proof_tree())
    signal = c.cancellation_signal()
    for key in ["first", "second"]:
        (tmp_path / key).mkdir()
        p = tmp_path / key / "cancel-request.json"
        p.write_text("{}")
        assert signal.is_set()
        p.unlink()
        assert not signal.is_set()
    inherited.set()
    assert signal.is_set()


def test_second_failed_client_cannot_be_used_to_start_unbounded_fourth_attempt(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(owner, "directory", lambda k: tmp_path / k)
    (tmp_path / "third").mkdir()
    (tmp_path / "third/input.json").write_text(
        json.dumps({"schema_version": owner.RECOVERY_VERSION})
    )
    with pytest.raises(ValueError, match="generic_acquisition_failed_chain_bound"):
        owner.chained_origin("root", "check", "old", "third", "r3", "unused")


@pytest.mark.parametrize("changed", [False, True])
def test_version_three_historical_restore_rechecks_original_proof_preserves_old_freeze(
    tmp_path, monkeypatch, changed
):
    monkeypatch.setattr(owner, "directory", lambda k: tmp_path / k)
    proof = proof_tree()
    proof["client_binding"] = {"path": "public-client.json"}
    checked = SimpleNamespace(proof=proof, binding={"case": "same"})
    value = owner.selection_for(checked, "third")
    value["implementation"] = {"old.py": "sha256:" + "a" * 64}
    for path, content in {
        "third/input.json": value,
        "third/binding.json": checked.binding,
        "second/acquisition-successor.json": value,
        "third/evidence/result.json": {"complete": True, "phases": [{"target": None}]},
    }.items():
        p = tmp_path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(content))
    if changed:
        checked = SimpleNamespace(proof=dict(proof, previous_requests=14), binding=checked.binding)
    monkeypatch.setattr(owner, "chained_origin", lambda *args: checked)
    calls = []
    monkeypatch.setattr(owner, "FailedAcquisitionSuccessor", lambda c, key: (c, key))
    monkeypatch.setattr(
        owner.pending, "confirmed_campaign", lambda c: calls.append(c) or "confirmed"
    )
    if changed:
        with pytest.raises(ValueError, match="generic_acquisition_restore_binding"):
            owner.restore(tmp_path / "third")
        assert not calls
    else:
        assert owner.restore(tmp_path / "third") == "confirmed"
    assert json.loads((tmp_path / "third/input.json").read_bytes()) == value


def test_original_accounting_includes_root_and_both_failed_children_once(tmp_path, monkeypatch):
    monkeypatch.setattr(owner, "directory", lambda k: tmp_path / k)
    c = owner.FailedAcquisitionSuccessor.__new__(owner.FailedAcquisitionSuccessor)
    c.checked = SimpleNamespace(proof=proof_tree())
    c._current = lambda: None
    results = {}
    for key, count in [("root", 13), ("first", 1), ("second", 1)]:
        r = {"new_requests": count, "phases": [{"run_id": key}]}
        results[key] = r
        p = tmp_path / key / "evidence/result.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps(r))
    c.documents = lambda key: SimpleNamespace(read=lambda _: (results[key],))
    values = c.original_accounting()
    assert [Path(v["directory"]).name for v in values] == ["root", "first", "second"]
    assert sum(v["requests"] for v in values) == 15


@pytest.mark.parametrize("fault", [None, "client", "budget", "identity"])
def test_v3_derivation_rebuilds_each_ancestor_and_retains_total(tmp_path, monkeypatch, fault):
    from packages.evidence_details import live_permission
    from packages.llm_provider import client_selection
    from packages.llm_provider import config as provider_config

    monkeypatch.setattr(owner, "directory", lambda key: tmp_path / key)
    root = SimpleNamespace(
        proof=dict(
            schema_version=owner.VERSION,
            parent_key="root",
            target="check",
            request_id="r0",
            previous_requests=37 if fault == "budget" else 13,
        ),
        campaign=SimpleNamespace(
            config=SimpleNamespace(schema_version="agent-provider-config/7.1.0")
        ),
        binding={"case": "same"},
    )
    monkeypatch.setattr(owner, "origin", lambda *a: root)
    client = {"path": "client.json", "client": {"test": "fixed"}}
    monkeypatch.setattr(
        client_selection,
        "load_selection",
        lambda path: (
            dict(client, client={"test": "changed"})
            if fault == "client" and path == "changed.json"
            else client
        ),
    )
    monkeypatch.setattr(provider_config, "with_http_client", lambda *a: {"synthetic": "config"})
    monkeypatch.setattr(live_permission, "validate_live_config", lambda v: v)
    calls = []

    def failed(checked, key, request):
        calls.append((checked.proof["schema_version"], key, request))
        return {"key": key, "request_id": request, "requests": 1}

    monkeypatch.setattr(owner, "failed_successor", failed)
    (tmp_path / "first").mkdir()
    (tmp_path / "first/input.json").write_text(json.dumps({"schema_version": owner.VERSION}))
    first = owner.chained_origin("root", "check", "r0", "first", "r1", "client.json")
    p = dict(first.proof)
    if fault == "identity":
        p["target"] = "foreign"
    (tmp_path / "second").mkdir()
    (tmp_path / "second/input.json").write_text(
        json.dumps({"schema_version": owner.CHAIN_VERSION, "proof": p})
    )
    calls.clear()
    if fault:
        reason = {
            "client": "recovery_client_changed",
            "budget": "cumulative_budget",
            "identity": "failed_chain_identity",
        }[fault]
        with pytest.raises(ValueError, match=reason):
            owner.chained_origin(
                "root",
                "check",
                "r0",
                "second",
                "r2",
                "changed.json" if fault == "client" else "client.json",
            )
    else:
        result = owner.chained_origin("root", "check", "r0", "second", "r2", "client.json")
        assert result.proof["previous_requests"] == 15
        assert result.proof["schema_version"] == owner.RECOVERY_VERSION
        assert calls == [(owner.VERSION, "first", "r1"), (owner.CHAIN_VERSION, "second", "r2")]
        assert result.proof["original"] == first.proof
        assert root.proof["previous_requests"] == 13


def test_v3_freeze_does_not_modify_protocol_after_original_persistence(tmp_path, monkeypatch):
    c = owner.FailedAcquisitionSuccessor.__new__(owner.FailedAcquisitionSuccessor)
    c.checked = SimpleNamespace(proof={"schema_version": owner.RECOVERY_VERSION})
    p = tmp_path / "protocol.json"

    def freeze(_):
        value = {
            "schema_version": "generic-evidence-campaign/1.0.0",
            "observations": {"client_implementation": {"public-client": "hash"}},
        }
        p.write_text(json.dumps(value))
        c.protocol = value
        return value

    monkeypatch.setattr(owner.AcquisitionSuccessor, "freeze", freeze)
    value = c.freeze()
    assert value == c.protocol == json.loads(p.read_bytes())


def test_v3_client_freeze_is_inside_observations_before_persistence(monkeypatch):
    c = owner.FailedAcquisitionSuccessor.__new__(owner.FailedAcquisitionSuccessor)
    c.checked = SimpleNamespace(proof={"schema_version": owner.RECOVERY_VERSION})
    original = {"proof": "same"}
    monkeypatch.setattr(owner.AcquisitionSuccessor, "freeze_observations", lambda _: dict(original))
    monkeypatch.setattr(owner, "client_implementation", lambda _: {"public-client": "hash"})
    assert c.freeze_observations() == dict(
        original, client_implementation={"public-client": "hash"}
    )
    assert original == {"proof": "same"}
