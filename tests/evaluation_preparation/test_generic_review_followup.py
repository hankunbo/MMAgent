"""Bounded route followup, without keys, requests, tools or research DB writes."""

import json
from types import SimpleNamespace

import pytest

from apps.vectorengine_single_case import load_config
from packages.evaluation_preparation import generic_review as r
from packages.evaluation_preparation import generic_review_followup as f
from packages.grounding.models import digest

ROUTE = "configs/agents/server_local_https_route_v1.json"


def config():
    from packages.llm_provider.config import read_provider_config

    raw = load_config("configs/agents/fairvlmed_relay_local_gpt6_review_v1.json").model_dump(
        mode="json"
    )
    raw["network_route"] = f.read("configs/agents/environment_https_route_v1.json")
    raw["transport"]["max_retries"] = 1
    return read_provider_config(raw)


def test_route_only_change_matches_publicly_verified_provider():
    origin = config()
    current = f.configuration(origin, ROUTE)
    assert current == load_config("configs/agents/fairvlmed_relay_local_gpt6_review_v1.json")
    a, b = origin.model_dump(mode="json"), current.model_dump(mode="json")
    assert {k for k in a if a[k] != b[k]} == {"network_route", "transport"}
    assert b["transport"] == dict(a["transport"], max_retries=0)


@pytest.mark.parametrize("fault", ["same", "private", "retry", "batch", "prompt", "version"])
def test_invalid_route_or_original_contract_rejected(fault):
    origin, path = config(), ROUTE
    if fault == "same":
        path = "configs/agents/environment_https_route_v1.json"
    elif fault == "private":
        path = ".local/vectorengine/config.json"
    elif fault == "retry":
        origin = origin.model_copy(
            update={"transport": origin.transport.model_copy(update={"max_retries": 2})}
        )
    elif fault == "batch":
        origin = origin.model_copy(
            update={"review_batching": origin.review_batching.model_copy(update={"batch_size": 1})}
        )
    elif fault == "prompt":
        origin = origin.model_copy(
            update={
                "roles": tuple(
                    x.model_copy(update={"prompt_version": "wrong"}) if x.role == "review" else x
                    for x in origin.roles
                )
            }
        )
    else:
        origin = origin.model_copy(update={"schema_version": "agent-provider-config/7.0.0"})
    with pytest.raises(ValueError, match="generic_followup_"):
        f.configuration(origin, path)


@pytest.mark.parametrize(
    "extra",
    [
        dict(acknowledge_unknown_request="request:x"),
        dict(review_profile="old"),
        dict(method_origin="old"),
    ],
)
def test_followup_is_not_an_unknown_reassessment_or_other_method(extra):
    with pytest.raises(ValueError, match="explicit_scope"):
        r.run_review("root", "next", prior_review_key="prior", network_route=ROUTE, **extra)


def setup_entry(tmp_path, monkeypatch):
    for name in ("root", "prior"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "binding.json").write_text('{"original":true}')
    root_claim = b'{"old_claim":"immutable"}'
    (tmp_path / "root/review-successor.json").write_bytes(root_claim)
    checked = SimpleNamespace(
        config=config(),
        campaign=SimpleNamespace(spec=dict(key="root")),
        proof=dict(
            parent_key="root",
            prior_review_key="prior",
            route_path=ROUTE,
            max_new_requests=10,
            previous_requests=8,
            selected_ids=["unsent"],
        ),
    )
    calls = []
    monkeypatch.setattr(f, "directory", lambda k: tmp_path / k)
    monkeypatch.setattr(f, "origin", lambda *a: checked)
    monkeypatch.setattr(f, "route_opener", lambda *a: calls.append("route"))
    from packages.evaluation_preparation import generic_control, generic_run

    def construct(*a):
        (tmp_path / "next/evidence").mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(execute=lambda: calls.append("execute") or dict(complete=False))

    monkeypatch.setattr(f, "FollowupReview", construct)
    monkeypatch.setattr(
        generic_control,
        "start",
        lambda value: (tmp_path / "next/started.json").write_text(json.dumps(value)),
    )
    monkeypatch.setattr(generic_run, "persist_export", lambda k, c, result: dict(result=result))
    return calls, checked, root_claim


def test_preflight_no_new_key_claim_and_repeat_no_dispatch(tmp_path, monkeypatch):
    calls, checked, root_claim = setup_entry(tmp_path, monkeypatch)
    result = f.run("root", "next", "prior", ROUTE, preflight=True)
    assert result["api_requests"] == result["database_writes"] == 0
    assert (
        not (tmp_path / "next").exists() and not (tmp_path / "prior/review-successor.json").exists()
    )
    f.run("root", "next", "prior", ROUTE)
    assert calls.count("execute") == 1
    assert (tmp_path / "root/review-successor.json").read_bytes() == root_claim
    claim = json.loads((tmp_path / "prior/review-successor.json").read_bytes())
    assert claim["schema_version"] == f.VERSION and claim["proof"]["previous_requests"] == 8
    with pytest.raises(ValueError, match="claimed_use_status"):
        f.run("root", "next", "prior", ROUTE)
    assert calls.count("execute") == 1


def test_failed_preflight_does_not_consume_child_claim(tmp_path, monkeypatch):
    calls, checked, root_claim = setup_entry(tmp_path, monkeypatch)

    def fail(*a):
        raise ValueError("current_source_changed")

    monkeypatch.setattr(f, "origin", fail)
    with pytest.raises(ValueError, match="source_changed"):
        f.run("root", "next", "prior", ROUTE)
    assert not (tmp_path / "prior/review-successor.json").exists()
    assert not calls


def test_actual_cli_dispatches_explicit_followup_preflight(monkeypatch):
    calls = []
    monkeypatch.setattr(f, "run", lambda *a, **kw: calls.append((a, kw)) or {})
    assert (
        r.main(
            [
                "--key",
                "root",
                "--review-key",
                "next",
                "--prior-review-key",
                "prior",
                "--network-route",
                ROUTE,
                "--preflight",
            ]
        )
        == 0
    )
    assert calls == [(("root", "next", "prior", ROUTE), dict(preflight=True))]


@pytest.mark.parametrize(
    "fault",
    [None, "root_journal", "root_control", "child_journal", "child_control", "claim", "result"],
)
def test_every_dispatch_preserves_ancestor_control_journals_and_budget(
    tmp_path, monkeypatch, fault
):
    journals = {"root-run": [{"reserved": 7}], "child-run": [{"reserved": 1}]}
    controls = {"root-run": {"stopped": True}, "child-run": {"stopped": True}}
    values = {"dynamic-case-result": ({"old": True},), "dynamic-case-report": ({"report": True},)}
    child_docs = SimpleNamespace(run_id="child-run", read=lambda k: values[k])
    root_docs = SimpleNamespace(run_id="root-run")
    child = SimpleNamespace(
        documents=child_docs,
        reserved=1,
        journal_checksum=digest(journals["child-run"]),
        inventory=dict(control_checksum=digest(controls["child-run"])),
        result=values["dynamic-case-result"][0],
        report=values["dynamic-case-report"],
        claim={"original": True},
        selection={"child": True},
    )
    proof = dict(
        previous_requests=8,
        prior_review_key="prior",
        original=dict(
            inventory=dict(
                run_id="root-run",
                journal_checksum=digest(journals["root-run"]),
                control_checksum=digest(controls["root-run"]),
            )
        ),
    )
    c = object.__new__(f.FollowupReview)
    c.chain = SimpleNamespace(child=child, proof=proof)
    c.spec = dict(key="root")
    c.sessions = SimpleNamespace(read=lambda run: controls[run])
    c.documents = lambda _: root_docs
    for key in ("root", "prior"):
        (tmp_path / key).mkdir()
    (tmp_path / "root/review-successor.json").write_text(json.dumps(child.claim))
    (tmp_path / "prior/input.json").write_text(json.dumps(child.selection))
    monkeypatch.setattr(f, "directory", lambda key: tmp_path / key)
    monkeypatch.setattr(
        f, "RuntimeAgentJournal", lambda docs: SimpleNamespace(read=lambda: journals[docs.run_id])
    )
    monkeypatch.setattr(r.GenericReview, "prior", lambda _: dict(reservations=7))
    if fault in {"root_journal", "child_journal"}:
        journals[fault.split("_")[0] + "-run"].append({"changed": True})
    elif fault in {"root_control", "child_control"}:
        controls[fault.split("_")[0] + "-run"]["cancelled"] = True
    elif fault == "claim":
        (tmp_path / "root/review-successor.json").write_text("{}")
    elif fault == "result":
        values["dynamic-case-result"] = ({"other": True},)
    if fault:
        with pytest.raises(ValueError, match="ancestor_changed"):
            c.prior()
    else:
        assert c.prior()["reservations"] == 8


@pytest.mark.parametrize(
    "field", ["seconds", "max_cumulative_requests", "schema_version", "input_checksum"]
)
def test_cold_restore_rejects_modified_selection_without_execution(tmp_path, monkeypatch, field):
    calls, checked, _ = setup_entry(tmp_path, monkeypatch)
    path = tmp_path / "next"
    path.mkdir()
    selection = dict(f.selection_for(checked, "next"), key="next")
    selection[field] = "changed"
    (path / "input.json").write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="saved_selection_changed"):
        f.restore(path)
    assert "execute" not in calls
