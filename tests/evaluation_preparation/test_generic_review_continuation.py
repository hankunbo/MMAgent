"""Continuation keeps current-owner selection, exact unknowns and cumulative budget."""

import json
from types import SimpleNamespace as NS

import pytest

from packages.evaluation_preparation import generic_control, generic_run
from packages.evaluation_preparation import generic_pending_review as pending
from packages.evaluation_preparation import generic_review as cli
from packages.evaluation_preparation import generic_review_continuation as c
from tests.evaluation_preparation.test_generic_review_followup import config


def snapshot():
    return NS(
        key="failed",
        docs=NS(run_id="failed-run"),
        state=NS(policy=NS(items=[NS(id=x) for x in ("a", "b", "c", "d")])),
        inventory={"selected_ids": ["c", "d"]},
    )


def test_unstarted_work_needs_no_unknown_permission(monkeypatch):
    monkeypatch.setattr(
        c.reassessment,
        "acknowledged_selection",
        lambda *a, **kw: pytest.fail("unexpected acknowledgment"),
    )
    assert c.select_work(snapshot(), (), ["old"], None) == (["c", "d"], [])


def test_exact_new_acknowledgment_goes_through_original_owner(monkeypatch):
    calls = []
    monkeypatch.setattr(
        c.reassessment,
        "acknowledged_selection",
        lambda *a, **kw: calls.append((a, kw)) or (["a", "b"], ["proof"]),
    )
    assert c.select_work(snapshot(), ["request:new"], ["old"], "config") == (
        ["a", "b", "c", "d"],
        ["proof"],
    )
    assert calls[0][0][1] == ["request:new"]
    assert calls[0][1]["all_ids"] == ["a", "b", "c", "d"]


@pytest.mark.parametrize("ids,accepted", [([], []), (["outside"], []), (["c"], ["c"])])
def test_empty_outside_or_accepted_selection_refused(ids, accepted):
    s = snapshot()
    s.inventory["selected_ids"] = ids
    with pytest.raises(ValueError, match="no_safe_selected_work"):
        c.select_work(s, (), accepted, None)


def test_cycle_or_excessive_history_rejected_before_read():
    for seen in (("a",), ("1", "2", "3")):
        with pytest.raises(ValueError, match="limit_or_cycle"):
            c.origin("a", [], seen=seen)


def test_cli_continuation_not_old_unknown_replay(monkeypatch):
    calls = []
    monkeypatch.setattr(c, "run", lambda *a, **kw: calls.append((a, kw)) or {})
    assert (
        cli.main(
            [
                "--key",
                "root",
                "--review-key",
                "new",
                "--continue-review-key",
                "failed",
                "--acknowledge-reassessment-request",
                "request:new",
                "--preflight",
            ]
        )
        == 0
    )
    assert calls == [(("root", "new", "failed", ["request:new"]), {"preflight": True})]


@pytest.mark.parametrize(
    "extra",
    [
        dict(accepted_review_key="old"),
        dict(network_route="new"),
        dict(prior_review_key="old"),
        dict(acknowledge_unknown_request="request:old"),
    ],
)
def test_cannot_mix_old_permissions(extra):
    with pytest.raises(ValueError, match="explicit_scope"):
        cli.run_review("root", "new", continue_review_key="failed", **extra)


def entry(tmp_path, monkeypatch):
    for key in ("root", "failed"):
        (tmp_path / key).mkdir()
        (tmp_path / key / "binding.json").write_text('{"root":true}')
    checked = NS(
        base=NS(
            origin=NS(campaign=NS(spec={"key": "root"})),
            accepted=NS(config=config()),
            proof={"prior_review_key": "original-failure"},
        ),
        history=[NS(key="failed")],
        proof={
            "parent_key": "root",
            "accepted_review_key": "accepted",
            "continuation_of": "failed",
            "acknowledge_requests": ["request:new"],
            "max_new_requests": 10,
            "previous_requests": 12,
            "provider_checksum": "config",
            "selected_ids": ["a", "b"],
        },
    )
    calls = []
    monkeypatch.setattr(c, "directory", lambda key: tmp_path / key)
    monkeypatch.setattr(c, "origin", lambda *a, **kw: checked)
    monkeypatch.setattr("packages.llm_provider.network_route.route_opener", lambda *a: None)

    def construct(*a):
        (tmp_path / "new/evidence").mkdir(parents=True, exist_ok=True)
        return NS(execute=lambda: calls.append("execute") or {"complete": False})

    monkeypatch.setattr(c, "ContinuationReview", construct)
    monkeypatch.setattr(
        generic_control,
        "start",
        lambda value: (tmp_path / "new/started.json").write_text(json.dumps(value)),
    )
    monkeypatch.setattr(generic_run, "persist_export", lambda key, campaign, result: result)
    return checked, calls


def test_preflight_no_claim_and_stopped_run_no_redispatch(tmp_path, monkeypatch):
    _, calls = entry(tmp_path, monkeypatch)
    result = c.run("root", "new", "failed", ["request:new"], preflight=True)
    assert result["api_requests"] == result["database_writes"] == 0
    assert not (tmp_path / "new").exists()
    assert not (tmp_path / "failed/review-successor.json").exists()
    c.run("root", "new", "failed", ["request:new"])
    with pytest.raises(ValueError, match="claimed_use_status"):
        c.run("root", "new", "failed", ["request:new"])
    assert calls == ["execute"]


def test_saved_export_uses_zero_dispatch_restore(tmp_path, monkeypatch):
    _, calls = entry(tmp_path, monkeypatch)
    c.run("root", "new", "failed", ["request:new"])
    (tmp_path / "new/evidence/result.json").write_text("{}")
    monkeypatch.setattr(c, "restore", lambda p: (None, {"complete": True}))
    assert c.run("root", "new", "failed", ["request:new"]) == {"complete": True}
    assert calls == ["execute"]


def test_conflicting_successor_refused_in_preflight(tmp_path, monkeypatch):
    _, calls = entry(tmp_path, monkeypatch)
    (tmp_path / "failed/review-successor.json").write_text('{"other":true}')
    with pytest.raises(ValueError, match="successor_conflict"):
        c.run("root", "new", "failed", ["request:new"], preflight=True)
    assert not calls


def test_cold_restore_dispatches_version_17(tmp_path, monkeypatch):
    from packages.evaluation_preparation import generic_input

    (tmp_path / "input.json").write_text(json.dumps({"schema_version": c.VERSION}))
    monkeypatch.setattr(generic_input, "directory", lambda _: tmp_path)
    monkeypatch.setattr(c, "restore", lambda p: "original-owner")
    assert pending.restore_campaign(tmp_path) == "original-owner"


@pytest.mark.parametrize("changed", [None, "claim", "binding", "identity", "budget"])
def test_origin_keeps_failed_request_in_budget_and_exact_claim(tmp_path, monkeypatch, changed):
    for key in ("root", "accepted", "failed"):
        (tmp_path / key).mkdir()
        (tmp_path / key / "binding.json").write_text('{"root":true}')
    base = NS(
        proof={
            "parent_key": "root",
            "accepted_review_key": "accepted",
            "previous_requests": 39 if changed == "budget" else 11,
        },
        snapshot=NS(ids=["old"]),
    )
    selection = {
        "schema_version": c.reassessment.VERSION,
        "key": "failed",
        "review_key": "failed",
        "accepted_review_key": "accepted",
        "proof": {"acknowledge_requests": ["request:old"]},
    }
    expected = {k: v for k, v in selection.items() if k != "key"}
    (tmp_path / "accepted/review-successor.json").write_text(
        json.dumps(expected if changed != "claim" else {})
    )
    if changed == "identity":
        selection["key"] = "wrong"
    if changed == "binding":
        (tmp_path / "failed/binding.json").write_text("{}")
    (tmp_path / "failed/input.json").write_text(json.dumps(selection))
    monkeypatch.setattr(c, "directory", lambda key: tmp_path / key)
    monkeypatch.setattr(c.reassessment, "origin", lambda *a: base)
    monkeypatch.setattr(c.reassessment, "selection_for", lambda *a: expected)
    monkeypatch.setattr(c.reassessment, "ReassessmentReview", lambda *a: NS(config=config()))
    s = snapshot()
    s.reservations, s.proof = 1, {"failed": "exact"}
    monkeypatch.setattr(c, "failed_snapshot", lambda *a: s)
    if changed:
        with pytest.raises(ValueError, match="generic_continuation_"):
            c.origin("failed", ())
    else:
        result = c.origin("failed", ())
        assert result.proof["previous_requests"] == 12
        assert result.proof["selected_ids"] == ["c", "d"]
        assert result.proof["max_new_requests"] == 10
        assert result.proof["failed_history"] == [{"failed": "exact"}]


def test_aggregation_keeps_all_failed_and_accepted_ancestors(tmp_path, monkeypatch):
    from packages.grounding.models import digest

    (tmp_path / "root/evidence").mkdir(parents=True)
    root = {"new_requests": 7, "phases": [{"run_id": "root-run"}]}
    (tmp_path / "root/evidence/result.json").write_text(json.dumps(root))
    current = object.__new__(c.ContinuationReview)
    current.prior = lambda: {"reservations": 12}
    current.acquisition = NS(
        output=tmp_path / "root/evidence", documents=lambda _: NS(read=lambda _: (root,))
    )
    current.chain = NS(
        proof={"prior_review_key": "failed-first"}, child=NS(result={"first": True}, reserved=1)
    )
    current.reassessment = NS(
        proof={"accepted_review_key": "accepted"},
        snapshot=NS(result={"accepted": True}, reservations=3),
    )
    current.continuation = NS(
        history=[NS(key="failed-latest", result={"latest": True}, reservations=1)]
    )
    monkeypatch.setattr(c.reassessment, "directory", lambda key: tmp_path / key)
    rows = c.reassessment.aggregation_history(current)
    assert sum(v["requests"] for v in rows) == 12
    assert [v["directory"] for v in rows] == [
        str(tmp_path / p) for p in ("root", "failed-first", "accepted", "failed-latest")
    ]
    assert rows[-1]["checksum"] == digest({"latest": True})
    current.continuation.history[0].reservations = 0
    with pytest.raises(ValueError, match="accounting_ancestry"):
        c.reassessment.aggregation_history(current)
