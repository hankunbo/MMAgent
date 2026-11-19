"""Original-owner boundary tests; real owner integration has a separate PG test."""

import json
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation import generic_control, generic_run
from packages.evaluation_preparation import generic_reassessment as r
from packages.grounding.models import digest
from tests.evaluation_preparation.test_generic_review_followup import config


def inventory(monkeypatch, *, violation=None):
    histories = [
        ("root", SimpleNamespace(run_id="r0"), "s0"),
        ("child", SimpleNamespace(run_id="r1"), "s1"),
    ]
    calls = []

    def read(*, documents, state, config, acknowledge_request=None):
        ix = 0 if documents.run_id == "r0" else 1
        request = "request:" + str(ix)
        ids = ["a", "b"] if ix == 0 else ["c", "d"]
        if violation == "overlap" and ix == 1:
            ids = ["a", "d"]
        calls.append((documents.run_id, acknowledge_request))
        allowed = request == acknowledge_request
        return dict(
            selected_ids=ids if allowed else ["unsent"],
            journal_checksum="journal" + str(ix),
            control_checksum="control" + str(ix),
            quarantined_batches=[
                dict(
                    request_ids=[request],
                    assessment_ids=ids,
                    new_reassessment_authorized=allowed,
                    old_response_state_unchanged=violation != "rewritten",
                )
            ],
        )

    monkeypatch.setattr(r, "read_failed_review_attempt", read)
    return histories, calls


def test_exact_owner_requests_not_all_unknown_or_unsent(monkeypatch):
    histories, calls = inventory(monkeypatch)
    ids, proofs = r.acknowledged_selection(
        histories,
        ["request:1"],
        accepted_ids=["accepted"],
        all_ids=["a", "b", "c", "d", "unsent"],
        config=None,
    )
    assert ids == ["c", "d"]
    assert proofs[0]["history"] == "child"
    assert calls == [("r0", None), ("r1", None), ("r1", "request:1")]


@pytest.mark.parametrize(
    "requests", [[], ["bad"], ["request:0", "request:0"], ["request:" + str(i) for i in range(5)]]
)
def test_acknowledgments_are_explicit_and_bounded(monkeypatch, requests):
    histories, calls = inventory(monkeypatch)
    with pytest.raises(ValueError, match="exact_requests_required"):
        r.acknowledged_selection(histories, requests, accepted_ids=[], all_ids=[], config=None)
    assert not calls


@pytest.mark.parametrize("kind", ["accepted", "outside", "overlap", "rewritten", "unknown"])
def test_inconsistent_scope_never_authorized(monkeypatch, kind):
    histories, _ = inventory(monkeypatch, violation=kind)
    with pytest.raises(ValueError, match="generic_reassessment_"):
        r.acknowledged_selection(
            histories,
            ["request:0", "request:1"] if kind != "unknown" else ["request:2"],
            accepted_ids=["a"] if kind == "accepted" else [],
            all_ids=["a", "b", "c", "d"] if kind != "outside" else ["a", "b"],
            config=None,
        )


def entry(tmp_path, monkeypatch):
    for key in ("root", "prior", "accepted"):
        (tmp_path / key).mkdir()
        (tmp_path / key / "binding.json").write_text('{"original":true}')
    (tmp_path / "root/review-successor.json").write_text('{"old":"claim"}')
    checked = SimpleNamespace(
        origin=SimpleNamespace(campaign=SimpleNamespace(spec=dict(key="root"))),
        accepted=SimpleNamespace(config=config()),
        proof=dict(
            parent_key="root",
            prior_review_key="prior",
            accepted_review_key="accepted",
            selected_ids=["c", "d"],
            previous_requests=11,
            max_new_requests=10,
        ),
    )
    calls = []
    monkeypatch.setattr(r, "directory", lambda key: tmp_path / key)
    monkeypatch.setattr(r, "origin", lambda *a: checked)
    monkeypatch.setattr(
        "packages.llm_provider.network_route.route_opener", lambda *a: calls.append("route")
    )

    def construct(*a):
        (tmp_path / "next/evidence").mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(execute=lambda: calls.append("execute") or {"complete": False})

    monkeypatch.setattr(r, "ReassessmentReview", construct)
    monkeypatch.setattr(
        generic_control,
        "start",
        lambda value: (tmp_path / "next/started.json").write_text(json.dumps(value)),
    )
    monkeypatch.setattr(generic_run, "persist_export", lambda key, c, result: dict(result=result))
    return checked, calls


def test_entry_preflight_does_not_claim_and_running_repeat_no_dispatch(tmp_path, monkeypatch):
    checked, calls = entry(tmp_path, monkeypatch)
    before = (tmp_path / "root/review-successor.json").read_bytes()
    result = r.run("root", "next", "accepted", ["request:x"], preflight=True)
    assert result["api_requests"] == result["database_writes"] == 0
    assert not (tmp_path / "next").exists()
    assert not (tmp_path / "accepted/review-successor.json").exists()
    r.run("root", "next", "accepted", ["request:x"])
    with pytest.raises(ValueError, match="claimed_use_status"):
        r.run("root", "next", "accepted", ["request:x"])
    assert calls.count("execute") == 1
    assert (tmp_path / "root/review-successor.json").read_bytes() == before
    claim = json.loads((tmp_path / "accepted/review-successor.json").read_bytes())
    assert claim["proof"]["previous_requests"] == 11


def test_saved_result_recovery_uses_confirmation_not_dispatch(tmp_path, monkeypatch):
    checked, calls = entry(tmp_path, monkeypatch)
    r.run("root", "next", "accepted", ["request:x"])
    (tmp_path / "next/evidence/result.json").write_text('{"saved":true}')
    monkeypatch.setattr(
        r, "restore", lambda path: calls.append("restore") or (None, {"complete": True})
    )
    result = r.run("root", "next", "accepted", ["request:x"])
    assert result["result"]["complete"] is True
    assert calls.count("execute") == 1 and calls.count("restore") == 1


def test_cancelled_accepted_history_rejected_before_owner_or_api(tmp_path, monkeypatch):
    (tmp_path / "cancel-request.json").write_text("{}")
    monkeypatch.setattr(r, "directory", lambda key: tmp_path)
    with pytest.raises(ValueError, match="history_cancelled"):
        r.origin("accepted", ["request:x"])


@pytest.mark.parametrize("preflight", [False, True])
def test_claim_conflict_rejected_before_construction(tmp_path, monkeypatch, preflight):
    checked, calls = entry(tmp_path, monkeypatch)
    (tmp_path / "accepted/review-successor.json").write_text('{"other":"child"}')
    with pytest.raises(ValueError, match="successor_conflict"):
        r.run("root", "next", "accepted", ["request:x"], preflight=preflight)
    assert "execute" not in calls and not (tmp_path / "next").exists()


@pytest.mark.parametrize(
    "fault",
    [None, "journal", "control", "result", "report", "acceptance", "selection", "claim", "binding"],
)
def test_accepted_ancestor_rechecked_for_each_dispatch(tmp_path, monkeypatch, fault):
    cls = r.ReassessmentReview
    checked, calls = entry(tmp_path, monkeypatch)
    selection = dict(accepted="current", key="accepted")
    (tmp_path / "accepted/input.json").write_text(json.dumps(selection))
    (tmp_path / "prior/review-successor.json").write_text(json.dumps({"accepted": "current"}))
    values = {
        "dynamic-case-result": ({"old": True},),
        "dynamic-case-report": ({"report": True},),
        "accepted:k": ({"accepted": True},),
    }
    journal = [dict(phase="reserved")] * 3
    control = dict(stopped=True)
    docs = SimpleNamespace(run_id="accepted-run", read=lambda key: values[key])
    snapshot = SimpleNamespace(
        docs=docs,
        result=values["dynamic-case-result"][0],
        report=values["dynamic-case-report"],
        reservations=3,
        proof=dict(
            journal=digest(journal),
            control=digest(control),
            acceptance_key="k",
            acceptance=digest(values["accepted:k"]),
        ),
    )
    checked.snapshot = snapshot
    checked.selection = selection
    campaign = object.__new__(cls)
    campaign.reassessment = checked
    campaign.sessions = SimpleNamespace(read=lambda _: control)
    monkeypatch.setattr(r.followup.FollowupReview, "prior", lambda _: dict(reservations=8))
    monkeypatch.setattr(r, "RuntimeAgentJournal", lambda _: SimpleNamespace(read=lambda: journal))
    if fault == "journal":
        journal.append({"changed": True})
    elif fault == "control":
        control["cancelled"] = True
    elif fault in {"result", "report", "acceptance"}:
        key = {
            "result": "dynamic-case-result",
            "report": "dynamic-case-report",
            "acceptance": "accepted:k",
        }[fault]
        values[key] = ({"changed": True},)
    elif fault in {"selection", "claim", "binding"}:
        path = {
            "selection": "accepted/input.json",
            "claim": "prior/review-successor.json",
            "binding": "accepted/binding.json",
        }[fault]
        (tmp_path / path).write_text("{}")
    if fault:
        with pytest.raises(ValueError, match="accepted_history_changed"):
            campaign.prior()
    else:
        assert campaign.prior()["reservations"] == 11


@pytest.mark.parametrize(
    "field", ["seconds", "max_cumulative_requests", "provider_checksum", "proof"]
)
def test_cold_readback_rejects_changed_selection(tmp_path, monkeypatch, field):
    checked, calls = entry(tmp_path, monkeypatch)
    (tmp_path / "next").mkdir()
    selection = dict(r.selection_for(checked, "next"), key="next")
    selection["proof"] = dict(selection["proof"], acknowledge_requests=["request:x"])
    if field == "proof":
        selection["proof"]["previous_requests"] = 0
    else:
        selection[field] = "changed"
    (tmp_path / "next/input.json").write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="saved_selection_changed"):
        r.restore(tmp_path / "next")
    assert "execute" not in calls


@pytest.mark.parametrize("new", [False, True])
@pytest.mark.parametrize("fault", [None, "total", "bound_root", "duplicate_path"])
def test_aggregation_history_counts_failed_middle_once(tmp_path, monkeypatch, new, fault):
    cls = r.ReassessmentReview if new else r.followup.FollowupReview
    c = object.__new__(cls)
    root = dict(phases=[dict(run_id="root-run")], new_requests=7)
    (tmp_path / "root/evidence").mkdir(parents=True)
    (tmp_path / "root/evidence/result.json").write_text(json.dumps(root))
    bound = root if fault != "bound_root" else dict(root, new_requests=0)
    c.acquisition = SimpleNamespace(
        output=tmp_path / "root/evidence",
        documents=lambda _: SimpleNamespace(read=lambda _: (bound,)),
    )
    c.chain = SimpleNamespace(
        proof=dict(prior_review_key="prior" if fault != "duplicate_path" else "root"),
        child=SimpleNamespace(result=dict(failed=True, new_requests=1), reserved=1),
    )
    if new:
        c.reassessment = SimpleNamespace(
            proof=dict(accepted_review_key="accepted"),
            snapshot=SimpleNamespace(result=dict(accepted=True, new_requests=3), reservations=3),
        )
    monkeypatch.setattr(r, "directory", lambda key: tmp_path / key)
    monkeypatch.setattr(
        cls, "prior", lambda _: dict(reservations=(11 if new else 8) + (fault == "total"))
    )
    if fault:
        with pytest.raises(ValueError, match="accounting_"):
            r.aggregation_history(c)
    else:
        history = r.aggregation_history(c)
        assert [x["requests"] for x in history] == ([7, 1, 3] if new else [7, 1])
        assert len(history) == (3 if new else 2)
