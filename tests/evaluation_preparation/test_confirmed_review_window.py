"""Exact same-profile continuation, not a generic retry or a nonzero-score loop."""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import campaign_renewal as renewal
from packages.multimodal_research import confirmed_review_window as window
from packages.research_runtime.agent_records import RuntimeAgentJournal
from tests.agent_provider.test_review_continuation import continuation_fixture


def origin(tmp_path, monkeypatch):
    from packages.multimodal_research import campaign

    f = continuation_fixture(single_batches=True)
    docs = f.old.documents
    state = f.old.owner.current()
    docs.write("dynamic-phase-input", (state,))
    journal = RuntimeAgentJournal(docs)
    for event in f.old.budget.journal.read():
        journal.append(event)
    control = docs.sessions.read(docs.run_id)
    now = control.binding.deadline + timedelta(seconds=1)
    phase = dict(
        run_id=docs.run_id, target=None, accepted=False, acceptance_key=None, error="deadline"
    )
    result = dict(complete=False, phases=[phase])
    docs.write("dynamic-phase-result", (phase,))
    docs.write("multimodal-case-result", (result,))
    (tmp_path / "result.json").write_text(json.dumps(result))
    parent = SimpleNamespace(
        config=f.old.config,
        documents=lambda _: docs,
        sessions=docs.sessions,
        protocol=dict(deadline=control.binding.deadline.isoformat()),
        base=SimpleNamespace(
            contexts={},
            manifest=SimpleNamespace(checksum=digest("input")),
            task=SimpleNamespace(checksum=digest("task")),
        ),
        accepted_observations=lambda: {},
    )
    tip = dict(
        path=str(tmp_path),
        run_id=docs.run_id,
        previous_attempts=[dict(requests=2, directory=str(tmp_path), checksum=digest(result))],
    )
    checks = []
    saved_context = SimpleNamespace(run_id=docs.run_id)
    docs.contexts = SimpleNamespace(
        read_research_run=lambda key: saved_context if key == docs.run_id else None
    )

    class Projection:
        def __init__(self, *args, **kwargs):
            pass

        def project(self, **kwargs):
            assert parent.base.contexts[docs.run_id] is saved_context
            return f.old.owner.current(), None

        def verify(self, expected):
            assert f.old.owner.current() == expected
            checks.append(True)

    monkeypatch.setattr(campaign, "Phase", Projection)
    return f, parent, result, tip, now, journal, checks


def test_exact_original_reader_keeps_config_prompt_and_all_confirmed_bytes(tmp_path, monkeypatch):
    f, parent, result, tip, now, _, checks = origin(tmp_path, monkeypatch)
    before = digest(f.old.documents.saved), digest(f.old.budget.journal.read())
    reader = renewal.historical_reader(parent, tip["run_id"], preserve_prompt=True)
    assert reader.state_reader() == f.old.owner.current()
    proof = window.origin_proof(parent, result, tip, now=now)
    assert len(proof["confirmed_assessment_ids"]) == 2
    assert len(proof["pending_batch_ids"]) == 3 and checks
    assert proof["configuration_checksum"] == digest(parent.config)
    assert before == (digest(f.old.documents.saved), digest(f.old.budget.journal.read()))
    # No implicit migration of the legacy caller's compact-prompt behavior.
    old = renewal.historical_reader(parent, tip["run_id"]).state_reader()
    assert old.prompt_checksum == next(
        r.prompt_checksum for r in renewal.compact_configuration().roles if r.role == "review"
    )


@pytest.mark.parametrize(
    "fault",
    ["cancel", "early", "accepted", "error", "unknown", "pending_claim", "source", "response"],
)
def test_origin_rejects_invalid_or_unknown_without_dispatch(tmp_path, monkeypatch, fault):
    f, parent, result, tip, now, journal, _ = origin(tmp_path, monkeypatch)
    if fault == "cancel":
        (tmp_path / "cancel").touch()
    elif fault == "early":
        now -= timedelta(seconds=2)
    elif fault == "accepted":
        result["complete"] = True
    elif fault == "error":
        result["phases"][0]["error"] = "timeout"
    elif fault == "unknown":
        journal.append(dict(phase="reserved", request_id="unknown"))
    elif fault == "pending_claim":
        f.old.documents.write("review-batches:origin:A:batch:002:claim", ({"unknown": True},))
    elif fault == "source":

        def withdrawn():
            raise ValueError("source_withdrawn")

        monkeypatch.setattr(f.old.owner, "current", withdrawn)
    else:
        f.old.documents.saved["review-batches:origin:A:batch:000:response"][0][
            "raw_model_text"
        ] += " "
    count = len(f.old.budget.records())
    with pytest.raises(ValueError):
        window.origin_proof(parent, result, tip, now=now)
    assert len(f.old.budget.records()) == count


def test_plan_budget_uses_pending_batches_not_total_items_or_scores(tmp_path, monkeypatch):
    _, parent, result, tip, now, _, _ = origin(tmp_path, monkeypatch)
    proof = window.origin_proof(parent, result, tip, now=now)
    plan = window.plan_for(
        parent, result, tip, proof, output=tmp_path / "next", authorization="test", now=now
    )
    assert plan["provider"] == digest(parent.config)
    assert plan["max_new_requests"] == 5 and plan["max_model_dispatches"] == 4
    assert plan["seconds"] == 5400 and plan["new_vision_requests"] == 0
    assert plan["prior_deadline"] == parent.protocol["deadline"]
    monkeypatch.setattr(window, "origin_proof", lambda *args: proof)
    window.check(plan, parent, result)
    for field, value in (
        ("deadline", now.isoformat()),
        ("provider", digest("other")),
        ("max_model_dispatches", 5),
        ("acquisition_targets", ["other"]),
        ("same_profile_review", {}),
    ):
        with pytest.raises(ValueError):
            window.check({**plan, field: value}, parent, result)
    for limit in (0, 3, 6, 13):
        with pytest.raises(ValueError, match="scope_or_budget"):
            window.plan_for(
                parent,
                result,
                tip,
                proof,
                output=tmp_path / "next",
                authorization="test",
                now=now,
                request_limit=limit,
            )
    tip["previous_attempts"][0]["requests"] = 37
    with pytest.raises(ValueError, match="scope_or_budget"):
        window.plan_for(
            parent, result, tip, proof, output=tmp_path / "next", authorization="test", now=now
        )


def test_cli_selects_new_mode_and_cannot_mix_acquisition_or_repair(monkeypatch, tmp_path):
    import sys

    from apps import multimodal_case

    calls = []
    monkeypatch.setattr(
        window, "prepare", lambda **kwargs: calls.append(kwargs) or {"api_requests": 0}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "multimodal_case",
            "renewal-prepare",
            "--facts",
            str(tmp_path),
            "--previous",
            str(tmp_path),
            "--output",
            str(tmp_path / "new"),
            "--mode",
            "review",
            "--authorization",
            "test",
            "--same-profile-review",
        ],
    )
    assert multimodal_case.main() == 0
    assert len(calls) == 1 and calls[0]["reuse_confirmed_review"] is False


@pytest.mark.parametrize(
    "extra",
    [
        {"mode": "acquisition"},
        {"unstarted_context_review": True},
        {"repair_review_format": True},
        {"acquisition_target": ["x"]},
    ],
)
def test_new_mode_rejects_mixed_options_before_reading_owner(tmp_path, extra):
    args = dict(
        facts=tmp_path,
        previous=tmp_path,
        output=tmp_path / "next",
        mode="review",
        authorization="test",
    )
    with pytest.raises(ValueError, match="explicit_text_only_scope"):
        window.prepare(**(args | extra))


def test_new_version_review_reserve_remains_bounded(tmp_path):
    c = object.__new__(renewal.RenewedCampaign)
    c.base = SimpleNamespace(check=lambda: None, detailed=True)
    c.output = tmp_path
    c.protocol = dict(
        schema_version=window.VERSION,
        max_new_requests=5,
        previous_attempts=[],
        deadline=(datetime.now(UTC) + timedelta(seconds=180)).isoformat(),
    )
    c.usage = lambda: 2
    c.previous_attempts = lambda: []
    with c.phase_context(None):
        c.guard()
    assert not c._final_review_active


def test_prepare_freezes_exact_proof_and_unique_original_owner_claim(tmp_path, monkeypatch):
    f, parent, result, tip, now, _, _ = origin(tmp_path, monkeypatch)
    proof = window.origin_proof(parent, result, tip, now=now)
    monkeypatch.setattr(renewal, "completed_parent", lambda _: (parent, result))
    monkeypatch.setattr(renewal, "stopped_tip", lambda *a, **kw: tip)
    monkeypatch.setattr(window, "origin_proof", lambda *a: proof)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr(window, "datetime", Clock)
    out = tmp_path / "next"
    answer = renewal.prepare(
        facts=tmp_path,
        previous=tmp_path,
        output=out,
        mode="review",
        authorization="test",
        same_profile_review=True,
    )
    plan = json.loads((out / "protocol.json").read_bytes())
    assert plan["schema_version"] == window.VERSION and plan["same_profile_review"] == proof
    assert answer["api_requests"] == 0 and answer["pending_batches"] == 3
    assert f.old.documents.read("authorized-campaign-renewal") == (
        dict(output=str(out), plan_checksum=digest(plan)),
    )
    window.check(plan, parent, result)
    with pytest.raises(ValueError, match="already_claimed"):
        renewal.prepare(
            facts=tmp_path,
            previous=tmp_path,
            output=tmp_path / "another",
            mode="review",
            authorization="test",
            same_profile_review=True,
        )
    assert not (tmp_path / "another").exists()
