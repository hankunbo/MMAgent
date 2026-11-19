"""Explicit citation-safe successor; no reactivation of an old failed run."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import root_review_window as window
from tests.evaluation_preparation.test_review_shape_window import old_config


def parent_fixture():
    return SimpleNamespace(
        config=window.configuration(old_config()),
        protocol={"deadline": "2026-10-18T00:00:00+00:00"},
        base=SimpleNamespace(
            manifest=SimpleNamespace(checksum=digest("input")),
            task=SimpleNamespace(checksum=digest("task")),
        ),
        accepted_observations=lambda: {"a": "original"},
    )


def test_citation_successor_changes_only_explicit_wire_and_batch():
    parent = parent_fixture()
    before = parent.config.model_dump_json()
    new = window.configuration(
        parent.config, citation_projection=True, batch_size=3, citation_reassessment=True
    )
    a, b = parent.config.model_dump(mode="json"), new.model_dump(mode="json")
    original = next(r for r in a["roles"] if r["role"] == "review")
    selected = next(r for r in b["roles"] if r["role"] == "review")
    assert selected["prompt_version"] == "review-direction-2.7.0"
    selected.update(original)
    assert b["review_batching"]["batch_size"] == 3
    b["review_batching"]["batch_size"] = 1
    assert a == b and parent.config.model_dump_json() == before
    with pytest.raises(ValueError, match="original_configuration"):
        window.configuration(parent.config)
    with pytest.raises(ValueError, match="original_configuration"):
        window.configuration(
            old_config(), citation_projection=True, batch_size=3, citation_reassessment=True
        )
    with pytest.raises(ValueError, match="original_configuration"):
        window.configuration(parent.config, citation_reassessment=True)


def test_known_citation_refusal_requires_original_accepted_proof_not_error_label(monkeypatch):
    from tests.agent_provider.test_citation_origin_reuse import accepted

    f, _, _ = accepted(monkeypatch)
    assert f.requests.records()[-1].error == "scope"
    with pytest.raises(ValueError, match="unresolved_request_history"):
        window.confirmed_request_history(f.state, f.docs, f.requests)
    window.confirmed_request_history(f.state, f.docs, f.requests, citation_reassessment=True)
    record = [e["record"] for e in f.budget.journal.read() if e.get("phase") == "completed"][-1]
    original_error = record["error"]
    record["error"] = "timeout"
    with pytest.raises(ValueError):
        window.confirmed_request_history(f.state, f.docs, f.requests, citation_reassessment=True)
    record["error"] = original_error


def test_actual_batch_count_preserves_case_ceiling_and_original_deadline():
    parent = parent_fixture()
    proof = {"model_review_ids": list(range(9)), "review_target_ids": list(range(9))}
    tip = {"path": "old", "previous_attempts": [{"requests": 29}]}
    plan = window.plan_for(
        parent,
        {},
        tip,
        proof,
        authorization="citation wire successor",
        now=datetime.now(UTC),
        local_finalization="saved-citation-closure",
        review_batch_size=3,
        request_limit=4,
    )
    assert plan["schema_version"] == "multimodal-campaign-renewal/1.30.0"
    assert plan["max_new_requests"] == 4 and plan["max_model_dispatches"] == 3
    assert plan["planned_review_batches"] == 3 and plan["review_batch_size"] == 3
    assert plan["prior_deadline"] == parent.protocol["deadline"]
    assert plan["max_cumulative_requests"] == 40
    assert plan["new_vision_requests"] == 0 and not plan["new_public_search_allowed"]
    assert plan["provider"] == digest(window.configuration_for(parent.config, plan))
    for cap in (3, 6, 20):
        with pytest.raises(ValueError, match="scope_or_budget"):
            window.plan_for(
                parent,
                {},
                tip,
                proof,
                authorization="test",
                now=datetime.now(UTC),
                local_finalization="saved",
                review_batch_size=3,
                request_limit=cap,
            )
    tip["previous_attempts"][0]["requests"] = 37
    with pytest.raises(ValueError, match="scope_or_budget"):
        window.plan_for(
            parent,
            {},
            tip,
            proof,
            authorization="test",
            now=datetime.now(UTC),
            local_finalization="saved",
            review_batch_size=3,
            request_limit=4,
        )


def test_no_implicit_local_closure_or_invalid_batch_upgrade():
    parent = parent_fixture()
    proof = {"model_review_ids": ["a"], "review_target_ids": ["a"]}
    tip = {"path": "old", "previous_attempts": [{"requests": 0}]}
    for size in (0, 4, True, 1.5):
        with pytest.raises(ValueError, match="batch_size"):
            window.plan_for(
                parent,
                {},
                tip,
                proof,
                authorization="test",
                now=datetime.now(UTC),
                local_finalization="saved",
                review_batch_size=size,
            )
    with pytest.raises(ValueError, match="citation_closure_required"):
        window.plan_for(
            parent, {}, tip, proof, authorization="test", now=datetime.now(UTC), review_batch_size=3
        )


def test_original_cli_requires_explicit_root_mode_for_new_batch(monkeypatch, capsys):
    import sys

    from apps.multimodal_case import main
    from packages.multimodal_research.campaign_renewal import prepare

    calls = []
    monkeypatch.setattr(window, "prepare", lambda **kw: calls.append(kw) or {"ok": True})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "multimodal_case",
            "renewal-prepare",
            "--facts",
            "old",
            "--previous",
            "old",
            "--output",
            "new",
            "--mode",
            "review",
            "--authorization",
            "test",
            "--root-direction-review",
            "--local-finalization",
            "saved",
            "--review-batch-size",
            "3",
        ],
    )
    assert main() == 0 and calls[0]["review_batch_size"] == 3
    assert '"ok": true' in capsys.readouterr().out
    with pytest.raises(ValueError, match="requires_explicit_mode"):
        prepare(
            facts="old",
            previous="old",
            output="new",
            mode="review",
            authorization="test",
            review_batch_size=3,
        )
