"""Composition checks for the opt-in actual run adapter and saved report receipt."""

import copy
import json
from types import SimpleNamespace

import pytest

from packages.evidence_reuse.integration import DispatchCheck, saved_report_context
from packages.grounding.models import digest


def test_dispatch_check_not_a_persisted_authorization(monkeypatch):
    checker = object.__new__(DispatchCheck)
    from packages.evidence_reuse.cache import ShadowCache

    checker.cache = ShadowCache()
    checker.spec = dict(upstream_implementation={})
    checker.upstream = dict(
        skills=dict(
            plans=[dict(requirement_id="x", steps=[dict(port="segman")], missing_components=[])]
        )
    )
    counter = []

    def current():
        counter.append(1)
        return dict(active=True)

    monkeypatch.setattr(checker, "current", current)
    assert checker.needs_segman()
    checker.check()
    assert len(counter) == 4
    assert checker.cache.events[-1]["candidate_hit"]

    def revoked():
        raise ValueError("current_owner_revoked")

    monkeypatch.setattr(checker, "current", revoked)
    with pytest.raises(ValueError, match="current_owner_revoked"):
        checker.check()


def test_saved_report_carries_same_plan_receipt_and_wrong_case_rejected(tmp_path):
    plan = dict(upstream=dict(schema_version="evidence-reuse/1.0.0", components={}))
    receipt = dict(
        source_plan_checksum=digest(plan), case_code="CASE:A", input_checksum=digest("image")
    )
    (tmp_path / "source-plan.json").write_text(json.dumps(plan))
    path = tmp_path / "upstream-use.json"
    path.write_text(json.dumps(receipt))
    campaign = SimpleNamespace(
        base=SimpleNamespace(
            directory=tmp_path,
            binding=dict(case_code="CASE:A"),
            image_artifact=SimpleNamespace(checksum=digest("image")),
        )
    )
    result = saved_report_context(campaign)
    assert result["plan"] == plan["upstream"] and result["execution"] == receipt
    receipt["case_code"] = "CASE:B"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="report_binding"):
        saved_report_context(campaign)


def test_finish_retry_preserves_first_timing_without_restarting(tmp_path, monkeypatch):
    checker = object.__new__(DispatchCheck)
    checker.plan = dict(original="source_plan")
    checker.registered = (tmp_path, dict(case_code="CASE:A", image=dict(checksum=digest("image"))))
    checker.upstream = dict(skills=dict(plans=[]))
    checker.cache = SimpleNamespace(events=[dict(compute_seconds=0.1)])
    calls = []
    monkeypatch.setattr(checker, "check", lambda: calls.append(1))
    records = [dict(status="owner_returned", input_checksum=digest("image"))]
    first = checker.finish(records)
    checker.cache.events = [dict(compute_seconds=0.2)]
    assert checker.finish(copy.deepcopy(records)) == first
    assert len(calls) == 2 and json.loads((tmp_path / "upstream-use.json").read_bytes()) == first
    with pytest.raises(ValueError, match="recovery_receipt_conflict"):
        checker.finish([])
