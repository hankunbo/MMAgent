"""Current accepted origins, not caller selection, exclude inherited work."""

from copy import deepcopy
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from packages.multimodal_research import pending_review as p
from tests.agent_provider.test_accepted_review_reuse import accepted_fixture, state_from
from tests.evaluation_preparation.test_pending_reassessment_window import fixture, install_proof


def partition():
    f = accepted_fixture(new_observation=True)
    state = f.owner.current()
    origin = f.owner.confirmed_review_reader.current(state)
    parent = SimpleNamespace(
        base=SimpleNamespace(confirmed_review_reader=f.owner.confirmed_review_reader),
        protocol=dict(mode="acquisition", reuse_assessment_ids=list(origin.assessment_ids)),
        config=f.old.config,
    )
    attempt = dict(
        selected_ids=sorted([*origin.assessment_ids, "item:additional"]),
        acknowledge_request="request:old",
        quarantined_batches=[
            dict(
                assessment_ids=["item:additional"],
                request_ids=["request:old"],
                new_reassessment_authorized=True,
                old_response_state_unchanged=True,
            )
        ],
    )
    return f, state, parent, attempt


def test_retained_origin_excludes_old_items_and_keeps_exact_new_batch():
    f, state, parent, attempt = partition()
    before = deepcopy(attempt)
    value = p.retained_partition(parent, state, attempt, [dict(accepted=False)])
    assert value["selected_ids"] == ["item:additional"]
    assert value["accepted_ids"] == sorted(parent.protocol["reuse_assessment_ids"])
    assert value["acknowledge_unknown_request"] == "request:old"
    assert attempt == before
    assert value == p.retained_partition(parent, state, attempt, [dict(accepted=False)])
    assert not f.budget.records()


@pytest.mark.parametrize(
    "change",
    ["reader", "source", "receipt", "retained", "overlap", "extra", "ack", "phase", "mode"],
)
def test_bad_partition_cannot_redispatch_retained_work(change):
    f, state, parent, attempt = partition()
    phase = [dict(accepted=False)]
    if change == "reader":
        parent.base.confirmed_review_reader = None
    elif change == "source":
        raw = state.model_dump(mode="json")
        raw["material"]["sources"][0]["text"] += " changed"
        state = state_from(raw)
    elif change == "receipt":
        f.old.documents.saved["accepted:old-accepted"][3]["records_checksum"] = "changed"
    elif change == "retained":
        parent.protocol["reuse_assessment_ids"].append("invented")
    elif change == "overlap":
        attempt["quarantined_batches"][0]["assessment_ids"].append(
            parent.protocol["reuse_assessment_ids"][0]
        )
    elif change == "extra":
        attempt["selected_ids"].append("unrequested")
    elif change == "ack":
        attempt["quarantined_batches"][0]["new_reassessment_authorized"] = False
    elif change == "phase":
        phase = [dict(accepted=True)]
    elif change == "mode":
        parent.protocol["mode"] = "review"
    with pytest.raises(ValueError):
        p.retained_partition(parent, state, attempt, phase)
    assert not f.budget.records()


@pytest.mark.parametrize("expired", [False, True])
def test_partitioned_window_preserves_live_deadline_or_declares_new_window(monkeypatch, expired):
    install_proof(monkeypatch)
    parent, plan = fixture()
    parent.protocol.update(mode="acquisition", reuse_assessment_ids=["old:a"])
    start = datetime.fromisoformat(plan["started_at"])
    parent.protocol["deadline"] = (start + timedelta(seconds=-1 if expired else 900)).isoformat()
    plan["pending_review"].update(
        accepted_ids=["old:a"],
        initial_failed_partition=dict(retained_ids=["old:a"], origin_checksum="verified-owner"),
    )
    p.declare_reassessment_window(plan, parent)
    assert plan["schema_version"] == p.PARTITIONED_WINDOW_VERSION
    if not expired:
        assert plan["deadline"] == parent.protocol["deadline"] and plan["seconds"] == 900
    p.validate_reassessment_window(plan, parent)
    broken = deepcopy(plan)
    broken["pending_review"]["initial_failed_partition"]["retained_ids"] = []
    with pytest.raises(ValueError, match="partitioned_window"):
        p.validate_reassessment_window(broken, parent)


def test_old_155_does_not_silently_consume_new_partition_proof(monkeypatch):
    install_proof(monkeypatch)
    parent, plan = fixture()
    plan["pending_review"]["initial_failed_partition"] = dict(origin_checksum="anything")
    with pytest.raises(ValueError, match="reassessment_window"):
        p.validate_reassessment_window(plan, parent)
