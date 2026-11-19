"""Confirmed format correction is accounted history, never a reusable failed answer."""

from copy import deepcopy

import pytest

from packages.evidence_details.review_continuation import ConfirmedReviewReader
from packages.grounding.models import digest, validate
from tests.agent_provider.finalization_support import fixture


def corrected_origin(pg=None):
    f = fixture(pg, partial=True, corrected=True, reasoned=True, corrected_partial=True)
    raw = f.state.model_dump(mode="json")
    raw["material"]["scope"]["run_id"] = "RUN:CORRECTED_NEXT"
    raw["material"]["scope"].pop("material_checksum")
    raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
    next_state = validate(type(f.state), raw)
    reader = ConfirmedReviewReader(
        documents=f.docs, requests=f.requests, state_reader=f.owner.current
    )
    return f, reader, next_state


def test_corrected_confirmed_batch_is_reused_without_failed_answer_or_request():
    f, reader, state = corrected_origin()
    before = deepcopy(f.docs.saved), deepcopy(f.budget.journal.read())
    origin = reader.current(state)
    assert len(origin.submission.batches) == 1
    assert origin.submission.pending_batch_ids
    assert origin.submission.candidate == f.submission.candidate
    assert len(f.wire.calls) == 2
    assert reader.current(state) == origin
    assert before == (f.docs.saved, f.budget.journal.read())


@pytest.mark.parametrize(
    "fault", ["unknown", "feedback", "raw", "missing_correction", "pending_claim", "source"]
)
def test_corrected_history_still_rejects_ambiguous_changed_or_withdrawn(fault):
    f, reader, state = corrected_origin()
    if fault == "unknown":
        f.budget.journal.events.append(dict(phase="reserved", request_id="unknown"))
    elif fault in {"feedback", "missing_correction"}:
        index = next(
            i
            for i, e in enumerate(f.budget.journal.events)
            if e.get("schema_version") == "review-schema-correction/1.0.0"
        )
        if fault == "feedback":
            f.budget.journal.events[index]["failed_record_checksum"] = digest("wrong")
        else:
            f.budget.journal.events.pop(index)
    elif fault == "raw":
        key = next(
            k for k, v in f.docs.saved.items() if ":attempt:" in k and v[0].get("error") == "schema"
        )
        f.docs.saved[key][0]["raw_model_text"] += " "
    elif fault == "pending_claim":
        f.docs.saved["review-batches:synthetic-event:A:batch:001:claim"] = ({"unknown": True},)
    else:
        f.owner.withdrawn = True
    with pytest.raises(ValueError):
        reader.current(state)
    assert len(f.wire.calls) == 2
