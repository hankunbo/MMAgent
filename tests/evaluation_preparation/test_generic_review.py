"""Successor reviews retain original evidence; never restart stopped execution."""

import json
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation.generic_review import GenericReview


def fixture(tmp_path, *, complete=False, pending_ids=(), phase="stopped", pending=None):
    original = tmp_path / "original"
    original.mkdir()
    (original / "review").mkdir()
    result = dict(complete=complete, phases=[dict(run_id="RUN:OLD")])
    (original / "result.json").write_text(json.dumps(result))
    (original / "review/request.json").write_text(json.dumps(dict(lock=dict(run_id="RUN:OLD"))))
    report = dict(result=result, scores=dict(review_completion=dict(pending_ids=list(pending_ids))))
    c = object.__new__(GenericReview)
    c.acquisition = SimpleNamespace(
        output=original,
        usage=lambda: 8,
        paths=lambda: [original / "review"],
        phase_path=lambda t: original / t,
        accepted_observations=lambda: {"known": "original"},
    )
    c.sessions = SimpleNamespace(read=lambda _: SimpleNamespace(phase=phase, pending=pending))
    values = {"dynamic-case-result": (result,), "dynamic-case-report": (report,)}
    c.documents = lambda _: SimpleNamespace(read=values.get)
    c.output = tmp_path / "successor"
    c.review_key = "successor"
    c.spec = dict(key="original", sample="any.npz")
    return c, values


@pytest.mark.parametrize("complete", [False, True])
def test_failed_or_accepted_partial_can_be_reviewed_without_reacquisition(tmp_path, complete):
    c, _ = fixture(tmp_path, complete=complete, pending_ids=("pending",))
    assert c.prior()["reservations"] == 8
    assert c.before_review() is None
    assert c.paths() == [c.output / "review"]
    assert c.phase_path("known") == c.acquisition.output / "known"
    assert c.accepted_observations() == {"known": "original"}
    assert len(c.request_key(None)) <= 80
    with pytest.raises(ValueError, match="no_acquisition"):
        c.request_key("known")


@pytest.mark.parametrize("phase,pending", [("select", None), ("stopped", object())])
def test_unclosed_control_does_not_authorize_new_review(tmp_path, phase, pending):
    c, _ = fixture(tmp_path, phase=phase, pending=pending)
    with pytest.raises(ValueError, match="predecessor_not_closed"):
        c.prior()


def test_full_predecessor_does_not_create_unnecessary_review(tmp_path):
    c, _ = fixture(tmp_path, complete=True)
    with pytest.raises(ValueError, match="already_complete"):
        c.prior()


def test_predecessor_result_must_match_original_owner(tmp_path):
    c, values = fixture(tmp_path)
    values["dynamic-case-result"] = ({},)
    with pytest.raises(ValueError, match="bound_predecessor"):
        c.prior()


def test_review_does_not_use_long_unbounded_caller_ids(tmp_path):
    c, _ = fixture(tmp_path)
    c.review_key = "a" * 64
    c.spec["sample"] = "s" * 100 + ".npz"
    assert len(c.request_key(None)) <= 80
    assert len(c.archive_key("association:VASCULAR_SKELETON_DENSITY")) <= 80
    assert c.archive_key("glaucoma") != c.archive_key("other")
