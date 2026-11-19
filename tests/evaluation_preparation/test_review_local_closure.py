"""Narrow local closure contracts: no models, credentials or real database."""

from types import SimpleNamespace

import pytest

from agents.expert_agent.committed_review import confirmed_transport, read_committed_review
from apps.vectorengine_live_assembly import export_stage_report
from packages.grounding.models import digest
from packages.multimodal_research import stages
from packages.multimodal_research.stage_models import LocalReviewClosure


@pytest.mark.parametrize(
    "simulated,dispatched,confirmation,accepted",
    [
        (True, 0, None, True),
        (False, 1, True, True),
        (False, 0, False, True),
        (False, None, False, False),
        (False, 1, None, False),
        (False, None, None, False),
    ],
)
def test_local_completion_is_not_remote_confirmation(simulated, dispatched, confirmation, accepted):
    record = SimpleNamespace(
        request_id="original-request",
        simulated=simulated,
        api_requests=dispatched,
        usage=SimpleNamespace(status="unknown"),
    )
    events = [
        dict(phase="response_confirmation", request_id=record.request_id, confirmed=confirmation)
    ]
    readonly = SimpleNamespace(
        records=lambda: (record,), journal=SimpleNamespace(read=lambda: events), used=1
    )
    if accepted:
        assert confirmed_transport(readonly)["remote_response_unknown"] == 0
    else:
        with pytest.raises(ValueError, match="committed_review_unknown_request"):
            confirmed_transport(readonly)


def test_missing_intent_never_scans_for_accepted_objects():
    names = []

    def read(key):
        names.append(key)
        return None

    assert read_committed_review(SimpleNamespace(read=read), parent=None, config=None) is None
    assert names == ["live-stage-intent:review"]


@pytest.mark.parametrize("failed", [True, False])
def test_closure_report_names_are_bounded_and_restored(tmp_path, failed):
    campaign = SimpleNamespace(output=tmp_path, report_key="original")
    review = dict(schema_version="dynamic-phase-local-closure/1.0.0", closure={"proof": True})
    try:
        with stages.recovered_report(campaign, review):
            assert campaign.result_key == "multimodal-stage-recovered-case-result"
            assert campaign.result_filename != "result.json"
            assert campaign.report_key != "original"
            if failed:
                raise OSError("local")
    except OSError:
        assert failed
    assert campaign.report_key == "original" and not hasattr(campaign, "result_key")
    assert not hasattr(campaign, "local_review_closure")


@pytest.mark.parametrize("field", ["checksum", "run", "target", "requests", "none"])
def test_closure_binds_original_phase_and_zero_requests(field):
    original = dict(run_id="RUN:ONE", target=None, accepted=False)
    proof = dict(
        run_id="RUN:ONE",
        event=dict(role="event", code="EVENT:ONE", version="1.0.0", checksum=digest("event")),
        acceptance_key="event:domain",
        **{
            k: digest(k)
            for k in (
                "parent_checksum",
                "accepted_checksum",
                "report_checksum",
                "intent_checksum",
                "journal_checksum",
            )
        },
    )
    raw = dict(
        original_phase=original, original_phase_checksum=digest(original), confirmation=proof
    )
    if field == "checksum":
        raw["original_phase_checksum"] = digest("different")
    if field == "run":
        proof["run_id"] = "RUN:OTHER"
    if field == "target":
        original["target"] = "vision"
        raw["original_phase_checksum"] = digest(original)
    if field == "requests":
        raw["new_requests"] = 1
    if field == "none":
        assert LocalReviewClosure.model_validate(raw).new_requests == 0
    else:
        with pytest.raises(ValueError):
            LocalReviewClosure.model_validate(raw)


@pytest.mark.parametrize("where", ["json", "html"])
def test_inner_report_io_failure_does_not_write_acceptance(tmp_path, monkeypatch, where):
    from agents.expert_agent import reporting
    from apps import vectorengine_live_assembly as assembly
    from packages.multimodal_research import report_io

    # Only local renderer I/O is under test; durable integration lives in PG.
    state = object()
    monkeypatch.setattr(assembly, "validate", lambda cls, raw: state)
    monkeypatch.setattr(reporting, "read_research_report", lambda *a, **kw: {"saved": True})
    monkeypatch.setattr(reporting, "render_research_html", lambda *a: "saved HTML")
    if where == "json":

        def fail(*a):
            raise OSError("json export")

        monkeypatch.setattr(report_io, "publish_json", fail)
    else:

        def fail(*a):
            raise OSError("html export")

        monkeypatch.setattr(reporting, "render_research_html", fail)
    docs = SimpleNamespace(read=lambda key: ({"state": {}},))
    with pytest.raises(OSError, match=where):
        export_stage_report(
            review=SimpleNamespace(acceptance=object()),
            documents=docs,
            output=tmp_path,
            role="review",
            event=SimpleNamespace(code="event"),
            result={"acceptance_key": "original"},
        )
