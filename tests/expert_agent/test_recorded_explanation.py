"""Original failure -> explicit local admission; no re-dispatch or history rewrite."""

import pytest

from agents.expert_agent.reporting import read_research_report, render_research_html
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.acceptance_models import read_records
from packages.evidence_details.recorded_explanation import (
    ACCEPTANCE_KEY,
    GRANT_KEY,
    RecordedReviewPermission,
)
from packages.grounding.models import digest
from tests.expert_agent.explanation_support import failed_explanation


def test_original_schema_failure_local_score_and_idempotent_readback(monkeypatch, tmp_path):
    f = failed_explanation(monkeypatch, tmp_path)
    before, events = f.planner.read(f.binding.run_id), f.ports.budget.journal.read()
    assert len(f.wire.calls) == 1 and f.requests.records()[0].error == "schema"
    p, sub = RecordedReviewPermission.prepare(
        owner=f.owner, documents=f.documents, requests=f.requests, config=f.config
    )
    service = ResearchAcceptance(
        owner=f.owner, documents=f.documents, requests=f.requests, live_permission=p
    )
    assert service.submit(sub, key=ACCEPTANCE_KEY)["status"] == "submitted_not_accepted"
    assert service.accept(key=ACCEPTANCE_KEY).status == "accepted"
    saved = service.read(key=ACCEPTANCE_KEY)
    assert read_records(saved[0]).local_interpretation == sub.local_interpretation
    report = read_research_report(service, key=ACCEPTANCE_KEY, parent=f.state)
    assert digest(report) in render_research_html(report)
    n = len(f.documents.audit)
    assert service.accept(key=ACCEPTANCE_KEY).status == "accepted"
    assert service.read(key=ACCEPTANCE_KEY) == saved and len(f.documents.audit) == n
    assert f.planner.read(f.binding.run_id) == before and f.ports.budget.journal.read() == events
    assert len(f.wire.calls) == 1 and f.documents.read("multimodal-case-result") == (f.failed,)


@pytest.mark.parametrize("fault", ["claim", "raw", "parent", "cancel", "http"])
def test_changed_origin_cannot_be_locally_admitted(monkeypatch, tmp_path, fault):
    f = failed_explanation(monkeypatch, tmp_path)
    if fault == "claim":
        key = next(k for k in f.documents.saved if k.endswith(":claim"))
        f.documents.saved[key][0]["input_checksum"] = digest("changed")
    elif fault == "raw":
        key = next(k for k in f.documents.saved if ":attempt:" in k)
        f.documents.saved[key][0]["raw_model_text"] += " "
    elif fault == "parent":
        current = f.owner.current()
        monkeypatch.setattr(
            f.owner,
            "current",
            lambda: current.model_copy(update={"config_checksum": digest("changed")}),
        )
    elif fault == "cancel":
        current = f.documents.sessions.read(f.binding.run_id)
        monkeypatch.setattr(
            f.documents.sessions, "read", lambda _: current.model_copy(update={"cancelled": True})
        )
    else:
        monkeypatch.setattr("packages.llm_provider.http_recovery.observation_for", lambda *_: None)
    with pytest.raises(ValueError, match="explanation_"):
        RecordedReviewPermission.prepare(
            owner=f.owner, documents=f.documents, requests=f.requests, config=f.config
        )
    assert f.documents.read(GRANT_KEY) is None and len(f.wire.calls) == 1
