"""Known format rejection is accounted, not an UNKNOWN or a reused clinical answer."""

import pytest

from agents.provider_agents.batched_review import run_batches
from packages.grounding.models import digest
from tests.agent_provider.test_review_continuation import continuation_fixture


def test_good_batches_reused_failed_answer_not_reused_scores_saved():
    from agents.expert_agent.reporting import read_research_report, render_research_html
    from packages.evidence_details.acceptance_models import read_records

    f = continuation_fixture(format_failure=True)
    from types import SimpleNamespace

    from packages.multimodal_research.campaign_renewal import historical_reader

    # Exercise the production read-only request adapter, not just RequestBudget.
    f.old.documents.write("dynamic-phase-input", (f.old.owner.current(),))
    f.owner.confirmed_review_reader = historical_reader(
        SimpleNamespace(documents=lambda _: f.old.documents),
        f.old.owner.current().material.scope.run_id,
        format_config=f.agent.provider.config,
    )
    # Fixture request journal is in memory; install it on the production adapter.
    f.owner.confirmed_review_reader.requests.journal = f.old.budget.journal
    before = digest(f.old.documents.saved), digest(f.old.budget.journal.read())
    origin = f.owner.confirmed_review_reader.current(f.owner.current())
    f.owner.confirmed_review_reader.expected_format_origin = digest(origin)
    sub = run_batches(
        f.agent, state=f.owner.current(), documents=f.documents, key="new:A", round_id="A"
    )
    assert sub.schema_version == "research-batched-submission/8.0.0"
    assert len(sub.historical.submission.candidate.items) == 4
    assert len(sub.historical.confirmed_format_failures) == 1
    assert len(sub.candidate.items) == 5 and len(f.budget.records()) == 1
    f.acceptance.submit(sub, key="closure")
    receipt = f.acceptance.accept(key="closure")
    assert receipt.status == "accepted"
    saved = f.acceptance.read(key="closure")
    records = read_records(saved[0])
    assert records.schema_version == "case-observation-interpretation/2.10.0"
    failed = sub.historical.confirmed_format_failures[0].request_id
    assert failed not in records.request_ids
    report = read_research_report(f.acceptance, key="closure", parent=f.owner.current())
    assert len(report["scores"]["reviews"]) == 5
    assert "confirmed_schema_failure_not_reused" in render_research_html(report)
    audit = len(f.documents.audit)
    assert f.acceptance.accept(key="closure") == receipt and len(f.documents.audit) == audit
    assert before == (digest(f.old.documents.saved), digest(f.old.budget.journal.read()))


@pytest.mark.parametrize("mutation", ["unknown", "raw", "model", "frozen"])
def test_format_reuse_rejects_drift_before_new_request(mutation):
    f = continuation_fixture(format_failure=True)
    reader = f.owner.confirmed_review_reader
    if mutation == "unknown":
        f.old.budget.journal.append({"phase": "reserved", "request_id": "UNKNOWN"})
    elif mutation == "raw":
        key = next(
            k
            for k in f.old.documents.saved
            if ":attempt:" in k and f.old.documents.saved[k][0].get("error") == "schema"
        )
        f.old.documents.saved[key][0]["raw_model_text"] += " "
    elif mutation == "model":
        reader.format_config = reader.format_config.model_copy(
            update={
                "transport": reader.format_config.transport.model_copy(
                    update={"model": "other-model"}
                )
            }
        )
    else:
        reader.expected_format_origin = digest("different frozen origin")
    with pytest.raises(ValueError):
        run_batches(
            f.agent, state=f.owner.current(), documents=f.documents, key="new:A", round_id="A"
        )
    assert not f.budget.records()
