"""Partitioned readback is zero-dispatch and uses the original ancestor owner."""

import json

import pytest

from agents.expert_agent.committed_review import confirmed_batches
from agents.expert_agent.reporting import read_research_report, render_research_html
from packages.evidence_details.review_batches import read_submission
from packages.grounding.models import digest
from tests.evaluation_preparation.test_partition_increment import partition_fixture


def test_partition_readback_retains_original_requests_and_six_sections():
    f = partition_fixture()
    before = digest(f.documents.saved), digest(f.budget.journal.read())
    for _ in range(2):
        records = confirmed_batches(
            f.owner.current(),
            f.submission,
            f.documents,
            f.budget,
            confirmed_review_reader=f.owner.confirmed_review_reader,
        )
        assert records == tuple(f.budget.records())
        report = read_research_report(f.acceptance, key="new", parent=f.owner.current())
        page = render_research_html(report)
        assert "1. 来源事实" in page and "6. 计算依赖与判读背景" in page
    assert before == (digest(f.documents.saved), digest(f.budget.journal.read()))
    old = read_submission(f.old.documents.read("submission:original")[0]["submission"])
    assert confirmed_batches(f.old.owner.current(), old, f.old.documents, f.old.budget)


def test_partition_readback_requires_owner_and_unchanged_binding():
    f = partition_fixture()
    before = digest(f.documents.saved), digest(f.budget.journal.read())
    with pytest.raises(ValueError, match="committed_review_ancestor_owner_required"):
        confirmed_batches(f.owner.current(), f.submission, f.documents, f.budget)
    bad = f.submission.model_copy(
        update={
            "historical": f.submission.historical.model_copy(
                update={"acceptance_checksum": digest("different original commit")}
            )
        }
    )
    with pytest.raises(ValueError, match="review_continuation_origin_changed"):
        confirmed_batches(
            f.owner.current(),
            bad,
            f.documents,
            f.budget,
            confirmed_review_reader=f.owner.confirmed_review_reader,
        )
    assert before == (digest(f.documents.saved), digest(f.budget.journal.read()))


@pytest.mark.parametrize("supported", [True, False])
@pytest.mark.parametrize("version", ["1.12.0", "1.26.0", "1.47.0", "1.48.0", "1.60.0"])
def test_cli_finalizer_routes_partition_successor_to_original_owner(
    tmp_path, monkeypatch, supported, version
):
    from packages.multimodal_research import campaign_renewal, finalization

    previous, output = tmp_path / "original", tmp_path / "closure"
    previous.mkdir()
    old = {"complete": False, "new_requests": 6, "phases": [{"error": "deadline"}]}
    plan = dict(
        schema_version="multimodal-campaign-renewal/" + version,
        mode="acquisition",
        review_partition="owner_bound_reused_and_new/1.0.0",
    )
    if not supported:
        plan["mode"] = "definitions"
    (previous / "result.json").write_text(json.dumps(old))
    (previous / "protocol.json").write_text(json.dumps(plan))
    before = (previous / "result.json").read_bytes()
    original_owner = object()
    seen = []
    monkeypatch.setattr(campaign_renewal, "completed_parent", lambda path: (original_owner, old))

    def finish(campaign, result, out, module, **kwargs):
        assert campaign is original_owner and result == old and out == output
        assert module.__name__.endswith("recorded_finalization")
        assert kwargs == {"complete": True, "active": True}
        seen.append(True)
        return {"requests": 0}

    monkeypatch.setattr(finalization, "_finish_entry", finish)
    if supported:
        assert finalization.finalize_entry(previous, output, complete=True) == {"requests": 0}
        assert seen == [True]
    else:
        with pytest.raises(ValueError, match="recorded_finalization_renewal_version"):
            finalization.finalize_entry(previous, output, complete=True)
        assert not seen
    assert (previous / "result.json").read_bytes() == before and not output.exists()
