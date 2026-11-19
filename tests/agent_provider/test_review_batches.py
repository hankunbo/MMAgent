"""A-only offline contracts through Provider -> journal -> original domain owner."""

import copy

import pytest

from agents.expert_agent.reporting import read_research_report, render_research_html
from agents.provider_agents.batched_review import read_progress
from packages.evidence_details.acceptance_models import read_records
from packages.evidence_details.review_batches import (
    ReviewBatchPlan,
    aggregate,
    batch_payload,
    make_plan,
    validate_batch_receipts,
)
from packages.grounding.models import digest, validate
from packages.llm_provider.provider import ProviderError
from packages.llm_provider.transport import Reply
from tests.agent_provider.batch_support import batch_reply, fixture, generate
from tests.agent_provider.support import Documents


def test_plan_exact_45_full_parent_stable_keys_and_finite_cap():
    f = fixture(45)
    state = f.owner.current()
    plan = make_plan(state, f.config.review_batching, key="same-key")
    assert len(plan.batches) == 15
    assert len({i for b in plan.batches for i in b.assessment_ids}) == 45
    assert make_plan(state, f.config.review_batching, key="same-key") == plan
    for b in plan.batches:
        p = batch_payload(state, plan, b)
        assert p["material"] == state.material.model_dump(mode="json")
        assert tuple(p["assessment_ids"]) == b.assessment_ids
        assert "fixed_model_row" not in str(p)
    with pytest.raises(ValueError, match="batch_partition"):
        make_plan(state, f.config.review_batching.model_copy(update={"max_batches": 14}), key="x")
    raw = plan.model_dump(mode="json")
    raw["batches"][-1] = raw["batches"][0]
    with pytest.raises(ValueError):
        validate(ReviewBatchPlan, raw)


def test_original_provider_multireceipts_one_acceptance_score_report_readback():
    f = fixture()
    parent = f.owner.current()
    sub = generate(f)
    assert len(sub.batches) == f.budget.used == 3
    validate_batch_receipts(parent, sub, f.documents, f.budget)
    assert not f.documents.consumed(parent.material.scope.graph)
    assert f.acceptance.submit(sub, key="accept")["status"] == "submitted_not_accepted"
    receipt = f.acceptance.accept(key="accept")
    assert receipt.status == "accepted"
    records = read_records(f.acceptance.read(key="accept")[0])
    assert records.schema_version == "case-observation-interpretation/2.3.0"
    assert records.request_id is None and len(records.request_ids) == 3
    assert len(records.interpretations) == 7
    assert len({i.request_id for i in records.interpretations}) == 3
    report = read_research_report(f.acceptance, key="accept", parent=parent)
    assert report["scores"]["schema_version"] == "case-research-scoring/6.1.0"
    assert report["scores"]["reviews"][1]["judgment"]["value"] is None
    html = render_research_html(report)
    assert "非临床结论" in html
    audit = list(f.documents.audit)
    assert f.acceptance.accept(key="accept") == receipt
    assert f.documents.audit == audit
    assert generate(f) == sub and f.budget.used == 3
    assert f.acceptance.submit(sub, key="consume-again").status == "rejected"


@pytest.mark.parametrize(
    "mutation", ["scope", "round", "batch", "missing", "duplicate", "source", "version"]
)
def test_bad_batch_stops_no_acceptance_no_business_retry(mutation):
    def bad(body):
        raw = batch_reply(body)
        if mutation == "scope":
            raw["scope_checksum"] = digest("foreign")
        elif mutation == "round":
            raw["round"] = "B"
        elif mutation == "batch":
            raw["batch_id"] = "batch:foreign"
        elif mutation == "missing":
            raw["items"].pop()
        elif mutation == "duplicate":
            raw["items"].append(copy.deepcopy(raw["items"][0]))
        elif mutation == "source":
            raw["items"][0]["reference_citations"] = ["SOURCE:FOREIGN"]
        else:
            raw["schema_version"] = "research-review/1.0.0"
        return raw

    f = fixture(replies=[bad])
    with pytest.raises(ProviderError, match="scope|schema"):
        generate(f)
    assert f.budget.used == 1
    assert not read_progress(f.documents, "synthetic-batches")["aggregate"]
    assert not f.documents.consumed(f.owner.current().material.scope.graph)
    with pytest.raises(ProviderError, match="unknown_no_repeat"):
        generate(f)
    assert f.budget.used == 1


@pytest.mark.parametrize("error", ["truncated", "timeout"])
def test_last_batch_failure_keeps_earlier_candidates_without_complete_score(error):
    f = fixture(replies=[batch_reply, batch_reply, Reply(content=b'{"partial":', error=error)])
    with pytest.raises(ProviderError, match=error):
        generate(f)
    p = read_progress(f.documents, "synthetic-batches")
    assert [bool(b["result"]) for b in p["batches"]] == [True, True, False]
    assert p["aggregate"] is None and f.budget.used == 3
    with pytest.raises(ProviderError, match="unknown_no_repeat"):
        generate(f)
    assert f.budget.used == 3
    assert not any(k.startswith("accepted:") for k in f.documents.saved)


@pytest.mark.parametrize("suffix", [":attempt:", ":response", ":result", ":aggregate"])
def test_saved_confirmation_lost_restores_without_resend(suffix):
    class Lost(Documents):
        once = True

        def write(self, key, values):
            super().write(key, values)
            if self.once and suffix in key:
                self.once = False
                raise OSError("saved_confirmation_lost")

    f = fixture(documents=Lost())
    with pytest.raises(OSError, match="confirmation_lost"):
        generate(f)
    sub = generate(f)
    assert len(sub.batches) == f.budget.used == 3
    validate_batch_receipts(f.owner.current(), sub, f.documents, f.budget)


def test_claim_without_response_is_unknown_and_same_key_changed_parent_conflicts():
    f = fixture()
    plan = make_plan(f.owner.current(), f.config.review_batching, key="synthetic-batches")
    slot = "review-batches:synthetic-batches:" + plan.batches[0].id
    f.documents.write(slot + ":claim", ({"unknown": True},))
    with pytest.raises(ProviderError, match="unknown_no_repeat"):
        generate(f)
    assert f.budget.used == 0
    f.owner.state = f.owner.state.model_copy(update={"config_checksum": digest("other")})
    with pytest.raises(ValueError, match="idempotency_conflict"):
        generate(f)


@pytest.mark.parametrize("terminal", ["stopped", "cancelled", "owner_unknown", "deadline"])
def test_terminal_before_claim_and_between_batches_preserves_history(terminal):
    f = fixture()

    def control():
        raise ProviderError(terminal)

    f.budget.check_control = control
    with pytest.raises(ProviderError, match=terminal):
        generate(f)
    assert not f.documents.saved and f.budget.used == 0
    f.budget.check_control = lambda: None
    original = f.documents.write

    def write(key, values):
        original(key, values)
        if key.endswith(":result"):
            f.budget.check_control = control

    f.documents.write = write
    with pytest.raises(ProviderError, match=terminal):
        generate(f)
    assert f.budget.used == 1
    assert read_progress(f.documents, "synthetic-batches")["batches"][0]["result"]


def test_parent_withdrawal_and_receipt_tamper_rejected_at_original_acceptance():
    f = fixture()
    sub = generate(f)
    bad = sub.model_copy(update={"batches": tuple(reversed(sub.batches))})
    assert f.acceptance.submit(bad, key="bad").status == "rejected"
    f.acceptance.submit(sub, key="withdrawn")
    f.owner.state = f.owner.state.model_copy(update={"prompt_checksum": digest("withdrawn")})
    assert f.acceptance.accept(key="withdrawn").status == "rejected"
    with pytest.raises(ValueError):
        aggregate(f.owner.state, sub.plan, sub.batches[:-1])


@pytest.mark.parametrize("value", [-0.4, 0.0])
def test_signed_e_is_not_null_or_clipped(value):
    def negative(body):
        raw = batch_reply(body)
        raw["items"][0]["judgment"].update(value=value, lower=-0.6, upper=0.1)
        return raw

    f = fixture(1, replies=[negative])
    sub = generate(f)
    f.acceptance.submit(sub, key="negative")
    assert f.acceptance.accept(key="negative").status == "accepted"
    row = f.acceptance.read(key="negative")[2]["scores"]["reviews"][0]
    assert row["judgment"]["value"] == value
    assert (row["refutation"] > 0) == (value < 0)


def test_fixed_owner_row_never_sent_and_original_score_includes_it():
    from packages.evidence_scoring.accepted import build_accepted
    from packages.evidence_scoring.continuous_models import ContinuousReview
    from tests.agent_provider.test_review_boundaries import rebind

    prototype = fixture(1)
    sub = generate(prototype)
    fixed = build_accepted(prototype.owner.current(), sub.candidate)["reviews"][0]
    fixed.update(id="item:fixed", kind="model", role="model", title="OLD_FIXED_ANSWER_NOT_FOR_WIRE")
    fixed["judgment"]["method"] = "fixed_model_margin"
    fixed = validate(ContinuousReview, fixed)
    f = fixture(1)
    policy = f.owner.state.policy
    item = policy.items[0].model_copy(
        update=dict(
            id="item:fixed",
            kind="model",
            role="model",
            fixed_observation_ref=f.owner.state.material.observations[0].reference,
            fixed_model_row=fixed,
        )
    )
    rebind(f.owner, policy=policy.model_copy(update={"items": (*policy.items, item)}))
    parent = f.owner.current()
    sub = generate(f)
    assert sub.plan.fixed_assessment_ids == ("item:fixed",)
    assert "OLD_FIXED_ANSWER_NOT_FOR_WIRE" not in str(f.provider.transport.calls)
    assert "fixed_model_row" not in str(f.provider.transport.calls)
    assert len(sub.candidate.items) == 1
    f.acceptance.submit(sub, key="fixed")
    assert f.acceptance.accept(key="fixed").status == "accepted"
    report = read_research_report(f.acceptance, key="fixed", parent=parent)
    assert next(
        r for r in report["scores"]["reviews"] if r["id"] == "item:fixed"
    ) == fixed.model_dump(mode="json")
    assert "固定模型映射" in render_research_html(report)


def test_html_escapes_response_and_does_not_replace_missing_values():
    def hostile(body):
        raw = batch_reply(body)
        raw["items"][0]["comparison"] = "<script>alert('untrusted')</script>"
        return raw

    f = fixture(3, replies=[hostile])
    sub = generate(f)
    f.acceptance.submit(sub, key="html")
    f.acceptance.accept(key="html")
    report = read_research_report(f.acceptance, key="html", parent=f.owner.current())
    html = render_research_html(report)
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert html.count("<section>") == 18 and "<details>" in html
    assert report["scores"]["reviews"][1]["judgment"]["value"] is None


def test_source_withdrawn_by_current_owner_rejects_saved_submission():
    f = fixture(3)
    sub = generate(f)
    f.acceptance.submit(sub, key="withdrawn-source")

    def withdrawn():
        raise ValueError("source_withdrawn")

    f.owner.current = withdrawn
    with pytest.raises(ValueError, match="source_withdrawn"):
        f.acceptance.accept(key="withdrawn-source")
    assert f.acceptance.read(key="withdrawn-source") is None
