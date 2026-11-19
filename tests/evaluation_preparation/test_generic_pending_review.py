"""Pending-only CLI projection over original partial and subsequent acceptance."""

import json
import tempfile
from pathlib import Path

import pytest

from packages.evaluation_preparation import generic_pending_review as p
from packages.evaluation_preparation.generic_review import review_configuration
from packages.grounding.models import digest
from packages.llm_provider.config import read_provider_config
from tests.agent_provider.batch_support import generate
from tests.agent_provider.test_pending_truncated_review import inspect, partial
from tests.evaluation_preparation.test_dynamic_evidence import dynamic_fixture


def config():
    return read_provider_config(
        json.loads(Path("configs/agents/fairvlmed_relay_review_shape_v1.json").read_bytes())
    )


def test_profile_only_changes_batch_size_and_preserves_prompt_limits_route():
    before = config()
    after = review_configuration(before, p.PROFILE)
    raw = after.model_dump(mode="json")
    raw["review_batching"]["batch_size"] = 3
    assert raw == before.model_dump(mode="json")
    assert after.review_batching.batch_size == 1


@pytest.mark.parametrize("field", ["model", "batch"])
def test_incompatible_config_rejected(field):
    raw = config().model_dump(mode="json")
    if field == "model":
        raw["transport"]["model"] = "changed"
    else:
        raw["review_batching"]["batch_size"] = 2
    with pytest.raises(ValueError, match="generic_pending_profile_"):
        p.configuration(read_provider_config(raw))


def test_original_partial_only_selects_explicit_confirmed_truncation_and_unsent(monkeypatch):
    f, sub = partial(monkeypatch)
    before = digest(f.documents.saved), digest(f.budget.journal.read())
    inventory = inspect(f)
    ack = inventory["quarantined_batches"][0]["known_truncated_failure"]["last_request_id"]
    unsent = p.select(inventory, None)
    selected = p.select(inventory, ack)
    assert set(unsent["selected_ids"]) < set(selected["selected_ids"])
    assert set(selected["selected_ids"]) | set(inventory["accepted_ids"]) == {
        i.id for i in f.owner.current().policy.items
    }
    assert not set(selected["selected_ids"]) & {i.assessment_id for i in sub.candidate.items}
    with pytest.raises(ValueError, match="ack_mismatch"):
        p.select(inventory, "unknown-or-another-request")
    assert before == (digest(f.documents.saved), digest(f.budget.journal.read()))


def test_single_item_batches_use_real_provider_acceptance_and_preserve_full_context():
    f = dynamic_fixture(17, config_path=p.PROVIDER)
    initial = f.owner.current()
    selected = [i.id for i in initial.policy.items][-4:]
    current = p.selected_state(initial, selected)
    assert len(current.policy.requirements) == 17
    assert current.material.observations == initial.material.observations
    assert current.material.sources == initial.material.sources
    assert current.policy.groups == initial.policy.groups
    f.owner.state = current
    sub = generate(f)
    assert len(sub.plan.batches) == 4
    assert all(len(b.assessment_ids) == 1 for b in sub.plan.batches)
    assert f.acceptance.submit(sub, key="pending")["status"] == "submitted_not_accepted"
    assert f.acceptance.accept(key="pending").status == "accepted"
    scores = f.acceptance.read(key="pending")[2]["scores"]
    assert {r["id"] for r in scores["reviews"]} == set(selected)
    proof = dict(selected_ids=selected, inventory=dict(accepted_ids=["old"]))
    status = p.completion(proof, scores)
    assert status["pending_ids"] == []
    assert status["whole_case_aggregation_required"]
    assert not status["historical_judgments_included_in_this_score"]
    assert len(f.budget.records()) == 4


@pytest.mark.parametrize("ids", [[], ["not-current"]])
def test_changed_selection_rejects(ids):
    f = dynamic_fixture()
    with pytest.raises(ValueError, match="current_items_changed"):
        p.selected_state(f.owner.current(), ids)


def test_increment_report_reads_original_acceptance_and_is_not_whole_case():
    from types import SimpleNamespace

    f = dynamic_fixture(17, config_path=p.PROVIDER)
    initial = f.owner.current()
    selected = [i.id for i in initial.policy.items][-4:]
    current = p.selected_state(initial, selected)
    f.owner.state = current
    sub = generate(f)
    f.acceptance.submit(sub, key="pending")
    assert f.acceptance.accept(key="pending").status == "accepted"
    f.documents.write("dynamic-phase-input", (current,))
    proof = dict(selected_ids=selected, inventory=dict(accepted_ids=["old"]))
    campaign = SimpleNamespace(documents=lambda _: f.documents, pending_review_proof=proof)
    result = dict(phases=[dict(run_id=f.binding.run_id, accepted=True, acceptance_key="pending")])
    before = digest(f.budget.journal.read()), digest(f.acceptance.read(key="pending"))
    report = p.export_increment(campaign, result)
    saved = f.documents.read("dynamic-case-report")
    assert report["scores"] == f.acceptance.read(key="pending")[2]["scores"]
    assert not report["completion"]["all_obtained_reviewed"]
    assert report["completion"]["whole_case_aggregation_required"]
    assert "不是完整病例评分" in saved[1]["html"]
    audit = len(f.documents.audit)
    assert p.export_increment(campaign, result) == report
    assert len(f.documents.audit) == audit
    assert before == (digest(f.budget.journal.read()), digest(f.acceptance.read(key="pending")))


@pytest.fixture
def local_lock_directory():
    # This boundary exercises the host's flock. The repository temporary tree
    # can reside on NFS without locking support, unlike the local runtime lease.
    with tempfile.TemporaryDirectory(prefix="generic-preflight-lock-", dir="/tmp") as root:
        yield Path(root)


def test_rejected_preflight_does_not_claim_successor_or_start_budget(
    local_lock_directory, monkeypatch
):
    tmp_path = local_lock_directory
    from packages.evaluation_preparation import generic_review as r

    (tmp_path / "parent").mkdir()
    monkeypatch.setattr(r, "directory", lambda key: tmp_path / key)
    monkeypatch.setattr(r, "load", lambda _: {"schema_version": "fairvlmed-run-input/1.6.0"})
    monkeypatch.setattr(r, "provider_config", lambda _: config())
    monkeypatch.setattr(r, "register", lambda _: None)

    def reject(*args, **kwargs):
        raise ValueError("generic_pending_truncation_ack_mismatch")

    monkeypatch.setattr(r, "GenericReview", reject)
    with pytest.raises(ValueError, match="ack_mismatch"):
        r.run_review(
            "parent", "child", review_profile=p.PROFILE, acknowledge_truncated_request="wrong"
        )
    assert not (tmp_path / "parent/review-successor.json").exists()
    assert not (tmp_path / "child/started.json").exists()


def test_cli_uses_explicit_pending_profile_and_exact_ack(monkeypatch):
    from packages.evaluation_preparation import generic_review as r

    calls = []
    monkeypatch.setattr(r, "run_review", lambda **kwargs: calls.append(kwargs) or {})
    assert (
        r.main(
            [
                "--key",
                "parent",
                "--review-key",
                "child",
                "--review-profile",
                p.PROFILE,
                "--acknowledge-truncated-request",
                "request:confirmed",
            ]
        )
        == 0
    )
    assert calls[0]["acknowledge_truncated_request"] == "request:confirmed"
    assert calls[0]["review_profile"] == p.PROFILE
