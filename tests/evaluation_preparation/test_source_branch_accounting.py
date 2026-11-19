"""Budget-only Source branches cannot inject observations or reset case costs."""

from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import campaign_renewal
from tests.evaluation_preparation.test_aggregation_increment import accounting_fixture


def fixture(monkeypatch, tmp_path):
    a, plan, campaign, old, readers = accounting_fixture(tmp_path)
    base = SimpleNamespace(
        manifest=SimpleNamespace(checksum=digest("input")),
        task="proposition",
        binding={"case_code": "case"},
    )
    campaign.base = base
    branch = SimpleNamespace(
        base=base,
        protocol={"mode": "definitions", "definition_increment_only": True},
    )
    result = dict(complete=True, previous_attempts=[old["previous_attempts"][0]], new_requests=4)
    path = tmp_path / "source"
    entry = dict(directory=str(path), requests=4, checksum=digest(result))
    proof = dict(
        result=digest(result),
        definition={"run_id": "source-run"},
        previous_attempts=[old["previous_attempts"][0], entry],
    )
    calls = []

    def completed(value):
        assert value == path
        calls.append("restore")
        return branch, result

    def terminal(owner, value, **kwargs):
        assert owner is branch and value == path and kwargs == {"completed_source": True}
        calls.append("terminal")
        return proof

    monkeypatch.setattr(a, "completed_parent", completed)
    monkeypatch.setattr(campaign_renewal, "stopped_tip", terminal)
    plan = a.read_plan(
        dict(
            plan.model_dump(mode="json"),
            schema_version="accepted-review-aggregation-plan/1.7.0",
            source_budget_origins=[dict(campaign=str(path), result_checksum=digest(result))],
        )
    )
    return SimpleNamespace(**locals())


def test_real_budget_union_deduplicates_ancestry_and_keeps_review_origin_unchanged(
    monkeypatch, tmp_path
):
    f = fixture(monkeypatch, tmp_path)
    original = digest((f.old, f.readers[0].accounting))
    attempts = f.a.request_history(f.plan, f.campaign, f.old, f.readers)
    assert sum(x["requests"] for x in attempts) == 23
    assert len(attempts) == 3 and f.calls == ["restore", "terminal"]
    assert digest((f.old, f.readers[0].accounting)) == original
    result = dict(previous_attempts=attempts, cumulative_requests=23, new_requests=0)
    f.a.verify_request_history(
        f.plan,
        f.campaign,
        f.old,
        f.readers,
        dict(request_history_checksum=digest(attempts)),
        result,
    )
    result["cumulative_requests"] = 19
    with pytest.raises(ValueError, match="request_history_binding"):
        f.a.verify_request_history(
            f.plan,
            f.campaign,
            f.old,
            f.readers,
            dict(request_history_checksum=digest(attempts)),
            result,
        )


@pytest.mark.parametrize(
    "fault",
    [
        "checksum",
        "case",
        "input",
        "task",
        "not_source",
        "incomplete",
        "no_common",
        "count",
        "no_definition",
        "duplicate",
        "owner_reject",
        "over_budget",
    ],
)
def test_branch_proof_failures_refuse_budget_merge(monkeypatch, tmp_path, fault):
    f = fixture(monkeypatch, tmp_path)
    f.branch.base = SimpleNamespace(**vars(f.base))
    if fault == "checksum":
        f.result["changed"] = True
    elif fault == "case":
        f.branch.base.binding = {"case_code": "other"}
    elif fault == "input":
        f.branch.base.manifest = SimpleNamespace(checksum=digest("other"))
    elif fault == "task":
        f.branch.base.task = "other"
    elif fault == "not_source":
        f.branch.protocol["mode"] = "review"
    elif fault == "incomplete":
        f.result["complete"] = False
    elif fault == "no_common":
        f.proof["previous_attempts"][0] = dict(
            f.proof["previous_attempts"][0], directory=str(tmp_path / "other")
        )
    elif fault == "count":
        f.proof["previous_attempts"][-1]["requests"] += 1
    elif fault == "no_definition":
        f.proof["definition"] = None
    elif fault == "duplicate":
        f.plan = f.plan.model_copy(
            update={"source_budget_origins": f.plan.source_budget_origins * 2}
        )
    elif fault == "owner_reject":

        def reject(*args, **kwargs):
            raise ValueError("original_source_owner_rejected")

        monkeypatch.setattr(campaign_renewal, "stopped_tip", reject)
    else:
        # Valid Source-only total, but the other branch pushes the union over 40.
        f.result["new_requests"] = 24
        f.proof["previous_attempts"][-1].update(requests=24, checksum=digest(f.result))
        f.proof["result"] = digest(f.result)
        f.plan = f.plan.model_copy(
            update={
                "source_budget_origins": (
                    f.plan.source_budget_origins[0].model_copy(
                        update={"result_checksum": digest(f.result)}
                    ),
                )
            }
        )
    with pytest.raises(ValueError):
        f.a.request_history(f.plan, f.campaign, f.old, f.readers)


def test_branch_path_symlink_rejected_before_owner(monkeypatch, tmp_path):
    f = fixture(monkeypatch, tmp_path)
    f.path.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="symlink"):
        f.a.request_history(f.plan, f.campaign, f.old, f.readers)
    assert not f.calls


def test_typed_branch_cost_is_verified_by_its_original_owner(monkeypatch, tmp_path):
    f = fixture(monkeypatch, tmp_path)
    f.entry["kind"] = "interrupted_public_material_deferred"
    seen = []
    monkeypatch.setattr(
        campaign_renewal,
        "attempted_acquisition_targets",
        lambda owner, entries: seen.append((owner, entries)),
    )
    attempts = f.a.request_history(f.plan, f.campaign, f.old, f.readers)
    assert seen == [(f.branch, [f.entry])]
    assert sum(a["requests"] for a in attempts) == 23


@pytest.mark.parametrize("fault", [None, "manifest", "history"])
def test_persisted_branch_owner_is_available_to_successor_not_caller_flag(
    monkeypatch, tmp_path, fault
):
    f = fixture(monkeypatch, tmp_path)
    f.entry["kind"] = "interrupted_public_material_deferred"
    history = f.proof["previous_attempts"]
    manifest = dict(plan=f.plan.model_dump(mode="json"), request_history_checksum=digest(history))
    records = {
        "accepted-aggregation-manifest": (manifest,),
        "multimodal-case-result": (dict(previous_attempts=history),),
    }
    f.campaign.aggregation_baseline = SimpleNamespace(
        manifest=manifest, documents=SimpleNamespace(read=lambda key: records.get(key))
    )
    successor = SimpleNamespace(parent=f.campaign)
    if fault == "manifest":
        records["accepted-aggregation-manifest"] = ({"caller_authorized": True},)
    if fault == "history":
        records["multimodal-case-result"] = (dict(previous_attempts=[]),)
    if fault:
        with pytest.raises(ValueError, match="source_budget_persistence"):
            f.a.budget_source_owner(successor, f.entry)
    else:
        assert f.a.budget_source_owner(successor, f.entry) is f.branch
        from packages.multimodal_research import interrupted_definition

        verified = []
        monkeypatch.setattr(
            interrupted_definition,
            "verify_history",
            lambda owner, entry: verified.append((owner, entry)),
        )
        assert campaign_renewal.attempted_acquisition_targets(successor, [f.entry]) == set()
        assert verified == [(f.branch, f.entry)]
