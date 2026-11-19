from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation import generic_capability_aggregation as composition
from packages.evaluation_preparation.dynamic_report import GenericAggregateReportView
from packages.evaluation_preparation.generic_capability_successor import CapabilitySuccessor
from packages.grounding.models import digest
from packages.multimodal_research import accepted_aggregation as owner


def plan():
    return owner.read_plan(
        dict(
            schema_version="accepted-review-aggregation-plan/1.14.0",
            facts="/new",
            origins=[dict(campaign="/old"), dict(campaign="/new")],
        )
    )


@pytest.mark.parametrize("fault", [None, "reverse", "foreign", "duplicate", "local"])
def test_exact_two_origins_and_full_current_projection(monkeypatch, fault):
    campaign = object.__new__(CapabilitySuccessor)
    campaign.spec, campaign.review_key = dict(key="old"), "new"
    campaign.selection_owner = object()
    monkeypatch.setattr(composition, "restore", lambda _: (campaign, {"complete": True}))
    monkeypatch.setattr(composition, "directory", lambda key: Path("/" + key))
    raw = plan().model_dump(mode="json")
    if fault == "reverse":
        raw["origins"].reverse()
    if fault == "foreign":
        raw["origins"][0]["campaign"] = "/foreign"
    if fault == "duplicate":
        raw["origins"].append(raw["origins"][0])
    if fault == "local":
        raw["origins"][0]["local_finalization"] = "/other"
    selected = owner.read_plan(raw)
    if fault:
        with pytest.raises(ValueError, match="exact_origins"):
            owner.plan_campaign(selected)
    else:
        result = owner.plan_campaign(selected)
        assert result[0] is campaign
        assert result[2].keywords["new_items_only"] is False
        assert result[2].keywords["selection_owner"] is campaign.selection_owner


def test_all_reservations_count_once_and_cannot_disappear():
    history = [
        dict(directory="/old", requests=10, checksum=digest("old")),
        dict(directory="/new", requests=4, checksum=digest("new")),
    ]
    readers = [SimpleNamespace(accounting=history[:1]), SimpleNamespace(accounting=history)]
    result = dict(previous_attempts=history, cumulative_requests=14, new_requests=0)
    manifest = dict(request_history_checksum=digest(history))
    assert owner.request_history(plan(), None, None, readers) == history
    owner.verify_request_history(plan(), None, None, readers, manifest, result)
    with pytest.raises(ValueError, match="history_binding"):
        owner.verify_request_history(
            plan(), None, None, readers, manifest, dict(result, cumulative_requests=4)
        )
    readers[1].accounting[1]["requests"] = 31
    with pytest.raises(ValueError, match="budget"):
        owner.request_history(plan(), None, None, readers)


def test_report_reads_new_observation_origin_without_widening_dispatch(tmp_path):
    campaign = object.__new__(CapabilitySuccessor)
    campaign.visual_targets = ("new-item",)
    campaign.output = tmp_path / "new"
    campaign.acquisition = SimpleNamespace(phase_path=lambda target: tmp_path / "old" / target)
    view = GenericAggregateReportView(campaign, tmp_path / "aggregate")
    assert view.observation_phase_path("new-item") == campaign.phase_path("new-item")
    assert view.observation_phase_path("old-item") == tmp_path / "old" / "old-item"
    with pytest.raises(ValueError, match="unselected_phase"):
        campaign.phase_path("old-item")


def test_cli_uses_explicit_capability_ports_without_dispatch(monkeypatch, capsys):
    import sys

    from apps import fairvlmed_case
    from packages.evaluation_preparation import generic_capability_successor

    monkeypatch.setattr(
        generic_capability_successor,
        "preflight",
        lambda key, successor_key: dict(key=key, successor_key=successor_key, api_requests=0),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["fairvlmed_case", "capability-preflight", "--key", "old", "--successor-key", "new"],
    )
    assert fairvlmed_case.main() == 0
    assert '"api_requests": 0' in capsys.readouterr().out


def accounted_plan():
    raw = plan().model_dump(mode="json")
    raw.update(
        schema_version="accepted-review-aggregation-plan/1.15.0",
        accounting="capability_owner_ancestry/1.0.0",
    )
    return owner.read_plan(raw)


def test_original_plan_preparer_defaults_to_accounted_version(monkeypatch, tmp_path):
    import json

    campaign = object.__new__(CapabilitySuccessor)
    campaign.spec, campaign.review_key = dict(key="old"), "new"
    monkeypatch.setattr(composition, "directory", lambda key: tmp_path / key)
    monkeypatch.setattr(composition, "restore", lambda _: (campaign, {"complete": True}))
    output = tmp_path / "plans" / "accounted.json"
    composition.prepare_plan("new", output)
    generated = owner.read_plan(json.loads(output.read_bytes()))
    assert generated.schema_version == "accepted-review-aggregation-plan/1.15.0"
    assert generated.accounting == "capability_owner_ancestry/1.0.0"
    assert [o.campaign for o in generated.origins] == [str(tmp_path / "old"), str(tmp_path / "new")]


def test_versioned_capability_accounting_includes_nonreview_windows(monkeypatch):
    history = [
        dict(directory="/old", requests=10, checksum=digest("old")),
        dict(directory="/acquisition-failed", requests=2, checksum=digest("failed")),
        dict(directory="/acquisition-completed", requests=2, checksum=digest("observed")),
        dict(directory="/review-unsent", requests=0, checksum=digest("unsent")),
        dict(directory="/new", requests=1, checksum=digest("new")),
    ]
    campaign, original = object(), object()

    def original_history(current, result):
        assert current is campaign and result is original
        return history

    monkeypatch.setattr(owner, "capability_request_history", original_history)
    readers = [
        SimpleNamespace(accounting=history[:1]),
        SimpleNamespace(accounting=[history[0], history[-1]]),
    ]
    assert owner.request_history(accounted_plan(), campaign, original, readers) == history
    # Historical 1.14 remains interpretable as the original origin-only ledger.
    assert owner.request_history(plan(), campaign, original, readers) == [history[0], history[-1]]
    manifest = dict(request_history_checksum=digest(history))
    result = dict(previous_attempts=history, cumulative_requests=15, new_requests=0)
    owner.verify_request_history(accounted_plan(), campaign, original, readers, manifest, result)
    with pytest.raises(ValueError, match="history_binding"):
        owner.verify_request_history(
            accounted_plan(),
            campaign,
            original,
            readers,
            manifest,
            dict(result, cumulative_requests=11),
        )
    readers[-1].accounting[-1] = dict(history[-1], requests=2)
    with pytest.raises(ValueError, match="history_conflict"):
        owner.request_history(accounted_plan(), campaign, original, readers)


def accounting_chain(monkeypatch, root_count=10):
    from packages.evaluation_preparation import generic_input

    monkeypatch.setattr(generic_input, "directory", lambda key: Path("/" + key))
    spec = dict(key="old")
    root = dict(
        root_key="old",
        input_checksum=digest(spec),
        result_checksum=digest("root"),
        reservations=root_count,
    )
    previous, total = None, root_count
    for key, used in (("failed", 2), ("observed", 2), ("unsent", 0), ("new", 1)):
        current = object.__new__(CapabilitySuccessor)
        current.spec, current.review_key = spec, key
        current.checked = SimpleNamespace(history=root)
        current.selection = dict(key=key)
        current.protocol = dict(previous=dict(reservations=total, completed_origin=root))
        if previous is not None:
            inventory = dict(
                key=previous.review_key,
                parent_key="old",
                selection_checksum=digest(previous.selection),
                protocol_checksum=digest(previous.protocol),
                new_requests=previous.test_used,
                cumulative_requests=total,
                phases=[
                    dict(
                        reservations=previous.test_used,
                        status="unknown" if previous.review_key == "failed" else "confirmed",
                    )
                ],
            )
            current.selection["continuation"] = dict(
                predecessor_key=previous.review_key,
                inventory=inventory,
                inventory_checksum=digest(inventory),
            )
            current.checked.predecessor = previous
        current.test_used = used
        current.prior = lambda c=current: c.protocol["previous"]
        current.usage = lambda c=current: c.test_used
        previous, total = current, total + used
    return current, dict(new_requests=1, prior_graph_requests=total - 1)


def test_capability_owner_ancestry_preserves_failed_and_zero_request_windows(monkeypatch):
    campaign, result = accounting_chain(monkeypatch)
    history = owner.capability_request_history(campaign, result)
    assert [row["requests"] for row in history] == [10, 2, 2, 0, 1]
    assert [row["directory"] for row in history] == [
        "/old",
        "/failed",
        "/observed",
        "/unsent",
        "/new",
    ]
    assert history[-1]["checksum"] == digest(result)
    assert owner.capability_request_history(campaign, result) == history
    # Accounting retains UNKNOWN costs; it never authorizes their replay.
    assert (
        campaign.checked.predecessor.checked.predecessor.selection["continuation"]["inventory"][
            "phases"
        ][0]["status"]
        == "unknown"
    )


@pytest.mark.parametrize(
    "fault",
    [
        "cost",
        "inventory",
        "protocol",
        "parent",
        "omit",
        "cycle",
        "usage",
        "root",
        "budget",
        "owner_error",
    ],
)
def test_capability_accounting_refuses_missing_or_changed_ancestry(monkeypatch, fault):
    campaign, result = accounting_chain(monkeypatch, root_count=36 if fault == "budget" else 10)
    binding = campaign.selection["continuation"]
    if fault == "cost":
        result["prior_graph_requests"] -= 2
    elif fault == "inventory":
        binding["inventory"]["new_requests"] = 1
    elif fault == "protocol":
        campaign.checked.predecessor.protocol["changed"] = True
    elif fault == "parent":
        binding["predecessor_key"] = "foreign"
    elif fault == "omit":
        campaign.selection.pop("continuation")
    elif fault == "cycle":
        campaign.checked.predecessor = campaign
    elif fault == "usage":
        campaign.test_used = 2
    elif fault == "root":
        campaign.spec = dict(key="foreign")
    elif fault == "owner_error":

        def revoked():
            raise ValueError("original_source_revoked")

        campaign.prior = revoked
    with pytest.raises(ValueError, match="capability_accounting|original_source_revoked"):
        owner.capability_request_history(campaign, result)
