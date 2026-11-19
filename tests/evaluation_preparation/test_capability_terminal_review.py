"""Terminal unsent Review recovery preserves cumulative history and original claims."""

import copy
import json
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation import generic_capability_execution as execution
from packages.evaluation_preparation import generic_capability_recovery as recovery
from packages.evaluation_preparation.generic_capability_successor import (
    CONTINUATION_VERSION,
    REVIEW_CONTINUATION_VERSION,
    selection_for,
)
from packages.grounding.models import digest
from packages.object_store import checksum
from tests.evaluation_preparation.test_capability_continuation import setup


def terminal_fixture(monkeypatch):
    previous, inventory, _ = setup(monkeypatch)
    previous.selection["schema_version"] = CONTINUATION_VERSION
    previous.review_key = "terminal"
    previous.root_claim = lambda: {"key": "original-exclusive-claim"}
    previous.checked.projection["dispatch_targets"] = ["earlier", "current"]
    inventory.update(
        acquired_targets=["earlier", "current"],
        cumulative_requests=14,
        phases=[
            dict(target="current", accepted=True, status="accepted", reservations=1),
            dict(
                target=None,
                accepted=False,
                status="stopped_without_confirmed_response",
                acceptance_key=None,
                reservations=0,
                requests=[],
            ),
        ],
    )
    return previous, inventory


def test_terminal_zero_request_review_has_explicit_version_and_no_vision(monkeypatch):
    previous, inventory = terminal_fixture(monkeypatch)
    frozen = copy.deepcopy(inventory)
    checked = recovery.continuation_origin("terminal", {"path": "exact-history-chain"})
    result = selection_for(checked, "new-review")
    assert result["schema_version"] == REVIEW_CONTINUATION_VERSION
    assert result["continuation"]["root_claim"] == previous.root_claim()
    assert result["continuation"]["dispatch_targets"] == []
    assert result["continuation"]["reuse_targets"] == ["earlier", "current"]
    assert result["continuation"]["inventory"]["cumulative_requests"] == 14
    assert result["expected_vision_requests"] == 0
    assert result["minimum_review_requests"] == 1
    assert result["max_new_requests"] == 10
    assert result["max_cumulative_requests"] == 40
    assert inventory == frozen


@pytest.mark.parametrize(
    "fault",
    [
        "unknown",
        "confirmed",
        "accepted",
        "reserved",
        "receipt",
        "pending_acquisition",
        "missing_observation",
        "duplicate",
        "budget",
        "acknowledged",
    ],
)
def test_terminal_review_does_not_erase_or_reclassify_other_work(monkeypatch, fault):
    previous, inventory = terminal_fixture(monkeypatch)
    review = inventory["phases"][-1]
    if fault == "unknown":
        review["status"] = "unknown_requires_explicit_resolution"
    elif fault == "confirmed":
        review["status"] = "confirmed_response_requires_local_acceptance_check"
    elif fault == "accepted":
        review["accepted"] = True
    elif fault == "reserved":
        review["reservations"] = 1
    elif fault == "receipt":
        review["acceptance_key"] = "accepted:original"
    elif fault == "pending_acquisition":
        inventory["phases"][0]["accepted"] = False
    elif fault == "missing_observation":
        inventory["acquired_targets"].pop()
    elif fault == "duplicate":
        inventory["acquired_targets"].append("earlier")
    elif fault == "budget":
        inventory["cumulative_requests"] = 40
    with pytest.raises(ValueError, match="generic_capability_"):
        recovery.continuation_origin(
            "terminal", {"path": "chain"}, ["old-request"] if fault == "acknowledged" else ()
        )


def test_history_chain_binds_selection_and_does_not_rewrite_old_proof(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "docs/acceptance/test"
    root.mkdir(parents=True)
    path = root / "chain.json"
    value = dict(
        schema_version="generic-capability-history-chain/1.0.0",
        scope="expired_history_read_only_not_dispatch",
        windows={
            "terminal": dict(
                selection_checksum=digest("old"),
                software_reference={"path": "proof", "checksum": "exact"},
            )
        },
    )
    path.write_text(json.dumps(value))
    reference = dict(path=str(path), checksum=checksum(path.read_bytes()))
    before = path.read_bytes()
    assert recovery.software_chain(reference) == value["windows"]
    assert path.read_bytes() == before
    path.write_text(json.dumps(dict(value, scope="new_dispatch")))
    reference["checksum"] = checksum(path.read_bytes())
    with pytest.raises(ValueError, match="chain_scope"):
        recovery.software_chain(reference)


def test_nested_restore_requires_exact_window_selection_before_original_owner(
    tmp_path, monkeypatch
):
    root = tmp_path / "old"
    (root / "evidence").mkdir(parents=True)
    value = dict(schema_version=CONTINUATION_VERSION, key="old")
    for name in ("input.json", "binding.json", "started.json", "evidence/protocol.json"):
        (root / name).write_text(json.dumps(value if name == "input.json" else {}))
    monkeypatch.setattr(execution, "directory", lambda _: root)
    monkeypatch.setattr(
        recovery,
        "software_chain",
        lambda _: {"old": dict(selection_checksum=digest("wrong"), software_reference={})},
    )
    with pytest.raises(ValueError, match="chain_selection"):
        execution.load_selection("old", history_software={"path": "chain"})
    with pytest.raises(ValueError, match="cycle_or_depth"):
        execution.load_selection("old", _seen=("old",))


def test_nested_inventory_counts_ancestor_plus_current_not_original_root_only(
    tmp_path, monkeypatch
):
    from datetime import UTC, datetime, timedelta

    history = {"reservations": 10}
    selection = dict(
        schema_version=CONTINUATION_VERSION,
        original_history=history,
        continuation=dict(inventory=dict(acquired_targets=["earlier"])),
    )
    campaign = SimpleNamespace(
        review_key="old",
        spec={"key": "root"},
        selection=selection,
        selection_owner=SimpleNamespace(current=lambda: ({"original_history": history}, {})),
        checked=SimpleNamespace(history=history),
        visual_targets=("current",),
        protocol={"deadline": (datetime.now(UTC) - timedelta(hours=1)).isoformat()},
        usage=lambda: 2,
        prior=lambda: {"reservations": 12},
        accepted_observations=lambda: {"earlier": "original", "current": "original"},
    )
    monkeypatch.setattr(recovery, "directory", lambda _: tmp_path)
    monkeypatch.setattr(
        recovery,
        "phase_inventory",
        lambda _, target: dict(
            target=target, accepted=target is not None, reservations=2 if target else 0
        ),
    )
    value = recovery.snapshot(campaign)
    assert value["cumulative_requests"] == 14
    assert value["acquired_targets"] == ["earlier", "current"]
    assert value["schema_version"] == "generic-capability-recovery-inventory/1.1.0"


def test_chain_restore_preserves_original_reference_and_exact_frozen_protocol(
    tmp_path, monkeypatch
):
    from packages.evaluation_preparation.generic_capability_successor import CONTINUATION_CLAIM
    from packages.evaluation_preparation.generic_definition_successor import CLAIM
    from tests.evaluation_preparation.test_capability_history_software import fixture, save

    campaign, proof_ref, _, _, _ = fixture(tmp_path, monkeypatch)
    root = campaign.output.parent
    source = tmp_path / "original"
    parent = tmp_path / "parent"
    old_ref = dict(path="historical-proof.json", checksum="historical-exact")
    value = dict(
        schema_version=CONTINUATION_VERSION,
        key="case",
        parent_key="parent",
        continuation=dict(
            predecessor_key="original",
            software_reference=old_ref,
            acknowledged_requests=[],
            root_claim={"key": "original"},
        ),
    )
    paths = {"case": root, "original": source, "parent": parent}
    monkeypatch.setattr(execution, "directory", lambda k: paths[k])
    save(root / "input.json", value)
    save(root / "started.json", dict(key="case", input_checksum=digest(value)))
    save(root / "binding.json", {"case": "same"})
    save(parent / "binding.json", {"case": "same"})
    save(parent / CLAIM, {"key": "original"})
    save(source / CONTINUATION_CLAIM, value)
    chain_path = tmp_path / "docs/acceptance/test/chain.json"
    save(
        chain_path,
        dict(
            schema_version="generic-capability-history-chain/1.0.0",
            scope="expired_history_read_only_not_dispatch",
            windows={"case": dict(selection_checksum=digest(value), software_reference=proof_ref)},
        ),
    )
    chain = dict(path=str(chain_path), checksum=checksum(chain_path.read_bytes()))
    calls = []

    def original(key, reference, acknowledgments, **kwargs):
        calls.append((key, reference, acknowledgments, kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(recovery, "continuation_origin", original)
    monkeypatch.setattr(execution, "selection_for", lambda *a: value)
    monkeypatch.setattr(execution, "CapabilitySuccessor", lambda *a: campaign)
    protected = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    result = execution.load_selection("case", history_software=chain)
    assert result._capability_history_only is True
    assert calls == [("original", old_ref, [], {"history_chain": chain, "_seen": ("case",)})]
    assert protected == {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
