"""Source UNKNOWN lineage is not an attempted image examination or a replay grant."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from packages.multimodal_research.campaign_renewal import attempted_acquisition_targets
from packages.multimodal_research.discovery_continuation import ConfirmedDefinitionContinuation
from tests.evaluation_preparation.test_routed_source_backlog import routed_parent


@pytest.mark.parametrize("error", ["schema", "timeout", "transport"])
def test_routed_original_owner_is_consumed_without_source_or_budget_rewrite(tmp_path, error):
    c, ref, _, _, _, records, writes = routed_parent(tmp_path, final_error=error)
    c.definition_continuation = ConfirmedDefinitionContinuation(c, ref)
    attempts = deepcopy(c.definition_continuation.check_attempts())
    original = deepcopy(records)
    count = len(writes)
    assert attempted_acquisition_targets(c, attempts) == set()
    # Cold successors must find the same owner on the real bounded parent chain.
    child = SimpleNamespace(parent=c, base=c.base, documents=c.documents, contexts=c.contexts)
    assert attempted_acquisition_targets(child, attempts) == set()
    assert records == original and len(writes) == count
    assert attempts[0]["requests"] == 3
    assert attempts[0]["kind"] == "routed_source_failure_deferred"


def test_history_label_without_original_continuation_owner_is_not_authority(tmp_path):
    c, ref, _, _, _, _, _ = routed_parent(tmp_path)
    owner = ConfirmedDefinitionContinuation(c, ref)
    with pytest.raises(ValueError, match="routed_source_history_owner"):
        attempted_acquisition_targets(c, owner.attempts)
    c.parent = c
    with pytest.raises(ValueError, match="routed_source_history_owner"):
        attempted_acquisition_targets(c, owner.attempts)


@pytest.mark.parametrize("change", ["journal", "claim", "request_count", "acquisition_started"])
def test_original_receipt_or_attempt_change_is_rejected(tmp_path, change):
    c, ref, _, _, _, records, _ = routed_parent(tmp_path)
    owner = c.definition_continuation = ConfirmedDefinitionContinuation(c, ref)
    attempts = deepcopy(owner.attempts)
    if change == "journal":
        records["journal:1"][0]["record"]["error"] = "changed"
    elif change == "claim":
        records.pop("definition:confirmed-failure-successor")
    elif change == "request_count":
        attempts[0]["requests"] += 1
    else:
        c.contexts.read_research_run = lambda _: object()
    with pytest.raises(
        ValueError, match="history_changed|routed_source_history_owner|acquisition_started"
    ):
        attempted_acquisition_targets(c, attempts)
