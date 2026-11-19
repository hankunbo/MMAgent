"""An unscored increment is not a zero medical score in saved tree audits."""

import copy

import pytest

from packages.evaluation_preparation.cohort_recovery import check_tree
from packages.evaluation_preparation.evidence_tree_v3 import build, render_packet
from packages.grounding.models import digest
from tests.evaluation_preparation.test_unified_view import report


@pytest.mark.parametrize("unscored", [True, False])
def test_exact_saved_score_projection_and_tamper_rejection(unscored):
    value = report()
    if unscored:
        value["scores"] = value["original_review"] = None
    before = copy.deepcopy(value)
    packet = build(value, input_checksum="sha256:test")
    assert check_tree(value, render_packet(packet))
    assert value == before
    packet["scores"] = dict(packet["scores"], invented_score=0.1)
    packet["scores_checksum"] = digest(packet["scores"])
    with pytest.raises(ValueError, match="recovery_audit_tree_binding"):
        check_tree(value, render_packet(packet))
