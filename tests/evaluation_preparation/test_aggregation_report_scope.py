"""Focused export-branch regression; not a substitute for persisted-owner recovery."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.multimodal_research.report import pending_review_scope


def apply_scope(protocol, ids, result):
    path = Path(
        os.environ.get("MMAGENT_REPORT_SCOPE_CANDIDATE", "packages/multimodal_research/report.py")
    )
    tree = ast.parse(path.read_text())
    export = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "export")
    branches = []
    for node in export.body:
        if not isinstance(node, ast.If):
            continue
        condition = ast.unparse(node.test)
        if any(
            marker in condition
            for marker in (
                "review_increment_only",
                "acquisition_increment_only",
                "definition_increment_only",
                "local_accepted_review_aggregation_not_new_model_review",
            )
        ):
            branches.append(node)
    assert len(branches) == 4
    report = dict(
        completion=dict(all_obtained_reviewed=True),
        task_conclusion=dict(limitation="Research only."),
    )
    namespace = dict(
        campaign=SimpleNamespace(
            protocol=protocol,
            base=SimpleNamespace(
                review_target_ids=ids, definition_revision={"added": ["new"]}, ledger=[]
            ),
        ),
        result=result,
        final={"accepted": True},
        scores=dict(reviews=[dict(id="new")], review_completion=dict(pending_ids=[])),
        report=report,
        pending_review_scope=pending_review_scope,
    )
    exec(compile(ast.Module(body=branches, type_ignores=[]), str(path), "exec"), namespace)
    return report


@pytest.mark.parametrize(
    "flag",
    [
        "review_increment_only",
        "acquisition_increment_only",
        "review_acquired_only",
        "definition_increment_only",
    ],
)
def test_local_combination_does_not_apply_inherited_increment_scope(flag):
    protocol = {
        flag: True,
        "tip": {"run_id": "prior"},
        "review_target_ids": ["new"],
        "reuse_assessment_ids": ["old"],
        "pending_review": {"selected_ids": ["new"], "accepted_ids": ["old"]},
    }
    report = apply_scope(
        protocol,
        None,
        dict(
            execution_kind="local_accepted_review_aggregation_not_new_model_review",
            origins=[{"accepted": "old"}, {"accepted": "new"}],
        ),
    )
    assert report["review_scope"]["kind"] == "accepted_local_judgments_reaggregated"
    assert report["review_scope"]["new_model_requests"] == 0
    assert report["completion"]["all_obtained_reviewed"]
    assert "selected_increment_reviewed" not in report["completion"]
    assert "仅补评" not in report["task_conclusion"]["limitation"]
    assert protocol[flag] is True


def test_actual_pending_review_stays_increment_not_whole_case():
    report = apply_scope(
        dict(
            review_increment_only=True,
            tip={"run_id": "prior"},
            pending_review={"selected_ids": ["new"]},
        ),
        ["new"],
        {},
    )
    assert report["review_scope"]["kind"] == "pending_review_increment"
    assert not report["completion"]["all_obtained_reviewed"]
    assert report["completion"]["selected_increment_reviewed"]


def test_real_pending_missing_scope_still_rejects():
    with pytest.raises(TypeError):
        apply_scope(
            dict(
                review_increment_only=True,
                tip={"run_id": "prior"},
                pending_review={"selected_ids": ["new"]},
            ),
            None,
            {},
        )


def test_actual_definition_review_retains_increment_scope():
    report = apply_scope(
        {"definition_increment_only": True, "tip": {"run_id": "prior"}},
        ["new"],
        {"phases": [{"target": "new", "accepted": True}]},
    )
    assert report["review_scope"]["kind"] == "definition_increment_only"
    assert report["review_scope"]["review_ids"] == ["new"]
    assert not report["review_scope"]["whole_case_rescored"]
    assert report["definition_gate_positive"]
    assert not report["completion"]["all_obtained_reviewed"]
