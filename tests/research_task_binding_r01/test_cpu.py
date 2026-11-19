"""No PG or process launches. Constructor behavior is not JSONB evidence."""

import json
import math
from types import SimpleNamespace

import pytest

from packages.case_score_ledger.models import ScoreEnvelope
from packages.case_score_ledger.service import CaseScoreService
from packages.grounding.models import canonical, digest
from tests.case_score_ledger.test_models import definition
from tests.research_task_binding_r01.diagnostics import (
    binding_checks,
    constructed_report,
    differences,
    png,
    scalar_value,
)


@pytest.mark.parametrize(
    "value",
    [0.0, -0.0, -1.0, -0.125, 0.1],
    ids=["positive-zero", "negative-zero", "negative", "decimal-exact", "decimal"],
)
def test_cpu_legal_scalar_report_and_exact_h_bytes(value):
    decoded = scalar_value(value)
    assert decoded.hex() == value.hex()
    report, ledger = constructed_report(decoded)
    restored = ScoreEnvelope.model_validate_json(canonical(ledger))
    assert restored.entries[0].value.hex() == value.hex()
    assert restored.checksum == ledger.checksum
    assert report["model_entries"][0]["value"].hex() == value.hex()
    assert report["checksum"] == digest({k: v for k, v in report.items() if k != "checksum"})
    assert canonical(json.loads(canonical(report))) == canonical(report)


@pytest.mark.parametrize("pixel,expected", [(0, -0.0), (1, -1.0)], ids=["zero", "one"])
def test_cpu_original_png_decoder_legal_output_range(pixel, expected):
    value = CaseScoreService._decode(
        SimpleNamespace(synthetic=True),
        png(bytes([pixel]) * 6),
        "synthetic_png_logit/1.0.0",
        "value",
        definition(),
    )
    assert value.hex() == expected.hex()
    assert math.copysign(1, value) == -1


def test_cpu_zero_digest_distinction_is_not_a_pg_reproduction():
    positive, _ = constructed_report(0.0)
    negative, _ = constructed_report(-0.0)
    assert positive == negative or positive["model_entries"] == negative["model_entries"]
    assert positive["checksum"] != negative["checksum"]
    delta = differences(negative, positive)
    assert any(d["path"] == "$.model_entries[0].value" and d["numeric_equal"] for d in delta)


def test_cpu_diagnostic_lists_each_field_and_detects_postconstruction_tampering():
    report, _ = constructed_report(-0.0)
    state = dict(run_id=report["run_id"], binding=report)
    task = dict(
        plan_checksum=report["plan_checksum"],
        config_checksum=report["config_checksum"],
        score_policy=report["policy_checksum"],
    )
    assert all(c["matches"] for c in binding_checks(report, state, task).values())
    for field in binding_checks(report, state, task):
        changed = dict(report, **{field: False if field == "terminal" else "changed"})
        checks = binding_checks(changed, state, task)
        assert not checks[field]["matches"]
        assert not checks["checksum"]["matches"]
    typed = dict(report, terminal=1)
    assert not binding_checks(typed, state, task)["terminal"]["same_type"]
