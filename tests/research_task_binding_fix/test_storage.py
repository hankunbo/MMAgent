"""Small representation tests, not substitutes for the actual PG owner chain."""

import json
from copy import deepcopy

import pytest

from db.runtime.repository.research_report_storage import (
    EXACT,
    LEGACY,
    decode_lifecycle,
    encode_lifecycle,
)
from packages.grounding.models import canonical, digest
from tests.research_task_binding_r01.diagnostics import constructed_report, scalar_value


def state(value=-0.0, version=EXACT):
    report, _ = constructed_report(scalar_value(value))
    return dict(version=version, run_id=report["run_id"], phase="FINISHED", report=report)


@pytest.mark.parametrize(
    "value",
    [0.0, -0.0, -1.0, -0.125, 0.1],
    ids=["positive-zero", "negative-zero", "negative", "decimal-exact", "decimal"],
)
def test_exact_report_codec_preserves_original_checksum_and_h_value(value):
    original = state(value)
    encoded = encode_lifecycle(original)
    assert isinstance(encoded["report"]["canonical_json"], str)
    restored = decode_lifecycle(json.loads(canonical(encoded)))
    assert canonical(restored) == canonical(original)
    assert restored["report"]["model_entries"][0]["value"].hex() == value.hex()
    assert canonical(original) == canonical(state(value))


@pytest.mark.parametrize(
    "field",
    [
        "run_id",
        "plan_checksum",
        "config_checksum",
        "policy_checksum",
        "score_key",
        "execution_key",
        "checksum",
    ],
)
def test_constructed_report_tampering_is_not_resealed(field):
    original = state()
    original["report"][field] = "changed"
    with pytest.raises(ValueError, match="storage_integrity"):
        encode_lifecycle(original)


@pytest.mark.parametrize(
    "change", ["body", "checksum", "encoding", "extra", "noncanonical", "inner-resealed-only"]
)
def test_storage_tampering_and_noncanonical_content_rejected(change):
    wire = deepcopy(encode_lifecycle(state()))
    report = wire["report"]
    if change == "body":
        report["canonical_json"] = report["canonical_json"].replace("-0.0", "1.0")
    elif change == "checksum":
        report["checksum"] = digest("wrong")
    elif change == "encoding":
        report["version"] = "research-case-report-storage/99.0.0"
    elif change == "extra":
        report["unbound"] = True
    elif change == "noncanonical":
        report["canonical_json"] += " "
    else:
        value = json.loads(report["canonical_json"])
        value["model_entries"][0]["value"] = 0.0
        value["checksum"] = digest({k: v for k, v in value.items() if k != "checksum"})
        report["canonical_json"] = canonical(value).decode()
    with pytest.raises(ValueError, match="storage_"):
        decode_lifecycle(wire)


def test_legacy_report_keys_keep_exact_old_dispatch_even_invalid_old_checksum():
    old = state(version=LEGACY)
    assert encode_lifecycle(old) is old
    assert decode_lifecycle(old) is old
    # Legacy storage never repairs historical normalized bytes or upgrades a key.
    old["report"]["model_entries"][0]["value"] = 0.0
    assert decode_lifecycle(old) is old
    assert old["report"]["checksum"] != digest(
        {k: v for k, v in old["report"].items() if k != "checksum"}
    )
    wire = encode_lifecycle(state())
    wire["version"] = LEGACY
    assert decode_lifecycle(wire)["report"]["version"] == "research-case-report-storage/1.0.0"
    # A legacy reader will not silently unpack a new storage envelope.


def test_unfinished_and_unknown_version_are_not_upgraded():
    pending = dict(version=EXACT, run_id="RUN:SYNTHETIC", phase="SCORING", report=None)
    assert decode_lifecycle(encode_lifecycle(pending)) == pending
    for version in ("research-single-case/1.2.0", "invalid"):
        with pytest.raises(ValueError, match="storage_version"):
            decode_lifecycle(dict(pending, version=version))
    with pytest.raises(ValueError, match="storage_missing"):
        decode_lifecycle(dict(pending, phase="FINISHED"))
