"""Exact bounded chunks, including embedded images; no semantic transformations."""

from contextlib import nullcontext
from copy import deepcopy

import pytest

from packages.grounding.models import canonical, digest
from packages.multimodal_research.report_storage import VERSION, pack, save, unpack


def test_small_legacy_and_large_two_part_roundtrip():
    report = {"schema_version": "dynamic-case-report/1.0.0", "text": "原观察<&>"}
    assert pack(report, "html", key="report") == (
        report,
        dict(html="html", report_checksum=digest(report)),
    )
    report["large"] = "图" * 1500000
    page = "<img data>" * 600000
    values = pack(report, page, key="report")
    assert values[0]["schema_version"] == VERSION
    assert all(len(canonical(v)) < 4190000 for v in values)
    assert unpack(values, key="report") == (report, dict(html=page, report_checksum=digest(report)))


@pytest.mark.parametrize("fault", ["key", "order", "omit", "extra", "checksum", "bound"])
def test_chunk_corruption_or_cross_binding_rejected(fault):
    values = list(deepcopy(pack({"value": 1}, "x" * 4000000, key="report")))
    if fault == "key":
        values[0]["report_key"] = "other"
    elif fault == "order":
        values[1], values[2] = values[2], values[1]
    elif fault == "omit":
        values.pop()
    elif fault == "extra":
        values.append(values[-1])
    elif fault == "checksum":
        values[1]["checksum"] = "sha256:" + "0" * 64
    else:
        values[0]["parts"][0]["bytes"] = 100000000
    with pytest.raises(ValueError):
        unpack(tuple(values), key="report")


@pytest.mark.parametrize("legacy", [True, False])
def test_logical_replay_keeps_original_storage_and_rejects_content_changes(legacy):
    report, page = {"text": "original"}, "x" * 3500000

    class Documents:
        values = (report, dict(html=page, report_checksum=digest(report))) if legacy else ()
        writes = 0

        def session(self):
            return nullcontext()

        def read(self, key):
            assert key == "report"
            return self.values

        def write(self, key, values):
            assert key == "report" and not self.values
            self.values = values
            self.writes += 1

    docs = Documents()
    save(docs, report, page, key="report")
    physical = canonical(docs.values)
    save(docs, report, page, key="report")
    assert canonical(docs.values) == physical and docs.writes == (0 if legacy else 1)
    assert len(docs.values) == 2 if legacy else docs.values[0]["schema_version"] == VERSION
    with pytest.raises(ValueError, match="multimodal_report_content_conflict"):
        save(docs, report, page + "changed", key="report")
    assert canonical(docs.values) == physical
