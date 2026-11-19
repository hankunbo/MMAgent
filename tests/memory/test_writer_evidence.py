"""Stopped-run evidence is read-only; legacy/current simulation authority stays distinct."""

import pytest

from packages.counterfactual.service import render
from packages.grounding.models import digest
from tests.counterfactual.support import candidate, fixture, run
from tests.counterfactual.test_types import replacing


def stop(x):
    x.h.planner.step(x.f.binding.run_id, x.h.command("stop"))


def test_evidence_keeps_precise_cross_run_dependencies_and_no_execution(monkeypatch):
    x, alternative = replacing("replace_tool")
    c = candidate("replace_tool", alternative="alt")
    report = run(x, c)["report"]
    stop(x)

    def forbidden(*args, **kwargs):
        raise AssertionError("read evidence must not execute or observe cancellation")

    monkeypatch.setattr(x.service, "_gate", forbidden)
    monkeypatch.setattr(x.service, "_observe", forbidden)
    assert x.service.read_evidence(candidate_id=c.id) == report
    alternative.f.reader.withdrawn = True
    with pytest.raises(ValueError, match="source_withdrawn"):
        x.service.read_evidence(candidate_id=c.id)
    assert x.service.read(candidate_id=c.id, historical=True) == report


def test_evidence_legacy_report_rejected_without_upgrade():
    x = fixture()
    r = run(x)["report"]
    old = dict(r, schema_version="counterfactual-report/1.0.0")
    old["checksum"] = digest({k: v for k, v in old.items() if k != "checksum"})
    x.f.documents.saved["simulation:result:one"] = (old, render(old))
    stop(x)
    before = digest(x.f.documents.saved)
    with pytest.raises(ValueError, match="legacy_semantics"):
        x.service.read_evidence(candidate_id="one")
    assert x.service.read(candidate_id="one", historical=True) == old
    assert digest(x.f.documents.saved) == before


def test_evidence_unknown_io_propagates_and_active_control_refuses(monkeypatch):
    x = fixture()
    report = run(x)["report"]
    with pytest.raises(ValueError, match="simulation_evidence_terminal_required"):
        x.service.read_evidence(candidate_id="one")
    stop(x)

    def unknown():
        raise OSError("synthetic object io")

    monkeypatch.setattr(x.service, "reader", unknown)
    with pytest.raises(OSError, match="synthetic object io"):
        x.service.read_evidence(candidate_id="one")
    assert x.service.read(candidate_id="one", historical=True) == report
