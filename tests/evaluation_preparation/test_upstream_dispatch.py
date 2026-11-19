"""The new preflight must not authorize work after cancellation or mask rejection."""

from types import SimpleNamespace

import pytest

from packages.evaluation_preparation import case_execution, generic_control, generic_tools
from packages.evidence_reuse import integration


def setup(monkeypatch, tmp_path, check, execute):
    checker = SimpleNamespace(check=check, needs_segman=lambda: True)
    monkeypatch.setattr(integration, "DispatchCheck", lambda *_, **__: checker)
    monkeypatch.setattr(integration, "memory_owner", lambda: None)
    monkeypatch.setattr(case_execution, "segman", execute)
    spec = dict(
        key="test", sample="data_test.npz", upstream_profile="test", fairclip_configuration=None
    )
    registered = (tmp_path, dict(case_code="CASE:TEST", image=dict(checksum="image")), None, None)
    return spec, registered


def test_cancel_during_current_check_does_not_start_tool(monkeypatch, tmp_path):
    cancelled, launches = [], []

    def guard(_):
        if cancelled:
            raise RuntimeError("cancelled")

    def execute(*_, **kwargs):
        kwargs["before_dispatch"]()
        launches.append(1)

    monkeypatch.setattr(generic_control, "guard", guard)
    spec, registered = setup(monkeypatch, tmp_path, lambda: cancelled.append(True), execute)
    with pytest.raises(RuntimeError, match="cancelled"):
        generic_tools.acquire(spec, registered)
    assert launches == []


def test_return_time_source_error_not_disguised_as_execution_failure(monkeypatch, tmp_path):
    calls = []

    def rejected():
        raise ValueError("current_source_revoked")

    def old_failure(*_):
        calls.append(1)
        return dict(exit_code=1)

    monkeypatch.setattr(generic_control, "guard", lambda _: None)
    monkeypatch.setattr(generic_tools, "confirmed_failure", old_failure)
    spec, registered = setup(monkeypatch, tmp_path, rejected, lambda *_, **__: {"confirmed": True})
    with pytest.raises(ValueError, match="current_source_revoked"):
        generic_tools.acquire(spec, registered)
    assert calls == []
    assert not (tmp_path / "tools/segman.json").exists()
