"""Offline guard: no tool execution or HTTP transport in 11A."""

import pytest


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("11A prohibits real HTTP, model and tool execution")

    monkeypatch.setattr("urllib.request.OpenerDirector.open", forbidden)
    monkeypatch.setattr("packages.tool_execution.service.ToolExecutionService.execute", forbidden)
    monkeypatch.setattr(
        "packages.tool_execution.trusted_process.TrustedLocalProcessRunner.run", forbidden
    )
