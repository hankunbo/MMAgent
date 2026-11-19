"""Offline test boundary; HTTP uses ASGI only and PG uses the original lease."""

import pytest


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("12A forbids external HTTP, real tools, models and GPU")

    monkeypatch.setattr("urllib.request.OpenerDirector.open", forbidden)
    monkeypatch.setattr("packages.tool_execution.service.ToolExecutionService.execute", forbidden)
    monkeypatch.setattr(
        "packages.tool_execution.trusted_process.TrustedLocalProcessRunner.run", forbidden
    )
