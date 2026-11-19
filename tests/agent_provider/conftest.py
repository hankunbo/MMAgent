"""Standard 08C suite is offline. Even accidental HTTP attempts fail."""

import pytest


@pytest.fixture(autouse=True)
def no_network_or_tools(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("08C forbids real network/model/tool calls")

    monkeypatch.setattr("urllib.request.OpenerDirector.open", forbidden)
    from packages.tool_execution.service import ToolExecutionService
    from packages.tool_workflow.compiler import ToolWorkflowCompiler

    monkeypatch.setattr(ToolExecutionService, "execute", forbidden)
    monkeypatch.setattr(ToolWorkflowCompiler, "compile", forbidden)
