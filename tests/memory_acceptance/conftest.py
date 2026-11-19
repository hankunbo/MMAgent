"""No real network, tools, model processes or PostgreSQL in the new small probes."""

import pytest

from tests.expert_agent.conftest import no_network_or_tools  # noqa: F401


@pytest.fixture(autouse=True)
def no_process_or_database(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("10E preflight forbids process/database execution")

    monkeypatch.setattr("subprocess.Popen", forbidden)
    monkeypatch.setattr("os.system", forbidden)
    monkeypatch.setattr("psycopg.connect", forbidden)
