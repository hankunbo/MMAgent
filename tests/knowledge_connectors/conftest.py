"""All standard connector tests are offline, including DNS and socket creation."""

import socket
from pathlib import Path

import pytest

from packages.knowledge_connectors import load_configuration


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    attempts = []

    def denied(*args, **kwargs):
        attempts.append(True)
        raise AssertionError("standard 04B1 tests must not access live DNS or sockets")

    monkeypatch.setattr(socket, "getaddrinfo", denied)
    monkeypatch.setattr(socket, "socket", denied)
    yield
    assert not attempts, "a connector attempted live network access even if it caught the error"


@pytest.fixture
def configurations():
    configuration = load_configuration(Path("configs/knowledge_sources.toml"))
    return {p.adapter: p for p in reversed(configuration.providers)}
