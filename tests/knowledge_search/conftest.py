import socket

import pytest


@pytest.fixture(autouse=True)
def forbid_live_network(monkeypatch):
    calls = []

    def denied(*args, **kwargs):
        calls.append(True)
        raise AssertionError("04B2 standard fixtures cannot access live network")

    monkeypatch.setattr(socket, "socket", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    yield
    assert not calls
