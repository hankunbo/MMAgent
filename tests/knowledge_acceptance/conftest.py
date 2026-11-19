"""04D independent probes use the original session lease and deny external I/O."""

import socket

import pytest


@pytest.fixture(autouse=True)
def offline_boundary(monkeypatch):
    attempts = []
    original_connect = socket.socket.connect
    original_resolve = socket.getaddrinfo

    def connect(sock, address):
        if isinstance(address, tuple) and address[0] not in {"127.0.0.1", "::1", "localhost"}:
            attempts.append("connect")
            raise AssertionError("04D forbids external connections")
        return original_connect(sock, address)

    def resolve(host, *args, **kwargs):
        if host not in {"127.0.0.1", "::1", "localhost", None}:
            attempts.append("DNS")
            raise AssertionError("04D forbids external DNS")
        return original_resolve(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    yield
    assert not attempts, "caught external I/O attempts are also failures"
