"""Offline proxy routing, CONNECT identity, limits and no silent direct fallback."""

import http.client
import ssl

import pytest

from apps.knowledge_source_tools import configuration, page_client, transport_receipt
from packages.knowledge_connectors import (
    ConnectorError,
    ErrorCategory,
    ExplicitProxyHTTPSTransport,
    FakeClock,
    PinnedHTTPSTransport,
    ProxyPolicy,
    SafeHTTPClient,
    proxy,
)
from packages.knowledge_connectors.http import WireRequest

POLICY = ProxyPolicy(proxy_environment="HTTPS_PROXY")
REQUEST = WireRequest("https://www.nice.org.uk/guidance/ng81", (), 1.0)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "socks5://proxy:80",
        "https://proxy:80",
        "http://user:secret@proxy:80",
        "http://proxy:0",
        "http://proxy:99999",
        "http://proxy/path",
        "http://proxy?secret=1",
        "http://proxy#fragment",
        "http://proxy\r\nInjected:1",
        "http://proxy\\evil",
    ],
)
def test_invalid_proxy_is_sanitized_and_never_connects(monkeypatch, value):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid configuration must not create a connection")

    monkeypatch.setattr(proxy, "_TunnelConnection", forbidden)
    transport = ExplicitProxyHTTPSTransport(POLICY, environment=lambda _: value)
    with pytest.raises(ConnectorError) as caught:
        transport.send(REQUEST, "93.184.216.34", 100)
    assert caught.value.category is ErrorCategory.CONFIGURATION
    assert "secret" not in str(caught.value)


def test_only_nice_uses_explicit_policy_and_new_provider_version():
    assert isinstance(
        page_client("nice", configuration("nice")).transport, ExplicitProxyHTTPSTransport
    )
    assert configuration("nice").provider_version == "1.1.0"
    for key in ("nei", "egs", "dalhousie"):
        assert type(page_client(key, configuration(key)).transport) is PinnedHTTPSTransport
        assert configuration(key).provider_version == "1.0.0"
    assert transport_receipt("nice")["policy_checksum"] == POLICY.checksum()
    assert POLICY.checksum() != ProxyPolicy(proxy_environment="OTHER_PROXY").checksum()


@pytest.mark.parametrize("resolution", ["local_public_ip_pinned", "proxy_hostname"])
def test_connect_declared_destination_preserves_sni_and_certificate_validation(
    monkeypatch, resolution
):
    seen = {}
    raw, tls = object(), object()

    class Tunnel:
        def __init__(self, host, port, timeout):
            seen["proxy"] = (host, port, timeout)
            self.sock = raw

        def set_tunnel(self, host, port):
            seen["destination"] = (host, port)

        def connect(self):
            pass

        def close(self):
            seen["detached"] = self.sock is None

    class Context:
        def wrap_socket(self, sock, server_hostname):
            assert sock is raw
            seen["sni"] = server_hostname
            return tls

    transport = ExplicitProxyHTTPSTransport(
        ProxyPolicy(proxy_environment="HTTPS_PROXY", target_resolution=resolution),
        environment=lambda _: "http://proxy:80",
    )
    connection = transport._connection("www.nice.org.uk", "93.184.216.34", 1)
    assert connection._context.verify_mode == ssl.CERT_REQUIRED
    assert connection._context.check_hostname
    connection._context = Context()
    monkeypatch.setattr(http.client, "HTTPConnection", Tunnel)
    connection.connect()
    destination = "93.184.216.34" if resolution == "local_public_ip_pinned" else "www.nice.org.uk"
    assert seen["destination"] == (destination, 443)
    assert seen["sni"] == "www.nice.org.uk" and seen["detached"]
    assert connection.sock is tls


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1"])
@pytest.mark.parametrize("resolution", ["local_public_ip_pinned", "proxy_hostname"])
def test_private_destination_rejected_before_proxy_resolution(ip, resolution):
    def forbidden(_):
        pytest.fail("unsafe target must not read proxy environment")

    with pytest.raises(ConnectorError) as caught:
        ExplicitProxyHTTPSTransport(
            ProxyPolicy(proxy_environment="HTTPS_PROXY", target_resolution=resolution),
            environment=forbidden,
        ).send(REQUEST, ip, 100)
    assert caught.value.category is ErrorCategory.SECURITY


def test_response_size_enforced_and_connection_closed(monkeypatch):
    seen = {}

    class Response:
        status = 200

        def getheaders(self):
            return (("content-type", "text/html"),)

        def getheader(self, name):
            return None

        def read1(self, size):
            return b"x" * size

    class Connection:
        sock = None

        def __init__(self, *args):
            pass

        def request(self, method, path, headers):
            assert method == "GET" and path == "/guidance/ng81"
            assert "Proxy-Authorization" not in headers

        def getresponse(self):
            return Response()

        def close(self):
            seen["closed"] = True

    monkeypatch.setattr(proxy, "_TunnelConnection", Connection)
    transport = ExplicitProxyHTTPSTransport(POLICY, environment=lambda _: "http://proxy:80")
    with pytest.raises(ConnectorError) as caught:
        transport.send(REQUEST, "93.184.216.34", 8)
    assert caught.value.category is ErrorCategory.RESPONSE_SIZE and seen["closed"]


@pytest.mark.parametrize(
    "error,category",
    [
        (OSError("secret-proxy-host"), ErrorCategory.NETWORK),
        (TimeoutError("secret-proxy-host"), ErrorCategory.TIMEOUT),
        (ssl.SSLCertVerificationError("secret-proxy-host"), ErrorCategory.SECURITY),
    ],
)
@pytest.mark.parametrize("resolution", ["local_public_ip_pinned", "proxy_hostname"])
def test_no_direct_fallback_and_sanitized_errors(monkeypatch, error, category, resolution):
    class Connection:
        def __init__(self, *args):
            pass

        def request(self, *args, **kwargs):
            raise error

        def close(self):
            pass

    connection_name = (
        "_TunnelConnection"
        if resolution == "local_public_ip_pinned"
        else "_HostnameTunnelConnection"
    )
    monkeypatch.setattr(proxy, connection_name, Connection)
    transport = ExplicitProxyHTTPSTransport(
        ProxyPolicy(proxy_environment="HTTPS_PROXY", target_resolution=resolution),
        environment=lambda _: "http://proxy:80",
    )
    with pytest.raises(ConnectorError) as caught:
        transport.send(REQUEST, "93.184.216.34", 100)
    assert caught.value.category is category
    assert "secret-proxy-host" not in str(caught.value)


def test_allowlist_rejected_before_proxy_send():
    def forbidden(_):
        pytest.fail("disallowed source must never reach proxy")

    transport = ExplicitProxyHTTPSTransport(POLICY, environment=forbidden)
    client = SafeHTTPClient(
        configuration("nice"), transport, clock=FakeClock(), resolver=lambda _: ("93.184.216.34",)
    )
    with pytest.raises(ConnectorError) as caught:
        client.get("https://example.com/", operation="fetch", expected_mime=("text/html",))
    assert caught.value.category is ErrorCategory.SECURITY


def test_proxy_policy_preserves_legacy_receipt_and_names_hostname_trust():
    assert POLICY.receipt() == {
        "transport": "http-connect-public-ip/1.0.0",
        "proxy_environment": "HTTPS_PROXY",
        "target_resolution": "local_public_ip_pinned",
        "tls_verification": True,
        "direct_fallback": False,
    }
    hostname = ProxyPolicy(proxy_environment="HTTPS_PROXY", target_resolution="proxy_hostname")
    assert hostname.receipt()["transport"] == "http-connect-hostname/1.0.0"
    assert hostname.receipt()["target_resolution"] == "proxy_hostname"
    assert hostname.checksum() != POLICY.checksum()
