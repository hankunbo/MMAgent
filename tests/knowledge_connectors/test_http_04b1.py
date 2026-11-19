import threading
from urllib.parse import parse_qsl, urlsplit

import pytest

from packages.knowledge_connectors import (
    ConnectorError,
    ErrorCategory,
    FakeClock,
    ProviderConfig,
    SafeHTTPClient,
)
from packages.knowledge_connectors.http import (
    PinnedHTTPSConnection,
    PinnedHTTPSTransport,
    WireRequest,
    WireResponse,
)


class ScriptedTransport:
    contact_mode = "synthetic"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def send(self, request, ip, max_bytes):
        self.requests.append((request, ip, max_bytes))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def response(status=200, *, headers=(), body=b"{}"):
    return WireResponse(status, (("content-type", "application/json"),) + headers, body)


def client(configurations, *responses, resolver=None, changes=None, environment=None):
    config = configurations["crossref"]
    if changes:
        config = ProviderConfig.model_validate(config.model_copy(update=changes))
    transport = ScriptedTransport(*responses)
    http = SafeHTTPClient(
        config,
        transport,
        clock=FakeClock(),
        resolver=resolver or (lambda host: ("93.184.216.34",)),
        environment=environment or (lambda name: None),
    )
    return http, transport


def get(http, url="https://api.crossref.org/works"):
    return http.get(url, operation="search", expected_mime=("application/json",))


@pytest.mark.parametrize(
    "url",
    [
        "http://api.crossref.org/works",
        "file:///etc/passwd",
        "https://localhost/x",
        "https://127.0.0.1/x",
        "https://[::1]/x",
        "https://api.crossref.org.evil.example/x",
        "https://api.crossref.org@evil.example/x",
        "https://user:password@api.crossref.org/x",
        "https://api.crossref.org:8443/x",
        "https://api.crossref.org./x",
        "https://api.crossref.org/x#fragment",
        "https://api.crossref.org\\@evil.example/x",
        "https://api.crossref.org/\r\nx",
        "https://2130706433/x",
        "https://0x7f000001/x",
    ],
)
def test_unsafe_url_never_reaches_transport(configurations, url):
    http, transport = client(configurations, response())
    with pytest.raises(ConnectorError) as caught:
        get(http, url)
    assert caught.value.category is ErrorCategory.SECURITY
    assert not transport.requests


@pytest.mark.parametrize(
    "addresses",
    [
        ("127.0.0.1",),
        ("10.1.2.3",),
        ("169.254.169.254",),
        ("100.64.0.1",),
        ("0.0.0.0",),
        ("224.0.0.1",),
        ("::1",),
        ("fc00::1",),
        ("fe80::1",),
        ("::ffff:127.0.0.1",),
        ("2002:7f00:1::",),
        ("93.184.216.34", "10.0.0.1"),
        (),
    ],
)
def test_all_dns_answers_must_be_public(configurations, addresses):
    http, transport = client(configurations, response(), resolver=lambda host: addresses)
    with pytest.raises(ConnectorError) as caught:
        get(http)
    assert caught.value.category is ErrorCategory.SECURITY
    assert not transport.requests


@pytest.mark.parametrize(
    "location",
    [
        "http://api.crossref.org/x",
        "https://evil.example/x",
        "https://www.crossref.org/x",
        "file:///etc/passwd",
    ],
)
def test_redirect_rechecks_allowlist_and_origin(configurations, location):
    http, transport = client(
        configurations, response(302, headers=(("location", location),)), response()
    )
    with pytest.raises(ConnectorError) as caught:
        get(http)
    assert caught.value.category is ErrorCategory.SECURITY
    assert len(transport.requests) == 1


def test_redirect_dns_rebinding_is_rejected(configurations):
    addresses = iter([("93.184.216.34",), ("127.0.0.1",)])
    http, transport = client(
        configurations,
        response(302, headers=(("location", "/next"),)),
        resolver=lambda host: next(addresses),
    )
    with pytest.raises(ConnectorError) as caught:
        get(http)
    assert caught.value.category is ErrorCategory.SECURITY
    assert len(transport.requests) == 1


def test_redirect_budget_and_valid_relative_redirect(configurations):
    http, transport = client(
        configurations, response(302, headers=(("location", "/next"),)), response()
    )
    assert get(http).body == b"{}"
    assert transport.requests[1][0].url.endswith("/next")
    redirects = [response(302, headers=(("location", "/again"),))] * 3
    http, transport = client(configurations, *redirects)
    with pytest.raises(ConnectorError) as caught:
        get(http)
    assert caught.value.category is ErrorCategory.SECURITY and len(transport.requests) == 3


@pytest.mark.parametrize(
    "first,expected",
    [
        (response(429, headers=(("retry-after", "1"),)), ErrorCategory.RATE_LIMIT),
        (response(503), ErrorCategory.HTTP),
        (TimeoutError(), ErrorCategory.TIMEOUT),
        (OSError("secret transport error"), ErrorCategory.NETWORK),
    ],
)
def test_retry_provenance_and_success(configurations, first, expected):
    http, transport = client(configurations, first, response())
    result = get(http)
    assert len(result.metadata.attempts) == 2
    assert result.metadata.attempts[0].error is expected
    assert result.metadata.attempts[-1].http_status == 200
    assert transport.requests[0][0].url == transport.requests[1][0].url
    assert http.clock.elapsed >= 0.25


@pytest.mark.parametrize(
    "status,category",
    [
        (401, ErrorCategory.AUTHENTICATION),
        (403, ErrorCategory.AUTHENTICATION),
        (404, ErrorCategory.HTTP),
    ],
)
def test_nonretryable_status_is_not_empty_result(configurations, status, category):
    http, transport = client(configurations, response(status))
    with pytest.raises(ConnectorError) as caught:
        get(http)
    assert caught.value.category is category and len(transport.requests) == 1


def test_retry_exhaustion_preserves_final_error_and_caps_retry_after(configurations):
    http, transport = client(
        configurations, *[response(429, headers=(("retry-after", "999999"),))] * 3
    )
    with pytest.raises(ConnectorError) as caught:
        get(http)
    assert caught.value.category is ErrorCategory.RETRY_EXHAUSTED
    assert caught.value.last_error is ErrorCategory.RATE_LIMIT
    assert len(caught.value.metadata.attempts) == 3
    assert http.clock.elapsed == 20


@pytest.mark.parametrize(
    "bad,category",
    [
        (response(headers=(("content-type", "text/html"),)), ErrorCategory.PARSE),
        (WireResponse(200, (("content-type", "text/html"),), b"<html/>"), ErrorCategory.MIME),
        (response(headers=(("content-encoding", "gzip"),)), ErrorCategory.RESPONSE_SIZE),
        (response(headers=(("content-length", "5000"),)), ErrorCategory.RESPONSE_SIZE),
        (response(headers=(("content-length", "-1"),)), ErrorCategory.RESPONSE_SIZE),
    ],
)
def test_invalid_size_headers_and_mime(configurations, bad, category):
    http, transport = client(configurations, bad)
    with pytest.raises(ConnectorError) as caught:
        get(http)
    assert caught.value.category is category and len(transport.requests) == 1


def test_actual_body_size_and_request_deadline_are_enforced(configurations):
    http, _ = client(configurations, response(body=b" " * 1048577))
    with pytest.raises(ConnectorError) as caught:
        get(http)
    assert caught.value.category is ErrorCategory.RESPONSE_SIZE
    config = configurations["crossref"]
    policy = config.http.model_copy(update={"timeout_ms": 100, "total_timeout_ms": 100})
    http, _ = client(configurations, response(503), changes={"http": policy})
    with pytest.raises(ConnectorError) as caught:
        get(http)
    assert caught.value.category is ErrorCategory.TIMEOUT


def test_api_key_only_at_transport_boundary_and_redacted_from_outputs(configurations):
    secret = "SYNTHETIC_SECRET_NOT_REAL"
    http, transport = client(
        configurations,
        response(),
        changes={"auth_mode": "query", "api_key_env": "MMAGENT_TEST_KEY"},
        environment=lambda name: secret,
    )
    result = get(http)
    request = transport.requests[0][0]
    assert dict(parse_qsl(urlsplit(request.url).query))["api_key"] == secret
    assert secret not in repr(request)
    assert secret not in repr(result) and secret not in result.model_dump_json()
    assert secret not in http.config.model_dump_json()


@pytest.mark.parametrize("secret", [None, "bad\r\nheader", " ", "x" * 4097])
def test_missing_or_unsafe_secret_is_redacted_authentication_error(configurations, secret):
    http, transport = client(
        configurations,
        response(),
        changes={"auth_mode": "bearer", "api_key_env": "MMAGENT_TEST_KEY"},
        environment=lambda name: secret,
    )
    with pytest.raises(ConnectorError) as caught:
        get(http)
    assert caught.value.category is ErrorCategory.AUTHENTICATION
    assert not transport.requests


def test_rate_limit_applies_to_every_transport_request(configurations):
    http, transport = client(configurations, response(), response(), response())
    for _ in range(3):
        get(http)
    assert len(transport.requests) == 3 and http.clock.elapsed == 1.0


def test_tls_socket_is_pinned_and_uses_original_sni(configurations, monkeypatch):
    import packages.knowledge_connectors.http as module

    observed = {}

    class FakeSocket:
        def settimeout(self, timeout):
            observed["timeout"] = timeout

        def connect(self, endpoint):
            observed["endpoint"] = endpoint

        def close(self):
            observed["closed"] = True

    class Context:
        def wrap_socket(self, sock, *, server_hostname):
            observed["sni"] = server_hostname
            return sock

    monkeypatch.setattr(module.socket, "socket", lambda *args: FakeSocket())
    monkeypatch.setattr(module.ssl, "create_default_context", lambda: Context())
    connection = PinnedHTTPSConnection("api.crossref.org", "93.184.216.34", 1.0)
    connection.connect()
    assert observed["endpoint"] == ("93.184.216.34", 443)
    assert observed["sni"] == "api.crossref.org"
    connection.close()
    assert observed["closed"]


def test_real_transport_reads_bounded_stream_and_closes_on_error(monkeypatch):
    import packages.knowledge_connectors.http as module

    observed = {}

    class Response:
        status = 200

        def getheaders(self):
            return (("content-type", "application/json"),)

        def getheader(self, name):
            return None

        def read1(self, size):
            return b"x" * size

    class Connection:
        sock = None

        def __init__(self, host, ip, timeout):
            observed["ip"] = ip

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return Response()

        def close(self):
            observed["closed"] = True

    monkeypatch.setattr(module, "PinnedHTTPSConnection", Connection)
    with pytest.raises(ConnectorError) as caught:
        PinnedHTTPSTransport().send(
            WireRequest("https://api.crossref.org/works", (), 1.0), "93.184.216.34", 8
        )
    assert caught.value.category is ErrorCategory.RESPONSE_SIZE
    assert observed["closed"] and observed["ip"] == "93.184.216.34"


def test_provider_concurrency_is_bounded(configurations):
    entered = threading.Event()
    release = threading.Event()
    active = 0
    maximum = 0
    lock = threading.Lock()

    class Transport:
        def send(self, request, ip, max_bytes):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            entered.set()
            assert release.wait(2)
            with lock:
                active -= 1
            return response()

    http = SafeHTTPClient(
        configurations["crossref"],
        Transport(),
        clock=FakeClock(),
        resolver=lambda host: ("93.184.216.34",),
    )
    failures = []

    def run():
        try:
            get(http)
        except BaseException as error:
            failures.append(error)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    assert entered.wait(2)
    release.set()
    for thread in threads:
        thread.join(2)
    assert not failures and not any(t.is_alive() for t in threads)
    assert maximum == 1


def test_dns_wait_is_bounded_and_cannot_create_unlimited_workers(configurations):
    import time

    from packages.knowledge_connectors.http import SystemClock

    release = threading.Event()
    done = threading.Event()
    started = []

    def resolver(host):
        started.append(host)
        release.wait(2)
        done.set()
        return ("93.184.216.34",)

    config = configurations["crossref"]
    policy = config.http.model_copy(
        update={
            "timeout_ms": 30,
            "total_timeout_ms": 90,
            "backoff_ms": 0,
            "rate_limit_per_minute": 6000,
        }
    )
    config = ProviderConfig.model_validate(config.model_copy(update={"http": policy}))
    http = SafeHTTPClient(
        config, ScriptedTransport(response()), resolver=resolver, clock=SystemClock()
    )
    start = time.monotonic()
    try:
        with pytest.raises(ConnectorError) as caught:
            get(http)
        assert caught.value.category in (ErrorCategory.TIMEOUT, ErrorCategory.RETRY_EXHAUSTED)
        assert time.monotonic() - start < 1
        assert len(started) == 1
    finally:
        release.set()
        assert done.wait(1)


def test_tls_verification_failure_is_security_error_without_retry(configurations):
    import ssl

    http, transport = client(configurations, ssl.SSLCertVerificationError("synthetic invalid cert"))
    with pytest.raises(ConnectorError) as caught:
        get(http)
    assert caught.value.category is ErrorCategory.SECURITY
    assert len(transport.requests) == 1


def test_bearer_credential_is_never_forwarded_across_origins(configurations):
    secret = "SYNTHETIC_SECRET_NOT_REAL"
    http, transport = client(
        configurations,
        response(302, headers=(("location", "https://www.crossref.org/x"),)),
        changes={"auth_mode": "bearer", "api_key_env": "MMAGENT_TEST_KEY"},
        environment=lambda key: secret,
    )
    with pytest.raises(ConnectorError) as caught:
        get(http)
    assert caught.value.category is ErrorCategory.SECURITY and len(transport.requests) == 1
    assert dict(transport.requests[0][0].headers)["Authorization"] == "Bearer " + secret
    assert (
        secret not in repr(caught.value) and secret not in caught.value.metadata.model_dump_json()
    )
