"""Bounded HTTPS, exact allowlists, DNS/IP pinning, retries and safe provenance."""

from __future__ import annotations

import http.client
import ipaddress
import os
import queue
import socket
import ssl
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from pydantic import ValidationError

from .configuration import ProviderConfig, https_url, validate_config, validate_contact_email
from .ports import (
    Attempt,
    ConnectorError,
    ErrorCategory,
    RequestMetadata,
    TransientResponse,
    WireReceipt,
    digest,
)


@dataclass(frozen=True)
class WireRequest:
    url: str = field(repr=False)
    headers: tuple[tuple[str, str], ...] = field(repr=False)
    timeout_seconds: float


@dataclass(frozen=True)
class WireResponse:
    status: int
    headers: tuple[tuple[str, str], ...]
    body: bytes = field(repr=False)


@dataclass
class _RequestWindow:
    """One connector network/retry window; owner admission has an outer deadline.

    Only explicit owner preparation and receipt persistence are excluded.
    DNS, transport, redirects,
    backoff and rate waiting share this window; the governor's absolute source
    and campaign deadlines remain authoritative at every subsequent boundary.
    """

    expires_at: float


class _RequestPreparationError(ConnectorError):
    """Local identity/request failure must propagate, including through healthcheck."""


class _ValidatedContact:
    """Request-local immutable value; no model, dict, dataclass or pickle export."""

    __slots__ = ("__parameters",)

    def __init__(self, parameters: tuple[tuple[str, str], ...]):
        object.__setattr__(self, "_ValidatedContact__parameters", parameters)

    def __setattr__(self, name, value):
        raise AttributeError("validated contact is immutable")

    def __delattr__(self, name):
        raise AttributeError("validated contact is immutable")

    def __reduce_ex__(self, protocol):
        raise TypeError("validated contact cannot be serialized")

    def _items(self) -> tuple[tuple[str, str], ...]:
        return self.__parameters


class Transport(Protocol):
    # Explicit trust declaration: synthetic means no live outbound capability.
    # Implementations/wrappers without the declaration default to production.
    contact_mode: Literal["production", "synthetic"]

    def send(self, request: WireRequest, ip: str, max_bytes: int) -> WireResponse: ...


class RequestGovernor(Protocol):
    """Optional shared admission port. It never receives requests or identities."""

    def before_send(self, provider_code: str) -> float: ...
    def after_send(
        self, provider_code: str, status: int, headers: tuple[tuple[str, str], ...]
    ) -> None: ...
    def remaining(self) -> float: ...
    def retry_limit(self, configured_limit: int) -> int: ...
    def wait(self, seconds: float) -> None: ...
    def completed_request(self, metadata: RequestMetadata) -> None: ...
    def completed_wire(self, receipt: WireReceipt) -> None: ...


class Clock(Protocol):
    def now(self) -> datetime: ...
    def monotonic(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


def system_resolver(host: str) -> tuple[str, ...]:
    return tuple(
        sorted({entry[4][0] for entry in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)})
    )


def public_addresses(addresses: tuple[str, ...]) -> tuple[str, ...]:
    if not addresses or type(addresses) is not tuple:
        raise ConnectorError(ErrorCategory.SECURITY)
    for address in addresses:
        try:
            value = ipaddress.ip_address(address)
        except ValueError:
            raise ConnectorError(ErrorCategory.SECURITY) from None
        if (
            not value.is_global
            or value.is_multicast
            or value.is_reserved
            or getattr(value, "ipv4_mapped", None)
            or getattr(value, "sixtofour", None)
            or getattr(value, "teredo", None)
        ):
            raise ConnectorError(ErrorCategory.SECURITY)
    return addresses


class PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Connect to the validated IP, while TLS and Host use the original DNS name."""

    def __init__(self, host: str, ip: str, timeout: float):
        super().__init__(host, port=443, timeout=timeout, context=ssl.create_default_context())
        self._pinned_ip = ip

    def connect(self) -> None:
        public_addresses((self._pinned_ip,))
        family = socket.AF_INET6 if ":" in self._pinned_ip else socket.AF_INET
        raw = socket.socket(family, socket.SOCK_STREAM)
        try:
            raw.settimeout(self.timeout)
            raw.connect((self._pinned_ip, 443))
            self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
        except BaseException:
            raw.close()
            raise


class PinnedHTTPSTransport:
    """No proxy environment, redirects, cookies or unverified TLS in this layer."""

    contact_mode = "production"

    def _connection(self, host: str, ip: str, timeout: float):
        return PinnedHTTPSConnection(host, ip, timeout)

    def send(self, request: WireRequest, ip: str, max_bytes: int) -> WireResponse:
        parsed = urlsplit(request.url)
        connection = self._connection(parsed.hostname, ip, request.timeout_seconds)
        deadline = time.monotonic() + request.timeout_seconds
        try:
            path = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
            connection.request("GET", path, headers=dict(request.headers))
            response = connection.getresponse()
            headers = tuple(response.getheaders())
            length = response.getheader("Content-Length")
            if length is not None and (not length.isdecimal() or int(length) > max_bytes):
                raise ConnectorError(ErrorCategory.RESPONSE_SIZE)
            chunks: list[bytes] = []
            size = 0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                if connection.sock is not None:
                    connection.sock.settimeout(remaining)
                chunk = response.read1(min(65536, max_bytes + 1 - size))
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise ConnectorError(ErrorCategory.RESPONSE_SIZE)
                chunks.append(chunk)
            body = b"".join(chunks)
            if length is not None and len(body) != int(length):
                raise ConnectorError(ErrorCategory.NETWORK)
            return WireResponse(response.status, headers, body)
        finally:
            connection.close()


class SafeHTTPClient:
    def __init__(
        self,
        config: ProviderConfig,
        transport: Transport,
        *,
        resolver: Callable[[str], tuple[str, ...]] = system_resolver,
        clock: Clock | None = None,
        environment: Callable[[str], str | None] = os.environ.get,
        governor: RequestGovernor | None = None,
    ):
        self.config = validate_config(config)
        self.transport = transport
        self.resolver = resolver
        self.clock = clock or SystemClock()
        self.environment = environment
        self.governor = governor
        self._slots = threading.BoundedSemaphore(config.http.concurrency)
        self._dns_slots = threading.BoundedSemaphore(config.http.concurrency)
        self._rate_lock = threading.Lock()
        self._next_request = 0.0

    def _environment_value(self, name: str, category: ErrorCategory) -> str | None:
        try:
            return self.environment(name)
        except Exception:
            raise ConnectorError(category) from None

    def _validated_contact(self, config: ProviderConfig) -> _ValidatedContact:
        # Resolve only for adapters that own a contact policy, exactly once per wire attempt.
        if config.adapter not in ("pubmed", "crossref"):
            return _ValidatedContact(())
        mode = getattr(self.transport, "contact_mode", "production")
        if type(mode) is not str or mode not in ("production", "synthetic"):
            raise ConnectorError(ErrorCategory.CONFIGURATION)
        if config.contact_email_env is None:
            if config.adapter == "pubmed":
                raise ConnectorError(ErrorCategory.CONFIGURATION)
            return _ValidatedContact(())
        email = self._environment_value(config.contact_email_env, ErrorCategory.CONFIGURATION)
        try:
            validate_contact_email(email)
        except ValueError:
            raise ConnectorError(ErrorCategory.CONFIGURATION) from None
        domain = email.rsplit("@", 1)[1].lower()
        reserved = (
            "invalid",
            "example",
            "test",
            "localhost",
            "local",
            "internal",
            "example.org",
            "example.com",
            "example.net",
        )
        if mode == "production" and any(
            domain == suffix or domain.endswith("." + suffix) for suffix in reserved
        ):
            raise ConnectorError(ErrorCategory.CONFIGURATION)
        if config.adapter == "pubmed":
            return _ValidatedContact((("tool", config.pubmed_tool), ("email", email)))
        return _ValidatedContact((("mailto", email),))

    def _remaining(self, deadline: float | _RequestWindow) -> float:
        expires_at = deadline.expires_at if isinstance(deadline, _RequestWindow) else deadline
        remaining = expires_at - self.clock.monotonic()
        if self.governor is not None:
            remaining = min(remaining, self.governor.remaining())
        if remaining <= 0:
            raise ConnectorError(ErrorCategory.TIMEOUT)
        return remaining

    def _wait(self, seconds: float, deadline: float | _RequestWindow) -> None:
        if seconds >= self._remaining(deadline):
            raise ConnectorError(ErrorCategory.TIMEOUT)
        if self.governor is not None:
            self.governor.wait(max(0, seconds))
        else:
            self.clock.sleep(max(0, seconds))

    def _rate_limit(self, deadline: float | _RequestWindow) -> None:
        with self._rate_lock:
            now = self.clock.monotonic()
            delay = max(0, self._next_request - now)
            self._wait(delay, deadline)
            self._next_request = (
                self.clock.monotonic() + 60 / self.config.http.rate_limit_per_minute
            )

    def _resolve(self, host: str, deadline: float | _RequestWindow) -> tuple[str, ...]:
        # libc DNS cannot be cancelled. Bound waiting and outstanding daemon workers.
        timeout = min(self.config.http.timeout_ms / 1000, self._remaining(deadline))
        if not self._dns_slots.acquire(timeout=timeout):
            raise TimeoutError
        result = queue.Queue(maxsize=1)

        def resolve() -> None:
            try:
                result.put((True, self.resolver(host)))
            except Exception:
                result.put((False, None))
            finally:
                self._dns_slots.release()

        threading.Thread(target=resolve, daemon=True, name="knowledge-connector-dns").start()
        try:
            ok, addresses = result.get(timeout=min(timeout, self._remaining(deadline)))
        except queue.Empty:
            raise TimeoutError from None
        if not ok:
            raise OSError("DNS resolution failed")
        return public_addresses(addresses)

    def _authenticate(self, url: str) -> WireRequest:
        try:
            config = validate_config(self.config)
            headers = [("User-Agent", config.http.user_agent), ("Accept-Encoding", "identity")]
            contact = self._validated_contact(config)
            parsed = urlsplit(url)
            params = parse_qsl(parsed.query, keep_blank_values=True)
            reserved = {"tool", "email", "mailto", "api_key", config.api_key_parameter}
            if any(key in reserved for key, _ in params):
                raise ConnectorError(ErrorCategory.SECURITY)
            params.extend(contact._items())
            if config.api_key_env is not None:
                secret = self._environment_value(config.api_key_env, ErrorCategory.AUTHENTICATION)
                if (
                    type(secret) is not str
                    or not secret
                    or len(secret) > 4096
                    or any(ord(c) < 33 or ord(c) > 126 for c in secret)
                ):
                    raise ConnectorError(ErrorCategory.AUTHENTICATION)
                if config.auth_mode == "bearer":
                    headers.append(("Authorization", "Bearer " + secret))
                else:
                    params.append((config.api_key_parameter, secret))
            url = urlunsplit(parsed._replace(query=urlencode(params)))
            return WireRequest(url, tuple(headers), config.http.timeout_ms / 1000)
        except ConnectorError as error:
            raise _RequestPreparationError(error.category) from None
        except Exception:
            raise _RequestPreparationError(ErrorCategory.SECURITY) from None

    def get(
        self,
        url: str,
        *,
        operation: str,
        expected_mime: tuple[str, ...],
        search_request_checksum: str | None = None,
    ) -> TransientResponse:
        config = validate_config(self.config)
        if not config.enabled:
            raise ConnectorError(ErrorCategory.DISABLED)
        start = self.clock.monotonic()
        requested_at = self.clock.now()
        deadline = _RequestWindow(start + config.http.total_timeout_ms / 1000)
        attempts: list[Attempt] = []
        retry_limit = (
            self.governor.retry_limit(config.http.max_retries)
            if self.governor is not None
            else config.http.max_retries
        )
        request_checksum = digest(url.encode())

        def metadata() -> RequestMetadata:
            result = RequestMetadata(
                provider_code=config.provider_code,
                provider_version=config.provider_version,
                adapter=config.adapter,
                config_checksum=config.checksum(),
                request_checksum=request_checksum,
                search_request_checksum=search_request_checksum,
                operation=operation,
                requested_at=requested_at,
                elapsed_ms=max(0, int((self.clock.monotonic() - start) * 1000)),
                attempts=tuple(attempts),
            )
            if self.governor is not None:
                self.governor.completed_request(result)
            return result

        if not self._slots.acquire(timeout=config.http.total_timeout_ms / 1000):
            raise ConnectorError(ErrorCategory.TIMEOUT)
        try:
            for number in range(1, retry_limit + 2):
                response = None
                failure = None
                delay_ms = 0
                try:
                    response = self._send_with_redirects(url, deadline, operation)
                    status = response.status
                    if status in (401, 403):
                        failure = ErrorCategory.AUTHENTICATION
                    elif status == 429:
                        failure = ErrorCategory.RATE_LIMIT
                    elif status >= 500 or status == 408:
                        failure = ErrorCategory.HTTP
                    elif not 200 <= status < 300:
                        failure = ErrorCategory.HTTP
                    else:
                        headers = self._headers(response)
                        mime = headers.get("content-type", "").split(";", 1)[0].strip().lower()
                        if mime not in config.http.allowed_mime_types or mime not in expected_mime:
                            raise ConnectorError(ErrorCategory.MIME)
                        attempts.append(
                            Attempt(
                                number=number,
                                http_status=status,
                                response_checksum=digest(response.body),
                            )
                        )
                        return TransientResponse(
                            metadata=metadata(),
                            mime=mime,
                            response_checksum=digest(response.body),
                            byte_count=len(response.body),
                            body=response.body,
                        )
                except ssl.SSLCertVerificationError:
                    failure = ErrorCategory.SECURITY
                except TimeoutError:
                    failure = ErrorCategory.TIMEOUT
                except (OSError, http.client.HTTPException):
                    failure = ErrorCategory.NETWORK
                except _RequestPreparationError:
                    raise
                except ConnectorError as error:
                    failure = error.category
                except (ValueError, TypeError, ValidationError, UnicodeError):
                    failure = ErrorCategory.SECURITY
                retryable = failure in (
                    ErrorCategory.TIMEOUT,
                    ErrorCategory.NETWORK,
                    ErrorCategory.RATE_LIMIT,
                )
                if response is not None and (response.status >= 500 or response.status == 408):
                    retryable = True
                if retryable and number <= retry_limit:
                    delay_ms = min(
                        config.http.backoff_ms * 2 ** (number - 1), config.http.max_retry_after_ms
                    )
                    if response is not None:
                        retry_after = self._headers(response).get("retry-after", "")
                        if retry_after.isdecimal():
                            delay_ms = max(
                                delay_ms,
                                min(int(retry_after) * 1000, config.http.max_retry_after_ms),
                            )
                attempts.append(
                    Attempt(
                        number=number,
                        http_status=response.status if response else None,
                        error=failure,
                        response_checksum=digest(response.body) if response else None,
                        delay_ms=delay_ms,
                    )
                )
                if not retryable or number > retry_limit:
                    category = ErrorCategory.RETRY_EXHAUSTED if retryable else failure
                    raise ConnectorError(
                        category, metadata=metadata(), last_error=failure
                    ) from None
                try:
                    self._wait(delay_ms / 1000, deadline)
                except ConnectorError:
                    raise ConnectorError(
                        ErrorCategory.TIMEOUT, metadata=metadata(), last_error=failure
                    ) from None
            raise AssertionError("bounded retry loop must return or raise")
        finally:
            self._slots.release()

    def _headers(self, response: WireResponse) -> dict[str, str]:
        result: dict[str, str] = {}
        if type(response) is not WireResponse or type(response.status) is not int:
            raise ConnectorError(ErrorCategory.PARSE)
        if not 100 <= response.status <= 599 or type(response.body) is not bytes:
            raise ConnectorError(ErrorCategory.PARSE)
        if len(response.headers) > 100:
            raise ConnectorError(ErrorCategory.RESPONSE_SIZE)
        for name, value in response.headers:
            if not isinstance(name, str) or not isinstance(value, str) or len(value) > 8192:
                raise ConnectorError(ErrorCategory.PARSE)
            lowered = name.lower()
            if lowered in result and lowered in {
                "content-type",
                "content-length",
                "location",
                "content-encoding",
                "retry-after",
            }:
                raise ConnectorError(ErrorCategory.PARSE)
            result[lowered] = value
        return result

    def _send_wire(
        self, url: str, deadline: float | _RequestWindow, operation: str
    ) -> WireResponse:
        if self.governor is not None:
            prepare = getattr(self.governor, "prepare_send", None)
            if prepare is not None:
                started = self.clock.monotonic()
                prepare(self.config.provider_code)
                if isinstance(deadline, _RequestWindow):
                    deadline.expires_at += max(0.0, self.clock.monotonic() - started)
            self.governor.before_send(self.config.provider_code)
        requested_at = self.clock.now()
        response = None
        try:
            request = self._authenticate(url)
            self._rate_limit(deadline)
            addresses = self._resolve(urlsplit(url).hostname, deadline)
            request = WireRequest(
                request.url,
                request.headers,
                min(request.timeout_seconds, self._remaining(deadline)),
            )
            response = self.transport.send(
                request, addresses[0], self.config.http.max_response_bytes
            )
            return response
        finally:
            if self.governor is not None:
                valid = type(response) is WireResponse and type(response.body) is bytes
                receipt_started = self.clock.monotonic()
                try:
                    self.governor.completed_wire(
                        WireReceipt(
                            provider_code=self.config.provider_code,
                            provider_version=self.config.provider_version,
                            adapter=self.config.adapter,
                            config_checksum=self.config.checksum(),
                            operation=operation,
                            request_checksum=digest(url.encode()),
                            response_checksum=digest(response.body) if valid else None,
                            http_status=response.status if valid else None,
                            requested_at=requested_at,
                        )
                    )
                finally:
                    # Persisted outcome and current-owner checks are local work.
                    # Absolute governor deadlines and transport time still apply.
                    if isinstance(deadline, _RequestWindow):
                        deadline.expires_at += max(0.0, self.clock.monotonic() - receipt_started)

    def _send_with_redirects(
        self, url: str, deadline: float | _RequestWindow, operation: str
    ) -> WireResponse:
        origin = urlsplit(https_url(url, self.config.allowed_hosts)).hostname
        for hop in range(self.config.http.max_redirects + 1):
            https_url(url, self.config.allowed_hosts)
            parsed = urlsplit(url)
            if parsed.hostname != origin:
                raise ConnectorError(ErrorCategory.SECURITY)
            response = self._send_wire(url, deadline, operation)
            headers = self._headers(response)
            if self.governor is not None:
                limit_names = {
                    "retry-after",
                    "x-rate-limit-limit",
                    "x-rate-limit-interval",
                    "x-rate-limit-remaining",
                    "x-ratelimit-limit",
                    "x-ratelimit-remaining",
                    "x-ratelimit-reset",
                    "x-concurrency-limit",
                }
                self.governor.after_send(
                    self.config.provider_code,
                    response.status,
                    tuple((name, value) for name, value in headers.items() if name in limit_names),
                )
            self._remaining(deadline)
            if len(response.body) > self.config.http.max_response_bytes:
                raise ConnectorError(ErrorCategory.RESPONSE_SIZE)
            if headers.get("content-encoding", "identity").lower() != "identity":
                raise ConnectorError(ErrorCategory.RESPONSE_SIZE)
            if "content-length" in headers:
                length = headers["content-length"]
                if not length.isdecimal() or int(length) != len(response.body):
                    raise ConnectorError(ErrorCategory.RESPONSE_SIZE)
            if response.status not in (301, 302, 303, 307, 308):
                return response
            if hop == self.config.http.max_redirects or "location" not in headers:
                raise ConnectorError(ErrorCategory.SECURITY)
            url = urljoin(url, headers["location"])
        raise AssertionError("redirect loop is bounded")
