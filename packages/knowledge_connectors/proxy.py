"""Explicit HTTP CONNECT egress with a declared destination-resolution policy.

The proxy is a trusted deployment endpoint, not a source URL. No automatic proxy
discovery, cookies, redirects, credential forwarding or direct-route fallback.
"""

import http.client
import json
import os
import re
import ssl
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field

from .configuration import https_url
from .http import PinnedHTTPSTransport, public_addresses
from .ports import ConnectorError, ErrorCategory, StrictModel, digest


class ProxyPolicy(StrictModel):
    """Only environment variable names, never proxy addresses or credentials."""

    proxy_environment: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,127}$")
    target_resolution: Literal["local_public_ip_pinned", "proxy_hostname"] = (
        "local_public_ip_pinned"
    )

    def receipt(self):
        return {
            "transport": "http-connect-public-ip/1.0.0"
            if self.target_resolution == "local_public_ip_pinned"
            else "http-connect-hostname/1.0.0",
            "proxy_environment": self.proxy_environment,
            "target_resolution": self.target_resolution,
            "tls_verification": True,
            "direct_fallback": False,
        }

    def checksum(self):
        # Algorithm version is part of the digest, independently of old ProviderConfig.
        return digest(json.dumps(self.receipt(), sort_keys=True, separators=(",", ":")).encode())


class _TunnelConnection(http.client.HTTPSConnection):
    def __init__(self, host, ip, timeout, proxy_host, proxy_port):
        super().__init__(host, 443, timeout=timeout, context=ssl.create_default_context())
        self._target_ip = ip
        self._proxy_host = proxy_host
        self._proxy_port = proxy_port

    def _destination(self):
        return f"[{self._target_ip}]" if ":" in self._target_ip else self._target_ip

    def connect(self):
        public_addresses((self._target_ip,))
        # Both modes screen local DNS; hostname mode delegates the final address
        # to the explicitly trusted proxy and does not claim remote IP pinning.
        tunnel = http.client.HTTPConnection(
            self._proxy_host, self._proxy_port, timeout=self.timeout
        )
        tunnel.set_tunnel(self._destination(), 443)
        try:
            tunnel.connect()
            self.sock = self._context.wrap_socket(tunnel.sock, server_hostname=self.host)
            tunnel.sock = None
        finally:
            tunnel.close()


class _HostnameTunnelConnection(_TunnelConnection):
    def _destination(self):
        return self.host


class ExplicitProxyHTTPSTransport(PinnedHTTPSTransport):
    """Reuse bounded response reading from the direct transport."""

    def __init__(self, policy: ProxyPolicy, *, environment=os.environ.get):
        self.policy = ProxyPolicy.model_validate(policy)
        self.environment = environment

    def _connection(self, host, ip, timeout):
        https_url("https://" + host + "/")
        public_addresses((ip,))
        try:
            value = self.environment(self.policy.proxy_environment)
            if (
                type(value) is not str
                or not value
                or len(value) > 4096
                or any(ord(c) <= 32 or ord(c) >= 127 for c in value)
                or "\\" in value
            ):
                raise ValueError
            proxy = urlsplit(value)
            if (
                proxy.scheme != "http"
                or not proxy.hostname
                or not re.fullmatch(r"[A-Za-z0-9.:-]+", proxy.hostname)
                or proxy.username is not None
                or proxy.password is not None
                or proxy.path not in ("", "/")
                or proxy.query
                or proxy.fragment
                or proxy.port == 0
            ):
                raise ValueError
            connection = (
                _TunnelConnection
                if self.policy.target_resolution == "local_public_ip_pinned"
                else _HostnameTunnelConnection
            )
            return connection(host, ip, timeout, proxy.hostname, proxy.port or 80)
        except Exception:
            raise ConnectorError(ErrorCategory.CONFIGURATION) from None

    def send(self, request, ip, max_bytes):
        try:
            return super().send(request, ip, max_bytes)
        except ssl.SSLCertVerificationError:
            raise ConnectorError(ErrorCategory.SECURITY) from None
        except TimeoutError:
            raise ConnectorError(ErrorCategory.TIMEOUT) from None
        except (OSError, http.client.HTTPException):
            # Proxy errors may contain a proxy hostname; do not leak them to callers.
            raise ConnectorError(ErrorCategory.NETWORK) from None
