"""Versioned, secret-free connector configuration; TOML uses the standard library."""

from __future__ import annotations

import ipaddress
import json
import re
import tomllib
from pathlib import Path
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, StringConstraints, model_validator

from .ports import (
    Adapter,
    Identity,
    ShortText,
    SourceType,
    StrictModel,
    Text,
    Version,
    checked,
    model_checksum,
)


def https_url(value: str, allowed_hosts: tuple[str, ...] | None = None) -> str:
    if not isinstance(value, str) or len(value) > 8192 or not value.isascii():
        raise ValueError("invalid URL")
    if any(ord(c) < 33 or ord(c) == 127 for c in value) or "\\" in value:
        raise ValueError("URL contains unsafe characters")
    parsed = urlsplit(value)
    host = parsed.hostname
    if (
        parsed.scheme != "https"
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or parsed.port not in (None, 443)
    ):
        raise ValueError("only credential-free HTTPS on port 443 is allowed")
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", host) or ".." in host:
        raise ValueError("canonical ASCII hostname required")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValueError("IP-literal URLs are not allowed")
    if "." not in host or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("public DNS hostname required")
    if allowed_hosts is not None and host not in allowed_hosts:
        raise ValueError("hostname is outside the exact allowlist")
    return value


def validate_contact_email(value: str) -> str:
    if (
        type(value) is not str
        or len(value) > 256
        or not re.fullmatch(
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?\.[A-Za-z]{2,63}", value
        )
    ):
        raise ValueError("invalid contact email")
    local, domain = value.rsplit("@", 1)
    if (
        len(local) > 64
        or local.startswith(".")
        or local.endswith(".")
        or ".." in local
        or any(
            not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label)
            for label in domain.split(".")
        )
    ):
        raise ValueError("invalid contact email")
    return value


class HTTPPolicy(StrictModel):
    timeout_ms: int = Field(default=10000, ge=1, le=120000)
    total_timeout_ms: int = Field(default=30000, ge=1, le=300000)
    max_retries: int = Field(default=2, ge=0, le=7)
    backoff_ms: int = Field(default=250, ge=0, le=30000)
    max_retry_after_ms: int = Field(default=10000, ge=0, le=60000)
    concurrency: int = Field(default=1, ge=1, le=16)
    rate_limit_per_minute: int = Field(default=120, ge=1, le=6000)
    max_response_bytes: int = Field(default=1048576, ge=1, le=4 * 1024 * 1024)
    max_redirects: int = Field(default=2, ge=0, le=5)
    max_page_size: int = Field(default=50, ge=1, le=100)
    max_pages: int = Field(default=100, ge=1, le=10000)
    user_agent: ShortText
    allowed_mime_types: tuple[ShortText, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def safe_headers(self) -> Self:
        if any(ord(c) < 32 or ord(c) > 126 for c in self.user_agent):
            raise ValueError("invalid User-Agent")
        if self.total_timeout_ms < self.timeout_ms:
            raise ValueError("total timeout must cover one request timeout")
        if any(not re.fullmatch(r"[a-z0-9.+-]+/[a-z0-9.+-]+", m) for m in self.allowed_mime_types):
            raise ValueError("exact MIME types required")
        return self


class CatalogEntry(StrictModel):
    record_id: ShortText
    title: Text
    url: Text


class ProviderConfig(Identity):
    adapter: Adapter
    name: ShortText
    enabled: bool
    source_type: SourceType
    homepage_url: Text
    allowlist_version: Version
    additional_institution: bool = True
    allowed_hosts: tuple[ShortText, ...] = Field(min_length=1, max_length=64)
    search_endpoint: Text
    fetch_endpoint: Text
    health_endpoint: Text
    record_url_prefix: Text
    summary_endpoint: Text | None = None
    http: HTTPPolicy
    api_key_env: Annotated[str, StringConstraints(pattern=r"^[A-Z][A-Z0-9_]{0,127}$")] | None = None
    auth_mode: Literal["none", "query", "bearer"] = "none"
    api_key_parameter: ShortText = "api_key"
    pubmed_tool: (
        Annotated[
            str,
            StringConstraints(
                strict=True, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"
            ),
        ]
        | None
    ) = None
    identity_policy: Literal["contact-env/2.0.0"] = "contact-env/2.0.0"
    contact_email_env: (
        Annotated[str, StringConstraints(strict=True, pattern=r"^[A-Z][A-Z0-9_]{0,127}$")] | None
    ) = None
    catalog: tuple[CatalogEntry, ...] = Field(default=(), max_length=500)

    @model_validator(mode="after")
    def endpoints_and_scope(self) -> Self:
        if len(set(self.allowed_hosts)) != len(self.allowed_hosts):
            raise ValueError("duplicate allowlist host")
        for host in self.allowed_hosts:
            if urlsplit(https_url("https://" + host + "/")).hostname != host:
                raise ValueError("allowlist entries must be exact lowercase DNS hosts")
        for value in (
            self.homepage_url,
            self.search_endpoint,
            self.fetch_endpoint,
            self.health_endpoint,
            self.record_url_prefix,
            self.summary_endpoint,
        ):
            if value is not None:
                https_url(value, self.allowed_hosts)
                if urlsplit(value).query:
                    raise ValueError("endpoints must not embed query parameters or secrets")
        if (self.auth_mode == "none") != (self.api_key_env is None):
            raise ValueError("authentication requires an environment variable name")
        if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,63}", self.api_key_parameter):
            raise ValueError("invalid credential parameter name")
        if self.api_key_parameter in ("tool", "email", "mailto"):
            raise ValueError("credential parameter conflicts with contact identity")
        if self.adapter == "pubmed":
            if self.summary_endpoint is None or self.pubmed_tool is None:
                raise ValueError("PubMed requires summary endpoint and versioned tool identity")
            if self.auth_mode not in ("none", "query") or self.api_key_parameter != "api_key":
                raise ValueError("PubMed supports only the E-utilities api_key parameter")
            maximum = 180 if self.api_key_env is None else 600
            if self.http.rate_limit_per_minute > maximum:
                raise ValueError("PubMed rate exceeds the configured API-key tier")
        elif self.pubmed_tool is not None:
            raise ValueError("tool identity belongs only to PubMed")
        expected = {
            "pubmed": "ARTICLE",
            "europe_pmc": "ARTICLE",
            "crossref": "ARTICLE",
            "guideline": "GUIDELINE",
            "web_fallback": "WEB",
        }
        if self.source_type != expected[self.adapter]:
            raise ValueError("adapter/source_type mismatch")
        if self.catalog and self.adapter not in {"guideline", "web_fallback"}:
            raise ValueError("only guideline/web connectors own curated discovery catalogs")
        if len({entry.record_id for entry in self.catalog}) != len(self.catalog):
            raise ValueError("duplicate catalog record ID")
        for entry in self.catalog:
            https_url(entry.url, self.allowed_hosts)
        return self

    def checksum(self) -> str:
        return model_checksum(self)


class ConnectorConfiguration(StrictModel):
    config_version: Version
    providers: tuple[ProviderConfig, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def unique_providers(self) -> Self:
        if len({p.provider_code for p in self.providers}) != len(self.providers):
            raise ValueError("registry contains duplicate provider code")
        if sum(p.adapter == "web_fallback" for p in self.providers) > 1:
            raise ValueError("only one replaceable web fallback is permitted")
        return self


def load_configuration(path: Path) -> ConnectorConfiguration:
    if path.stat().st_size > 256 * 1024:
        raise ValueError("connector configuration exceeds size limit")
    with path.open("rb") as stream:
        content = tomllib.load(stream)
    return ConnectorConfiguration.model_validate_json(json.dumps(content, allow_nan=False))


def validate_config(config: ProviderConfig) -> ProviderConfig:
    return checked(ProviderConfig, config)
