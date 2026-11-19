"""Strict transient connector ports; these are not Knowledge persistence records."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Protocol, Self, runtime_checkable

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from packages.contracts import validate_stable_code

Text = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=2048)]
ShortText = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=256)]
Code = Annotated[ShortText, AfterValidator(lambda value: str(validate_stable_code(value)))]
Version = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"),
]
Checksum = Annotated[str, StringConstraints(strict=True, pattern=r"^sha256:[0-9a-f]{64}$")]
Adapter = Literal["pubmed", "europe_pmc", "crossref", "guideline", "web_fallback"]
SourceType = Literal["GUIDELINE", "ARTICLE", "CONSENSUS", "WEB"]


class StrictModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        hide_input_in_errors=True,
    )


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def model_checksum(model: BaseModel) -> str:
    return digest(
        json.dumps(
            model.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    )


def checked[T: StrictModel](kind: type[T], value: T) -> T:
    if type(value) is not kind:
        raise TypeError("exact connector port type required")
    return kind.model_validate(value)


class Identity(StrictModel):
    provider_code: Code
    provider_version: Version


class ConfiguredIdentity(Identity):
    adapter: Adapter
    config_checksum: Checksum

    def scope(self) -> tuple[str, str, str, str]:
        return self.provider_code, self.provider_version, self.adapter, self.config_checksum


class PageCursor(ConfiguredIdentity):
    query_checksum: Checksum
    token: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=4096)]
    page_index: int = Field(ge=1, le=10000)


class StructuredQuery(StrictModel):
    text: Text
    page_size: int = Field(default=20, ge=1, le=100)
    cursor: PageCursor | None = None

    def checksum(self) -> str:
        return digest(json.dumps([self.text, self.page_size], ensure_ascii=False).encode())


class SourceLocator(StrictModel):
    record_id: Text | None = None
    url: Text | None = None

    @model_validator(mode="after")
    def has_location(self) -> Self:
        if (self.record_id is None) == (self.url is None):
            raise ValueError("provide exactly one record_id or URL")
        return self


class Identifier(StrictModel):
    identifier_type: Literal["DOI", "PMID", "PMCID", "URL"]
    value: Text


class DiscoveryRecord(ConfiguredIdentity):
    record_id: Text
    source_type: SourceType
    title: Text | None = None
    url: Text
    identifiers: tuple[Identifier, ...] = Field(default=(), max_length=16)
    authors: tuple[ShortText, ...] = Field(default=(), max_length=100)
    date: ShortText | None = None
    snippet: Annotated[str, StringConstraints(strict=True, max_length=16384)] | None = None
    rank: int = Field(ge=1)
    # This is an admission ceiling, not an alternative lifecycle state machine.
    admission_status: Literal["draft"] = "draft"


class ErrorCategory(StrEnum):
    DISABLED = "disabled"
    CONFIGURATION = "configuration"
    UNSUPPORTED = "unsupported"
    SECURITY = "security"
    TIMEOUT = "timeout"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    HTTP = "http"
    PARSE = "parse"
    RESPONSE_SIZE = "response_size"
    MIME = "mime"
    RETRY_EXHAUSTED = "retry_exhausted"
    PAGE_LIMIT = "page_limit"


class Attempt(StrictModel):
    number: int = Field(ge=1)
    http_status: int | None = Field(default=None, ge=100, le=599)
    error: ErrorCategory | None = None
    response_checksum: Checksum | None = None
    delay_ms: int = Field(default=0, ge=0)


class RequestMetadata(ConfiguredIdentity):
    request_checksum: Checksum
    search_request_checksum: Checksum | None = None
    operation: Literal["search", "summary", "fetch", "healthcheck"]
    requested_at: datetime
    elapsed_ms: int = Field(ge=0)
    attempts: tuple[Attempt, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def aware_time(self) -> Self:
        if (self.operation == "summary") != (self.search_request_checksum is not None):
            raise ValueError("only summary must reference its search request checksum")
        if self.requested_at.tzinfo is None or self.requested_at.utcoffset() is None:
            raise ValueError("request time must have a timezone")
        return self


class WireReceipt(ConfiguredIdentity):
    """One admitted wire attempt, including redirects; contains no outbound URL."""

    operation: Literal["search", "summary", "fetch", "healthcheck"]
    request_checksum: Checksum
    response_checksum: Checksum | None
    http_status: int | None = Field(ge=100, le=599)
    requested_at: datetime


class TransientResponse(StrictModel):
    metadata: RequestMetadata
    mime: ShortText
    response_checksum: Checksum
    byte_count: int = Field(ge=0, le=4 * 1024 * 1024)
    body: bytes = Field(repr=False, exclude=True, max_length=4 * 1024 * 1024)

    @model_validator(mode="after")
    def bytes_match(self) -> Self:
        if len(self.body) != self.byte_count or digest(self.body) != self.response_checksum:
            raise ValueError("transient response byte count/checksum mismatch")
        return self


class SearchPage(ConfiguredIdentity):
    retrieved_at: datetime
    records: tuple[DiscoveryRecord, ...] = Field(max_length=100)
    next_cursor: PageCursor | None = None
    responses: tuple[TransientResponse, ...] = Field(default=(), max_length=2)
    query_checksum: Checksum
    page_checksum: Checksum
    result: Literal["results", "empty"]
    admission_status: Literal["draft"] = "draft"

    @model_validator(mode="after")
    def consistent_page(self) -> Self:
        if self.retrieved_at.tzinfo is None or self.retrieved_at.utcoffset() is None:
            raise ValueError("retrieval time must have a timezone")
        if (self.result == "empty") != (not self.records):
            raise ValueError("empty is a search result, not a medical negative")
        if any(r.scope() != self.scope() for r in self.records):
            raise ValueError("page contains a different provider/version")
        expected = digest(
            json.dumps(
                [r.model_dump(mode="json") for r in self.records],
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        )
        if self.page_checksum != expected:
            raise ValueError("discovery page checksum mismatch")
        if any(r.metadata.scope() != self.scope() for r in self.responses):
            raise ValueError("page response belongs to a different provider/version")
        operations = tuple(r.metadata.operation for r in self.responses)
        expected = (
            ()
            if self.adapter == "guideline"
            else ("search", "summary")
            if self.adapter == "pubmed" and self.records
            else ("search",)
        )
        if operations != expected:
            raise ValueError("search response operation sequence mismatch")
        if len(self.responses) == 2 and (
            self.responses[1].metadata.search_request_checksum
            != self.responses[0].metadata.request_checksum
        ):
            raise ValueError("summary does not reference this search request")
        if self.next_cursor is not None and (
            self.next_cursor.scope() != self.scope()
            or self.next_cursor.query_checksum != self.query_checksum
        ):
            raise ValueError("page cursor binding mismatch")
        return self


class FetchedDocument(ConfiguredIdentity):
    locator: SourceLocator
    response: TransientResponse
    admission_status: Literal["draft"] = "draft"

    @model_validator(mode="after")
    def response_scope(self) -> Self:
        if (
            self.response.metadata.scope() != self.scope()
            or self.response.metadata.operation != "fetch"
        ):
            raise ValueError("fetch response scope/operation mismatch")
        return self


class Capabilities(ConfiguredIdentity):
    search: bool
    fetch: bool
    single_page: bool = True
    metadata_only: bool
    discovery_mode: Literal["api", "curated_catalog", "disabled"]


class HealthResult(ConfiguredIdentity):
    available: bool
    category: ErrorCategory | None = None
    response: TransientResponse | None = None

    @model_validator(mode="after")
    def consistent_health(self) -> Self:
        if self.available:
            if self.category is not None or self.response is None:
                raise ValueError("successful health requires only a response")
            if (
                self.response.metadata.scope() != self.scope()
                or self.response.metadata.operation != "healthcheck"
            ):
                raise ValueError("health response scope/operation mismatch")
        elif self.category is None or self.response is not None:
            raise ValueError("failed health requires only an error category")
        return self


class ConnectorError(Exception):
    """Only fixed error categories and safe provenance, never URLs/keys/provider bodies."""

    def __init__(
        self,
        category: ErrorCategory,
        *,
        metadata: RequestMetadata | None = None,
        last_error: ErrorCategory | None = None,
    ):
        super().__init__(category.value)
        self.category = category
        self.metadata = metadata
        self.last_error = last_error


@runtime_checkable
class KnowledgeSourceConnector(Protocol):
    def search(self, query: StructuredQuery) -> SearchPage: ...
    def fetch(self, locator: SourceLocator) -> FetchedDocument: ...
    def healthcheck(self) -> HealthResult: ...
    def capabilities(self) -> Capabilities: ...
