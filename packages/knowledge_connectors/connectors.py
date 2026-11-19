"""Single-page provider adapters; no search orchestration or durable source archive."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import quote, urlencode

from pydantic import ValidationError

from .configuration import ProviderConfig, https_url, validate_config
from .http import SafeHTTPClient, _RequestPreparationError
from .ports import (
    Capabilities,
    ConnectorError,
    DiscoveryRecord,
    ErrorCategory,
    FetchedDocument,
    HealthResult,
    Identifier,
    PageCursor,
    SearchPage,
    SourceLocator,
    StructuredQuery,
    TransientResponse,
    checked,
    digest,
)


def _object(value: Any) -> dict:
    if type(value) is not dict:
        raise ValueError("provider object required")
    return value


def _array(value: Any) -> list:
    if type(value) is not list:
        raise ValueError("provider array required")
    return value


def _integer(value: Any, *, decimal_string: bool = False) -> int:
    if decimal_string and type(value) is str and re.fullmatch(r"0|[1-9][0-9]*", value):
        value = int(value)
    if type(value) is not int or value < 0:
        raise ValueError("provider nonnegative integer required")
    return value


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON number")


def _json(response: TransientResponse) -> dict:
    return _object(
        json.loads(response.body, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
    )


class ProviderConnector:
    adapter: str

    def __init__(self, config: ProviderConfig, client: SafeHTTPClient):
        self.config = validate_config(config)
        if self.config.adapter != self.adapter or client.config != self.config:
            raise ValueError("connector, configuration and HTTP client must match")
        self.client = client

    def _identity(self) -> dict:
        return {
            "provider_code": self.config.provider_code,
            "provider_version": self.config.provider_version,
            "adapter": self.config.adapter,
            "config_checksum": self.config.checksum(),
        }

    def _enabled(self) -> None:
        validate_config(self.config)
        if not self.config.enabled:
            raise ConnectorError(ErrorCategory.DISABLED)

    def capabilities(self) -> Capabilities:
        validate_config(self.config)
        return Capabilities(
            **self._identity(),
            search=self.config.enabled,
            fetch=self.config.enabled,
            metadata_only=self.adapter == "crossref",
            discovery_mode=(
                "disabled"
                if not self.config.enabled
                else "curated_catalog"
                if self.adapter == "guideline" or self.config.catalog
                else "api"
            ),
        )

    def _url(self, endpoint: str, parameters: dict[str, str | int]) -> str:
        return endpoint + ("?" + urlencode(parameters) if parameters else "")

    def _request(
        self,
        endpoint: str,
        parameters: dict,
        operation: str = "search",
        *,
        search_request_checksum: str | None = None,
    ) -> TransientResponse:
        return self.client.get(
            self._url(endpoint, parameters),
            operation=operation,
            expected_mime=("application/json",),
            search_request_checksum=search_request_checksum,
        )

    def _record(self, *, record_id: str, rank: int, url: str, **values: Any) -> DiscoveryRecord:
        https_url(url)  # Discovery URLs are not fetched here; fetch applies the exact allowlist.
        return DiscoveryRecord(
            **self._identity(),
            record_id=record_id,
            rank=rank,
            url=url,
            source_type=self.config.source_type,
            **values,
        )

    def _cursor(self, query: StructuredQuery) -> tuple[str | None, int]:
        if query.page_size > self.config.http.max_page_size:
            raise ConnectorError(ErrorCategory.PAGE_LIMIT)
        cursor = query.cursor
        if cursor is None:
            return None, 0
        if (
            (cursor.provider_code, cursor.provider_version)
            != (self.config.provider_code, self.config.provider_version)
            or cursor.adapter != self.adapter
            or cursor.config_checksum != self.config.checksum()
            or cursor.query_checksum != query.checksum()
        ):
            raise ValueError("cursor provider/configuration/query mismatch")
        if cursor.page_index >= self.config.http.max_pages:
            raise ConnectorError(ErrorCategory.PAGE_LIMIT)
        return cursor.token, cursor.page_index

    def search(self, query: StructuredQuery) -> SearchPage:
        self._enabled()
        query = checked(StructuredQuery, query)
        token, page_index = self._cursor(query)
        responses: list[TransientResponse] = []
        try:
            records, next_token = self._search(query, token, responses)
            if len(records) > query.page_size:
                raise ValueError("provider exceeded requested page size")
            if len({r.record_id for r in records}) != len(records):
                raise ValueError("provider returned duplicate IDs within one page")
            next_cursor = None
            if next_token and records:
                next_cursor = PageCursor(
                    **self._identity(),
                    query_checksum=query.checksum(),
                    token=next_token,
                    page_index=page_index + 1,
                )
            page_checksum = digest(
                json.dumps(
                    [r.model_dump(mode="json") for r in records], sort_keys=True, ensure_ascii=False
                ).encode()
            )
            return SearchPage(
                **self._identity(),
                retrieved_at=self.client.clock.now(),
                records=tuple(records),
                next_cursor=next_cursor,
                responses=tuple(responses),
                query_checksum=query.checksum(),
                page_checksum=page_checksum,
                result="results" if records else "empty",
            )
        except ConnectorError:
            raise
        except (ValueError, TypeError, KeyError, IndexError, RecursionError, ValidationError):
            raise ConnectorError(
                ErrorCategory.PARSE, metadata=responses[-1].metadata if responses else None
            ) from None

    def _search(
        self, query: StructuredQuery, token: str | None, responses: list[TransientResponse]
    ) -> tuple[list[DiscoveryRecord], str | None]:
        raise NotImplementedError

    def fetch(self, locator: SourceLocator) -> FetchedDocument:
        self._enabled()
        locator = checked(SourceLocator, locator)
        url, expected_mime = self._fetch_url(locator)
        response = self.client.get(url, operation="fetch", expected_mime=expected_mime)
        return FetchedDocument(**self._identity(), locator=locator, response=response)

    def _fetch_url(self, locator: SourceLocator) -> tuple[str, tuple[str, ...]]:
        raise NotImplementedError

    def healthcheck(self) -> HealthResult:
        if not self.config.enabled:
            return HealthResult(
                **self._identity(), available=False, category=ErrorCategory.DISABLED
            )
        try:
            response = self.client.get(
                self.config.health_endpoint,
                operation="healthcheck",
                expected_mime=self.config.http.allowed_mime_types,
            )
            return HealthResult(**self._identity(), available=True, response=response)
        except _RequestPreparationError:
            raise
        except ConnectorError as error:
            return HealthResult(**self._identity(), available=False, category=error.category)


class PubMedConnector(ProviderConnector):
    adapter = "pubmed"

    def _search(self, query, token, responses):
        offset = _integer(token or "0", decimal_string=True)
        if offset + query.page_size > 10000:
            raise ConnectorError(ErrorCategory.PAGE_LIMIT)
        response = self._request(
            self.config.search_endpoint,
            {
                "db": "pubmed",
                "term": query.text,
                "retmode": "json",
                "retstart": offset,
                "retmax": query.page_size,
            },
        )
        responses.append(response)
        result = _object(_json(response)["esearchresult"])
        count = _integer(result["count"], decimal_string=True)
        ids = _array(result["idlist"])
        if len(ids) > query.page_size or any(
            type(i) is not str or not re.fullmatch(r"[1-9][0-9]*", i) for i in ids
        ):
            raise ValueError("invalid PubMed IDs")
        if not ids:
            return [], None
        summary = self._request(
            self.config.summary_endpoint,
            {"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            "summary",
            search_request_checksum=response.metadata.request_checksum,
        )
        responses.append(summary)
        documents = _object(_json(summary)["result"])
        if _array(documents["uids"]) != ids:
            raise ValueError("PubMed summary is not bound to requested IDs")
        records = []
        for index, uid in enumerate(ids):
            item = _object(documents[uid])
            if item.get("uid") != uid or "error" in item:
                raise ValueError("PubMed summary UID mismatch or error")
            identifiers = [Identifier(identifier_type="PMID", value=uid)]
            for identifier in _array(item.get("articleids", [])):
                kind = _object(identifier).get("idtype")
                if kind in ("doi", "pmc"):
                    identifiers.append(
                        Identifier(
                            identifier_type="DOI" if kind == "doi" else "PMCID",
                            value=identifier["value"],
                        )
                    )
            records.append(
                self._record(
                    record_id=uid,
                    rank=index + 1,
                    url=self.config.record_url_prefix + quote(uid, safe=""),
                    title=item.get("title"),
                    date=item.get("pubdate"),
                    identifiers=tuple(identifiers),
                    authors=tuple(_object(a)["name"] for a in _array(item.get("authors", []))),
                )
            )
        next_offset = offset + len(ids)
        return records, str(next_offset) if next_offset < min(count, 10000) else None

    def _fetch_url(self, locator):
        if locator.record_id is None or not re.fullmatch(r"[1-9][0-9]*", locator.record_id):
            raise ValueError("PubMed fetch requires a single PMID")
        return self._url(
            self.config.fetch_endpoint, {"db": "pubmed", "id": locator.record_id, "retmode": "xml"}
        ), ("application/xml", "text/xml")


class EuropePMCConnector(ProviderConnector):
    adapter = "europe_pmc"

    def _search(self, query, token, responses):
        response = self._request(
            self.config.search_endpoint,
            {
                "query": query.text,
                "format": "json",
                "resultType": "core",
                "pageSize": query.page_size,
                "cursorMark": token or "*",
            },
        )
        responses.append(response)
        result = _json(response)
        _integer(result["hitCount"])
        items = _array(_object(result["resultList"])["result"])
        records = []
        for index, raw in enumerate(items):
            item = _object(raw)
            source, identifier = item["source"], item["id"]
            if type(source) is not str or not re.fullmatch(r"[A-Z]{2,8}", source):
                raise ValueError("invalid Europe PMC source")
            if type(identifier) is not str or not re.fullmatch(r"[A-Za-z0-9._-]+", identifier):
                raise ValueError("invalid Europe PMC record ID")
            identifiers = tuple(
                Identifier(identifier_type=kind, value=item[key])
                for key, kind in (("pmid", "PMID"), ("pmcid", "PMCID"), ("doi", "DOI"))
                if key in item
            )
            records.append(
                self._record(
                    record_id=source + ":" + identifier,
                    rank=index + 1,
                    url=self.config.record_url_prefix + quote(source) + "/" + quote(identifier),
                    title=item.get("title"),
                    date=item.get("firstPublicationDate"),
                    identifiers=identifiers,
                    snippet=item.get("abstractText"),
                    authors=tuple(
                        _object(a)["fullName"]
                        for a in _array(_object(item.get("authorList", {})).get("author", []))
                    ),
                )
            )
        next_token = result.get("nextCursorMark")
        if next_token is not None and type(next_token) is not str:
            raise ValueError("invalid Europe PMC cursor")
        return records, next_token if len(records) == query.page_size else None

    def _fetch_url(self, locator):
        if locator.record_id is None or not re.fullmatch(r"PMC[1-9][0-9]*", locator.record_id):
            raise ValueError("Europe PMC full-text fetch requires an explicit PMCID")
        return (
            self.config.fetch_endpoint.rstrip("/") + "/" + locator.record_id + "/fullTextXML",
            ("application/xml", "text/xml"),
        )


class CrossrefResolver(ProviderConnector):
    """Bibliographic lookup with explicitly versioned remote ranking.

    Provider 2.1.0 pins score-desc ordering on every cursor page. Historical
    versions retain their original query parameters and replay identity; their
    stored results are not reinterpreted using today's API default ranking.
    The provider version participates in configuration/cursor/request binding.
    """

    adapter = "crossref"

    @staticmethod
    def _author_name(raw):
        author = _object(raw)
        for field in ("name", "given", "family"):
            if author.get(field) is not None and type(author[field]) is not str:
                raise ValueError("invalid Crossref author")
        # Crossref also represents institutional contributors using `name`.
        # Do not fabricate an empty personal name for that valid API shape.
        value = author.get("name")
        if value is None:
            value = " ".join(
                v for v in (author.get("given"), author.get("family")) if v is not None
            )
        if not value.strip():
            raise ValueError("Crossref author name required")
        return value

    def _search(self, query, token, responses):
        parameters = {
            "query.bibliographic": query.text,
            "rows": query.page_size,
            "cursor": token or "*",
        }
        if self.config.provider_version == "2.1.0":
            parameters.update(sort="score", order="desc")
        response = self._request(
            self.config.search_endpoint,
            parameters,
        )
        responses.append(response)
        result = _json(response)
        if result["status"] != "ok":
            raise ValueError("Crossref API reported an error")
        message = _object(result["message"])
        _integer(message["total-results"])
        items = _array(message["items"])
        records = []
        for index, raw in enumerate(items):
            item = _object(raw)
            doi = item["DOI"]
            if type(doi) is not str or not re.fullmatch(r"10\.[0-9]{4,9}/\S+", doi):
                raise ValueError("Crossref DOI required")
            titles = _array(item.get("title", []))
            records.append(
                self._record(
                    record_id=doi,
                    rank=index + 1,
                    url=item.get("URL", self.config.record_url_prefix + quote(doi, safe="/")),
                    title=titles[0] if titles else None,
                    identifiers=(Identifier(identifier_type="DOI", value=doi),),
                    snippet=item.get("abstract"),
                    authors=tuple(self._author_name(a) for a in _array(item.get("author", []))),
                    date=_object(item.get("created", {})).get("date-time"),
                )
            )
        next_token = message.get("next-cursor")
        if next_token is not None and type(next_token) is not str:
            raise ValueError("invalid Crossref cursor")
        return records, next_token if len(records) == query.page_size else None

    def _fetch_url(self, locator):
        if locator.record_id is None or not re.fullmatch(r"10\.[0-9]{4,9}/\S+", locator.record_id):
            raise ValueError("Crossref fetch requires a DOI")
        return (
            self.config.fetch_endpoint.rstrip("/") + "/" + quote(locator.record_id, safe=""),
            ("application/json",),
        )


class GuidelineSiteConnector(ProviderConnector):
    adapter = "guideline"

    def _search(self, query, token, responses):
        offset = _integer(token or "0", decimal_string=True)
        matches = [
            entry
            for entry in self.config.catalog
            if query.text.casefold() in entry.title.casefold()
        ]
        records = [
            self._record(
                record_id=entry.record_id,
                rank=index + 1,
                title=entry.title,
                url=entry.url,
                identifiers=(Identifier(identifier_type="URL", value=entry.url),),
            )
            for index, entry in enumerate(matches[offset : offset + query.page_size])
        ]
        next_offset = offset + len(records)
        return records, str(next_offset) if next_offset < len(matches) else None

    def _fetch_url(self, locator):
        url = locator.url
        if url is None:
            entries = [
                entry for entry in self.config.catalog if entry.record_id == locator.record_id
            ]
            if not entries:
                raise ConnectorError(ErrorCategory.UNSUPPORTED)
            url = entries[0].url
        return url, ("text/html", "application/pdf", "application/xml", "text/xml")


class WebSearchFallbackConnector(ProviderConnector):
    """JSON gateway, or an explicitly configured curated WEB-page catalog.

    Catalog mode is local title-token discovery, not an Internet search engine.
    Its exact entries are bound by the provider configuration checksum.
    """

    adapter = "web_fallback"

    def _search(self, query, token, responses):
        if self.config.catalog:
            import re

            # Preserve WEB discovery's real HTTP receipt contract. The response
            # observes the configured listing endpoint; candidate identities
            # come from the explicit catalog, NOT a claim of remote ranking.
            responses.append(
                self.client.get(
                    self.config.search_endpoint, operation="search", expected_mime=("text/html",)
                )
            )
            terms = set(re.findall(r"[a-z0-9]{3,}", query.text.casefold()))
            offset = _integer(token or "0", decimal_string=True)
            matches = [
                entry
                for entry in self.config.catalog
                if terms.intersection(re.findall(r"[a-z0-9]{3,}", entry.title.casefold()))
            ]
            records = [
                self._record(
                    record_id=entry.record_id,
                    rank=index + 1,
                    title=entry.title,
                    url=entry.url,
                    identifiers=(Identifier(identifier_type="URL", value=entry.url),),
                )
                for index, entry in enumerate(matches[offset : offset + query.page_size])
            ]
            next_offset = offset + len(records)
            return records, str(next_offset) if next_offset < len(matches) else None
        parameters = {"q": query.text, "limit": query.page_size}
        if token is not None:
            parameters["cursor"] = token
        response = self._request(self.config.search_endpoint, parameters)
        responses.append(response)
        result = _json(response)
        records = []
        for index, raw in enumerate(_array(result["results"])):
            item = _object(raw)
            records.append(
                self._record(
                    record_id=item["id"],
                    title=item.get("title"),
                    url=item["url"],
                    snippet=item.get("snippet"),
                    date=item.get("date"),
                    rank=index + 1,
                    identifiers=(Identifier(identifier_type="URL", value=item["url"]),),
                )
            )
        token = result.get("next_cursor")
        if token is not None and type(token) is not str:
            raise ValueError("invalid web gateway cursor")
        return records, token

    def _fetch_url(self, locator):
        if locator.url is None:
            raise ValueError("web fetch requires an explicitly allowlisted URL")
        return locator.url, ("text/html", "application/pdf", "application/json")


CONNECTOR_TYPES = {
    "pubmed": PubMedConnector,
    "europe_pmc": EuropePMCConnector,
    "crossref": CrossrefResolver,
    "guideline": GuidelineSiteConnector,
    "web_fallback": WebSearchFallbackConnector,
}
