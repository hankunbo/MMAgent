"""Bounded discovery; connector bodies live only within one page-processing call."""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from packages.knowledge_connectors import (
    ConnectorConfiguration,
    ConnectorError,
    Identifier,
    ProviderRegistry,
    SafeHTTPClient,
    SearchPage,
    StructuredQuery,
)
from packages.knowledge_connectors.http import Clock
from packages.knowledge_connectors.ports import checked, digest, model_checksum

from .models import (
    CandidateHit,
    ExecutionBinding,
    FederatedResult,
    HitMetadata,
    KnowledgeQueryPlan,
    PageReceipt,
    ProviderOutcome,
    derive_federated_status,
    safe_text,
)
from .planning import provider_query
from .policy import ProviderGovernor, RatePolicy, RunBudget, SearchStopped, SourceBudget
from .privacy import PrivacyError


class FederatedSearch:
    """One assembly owns every client and the provider-code governor.

    client_factory(config, governor) must attach that governor to SafeHTTPClient.
    Independent executors for the same provider are rejected within this process;
    deployment policy additionally requires one worker for each provider.
    """

    _ownership_lock = threading.Lock()
    _owners: set[str] = set()

    def __init__(
        self,
        configuration: ConnectorConfiguration,
        policy: RatePolicy,
        client_factory: Callable,
        *,
        clock: Clock | None = None,
        max_workers: int = 8,
        request_guard=None,
    ):
        self.configuration = checked(ConnectorConfiguration, configuration)
        if type(max_workers) is not int or not 1 <= max_workers <= 8:
            raise ValueError("bounded executor worker count required")
        self.governor = ProviderGovernor(
            configuration.providers, policy, clock, request_guard=request_guard
        )
        self.clock = self.governor.clock
        self.policy = self.governor.policy
        self._codes = {p.provider_code for p in configuration.providers}
        self._closed = False
        self._active = threading.BoundedSemaphore(4)
        self._provider_locks = {code: threading.Lock() for code in self._codes}
        with self._ownership_lock:
            if self._owners & self._codes:
                raise ValueError("a provider already belongs to an executor in this process")
            self._owners.update(self._codes)
        try:

            def factory(config):
                client = client_factory(config, self.governor)
                if type(client) is not SafeHTTPClient or client.governor is not self.governor:
                    raise ValueError("all clients must use the executor governor")
                return client

            self.registry = ProviderRegistry(configuration, factory)
            self._pool = ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="knowledge-search"
            )
        except BaseException:
            with self._ownership_lock:
                self._owners.difference_update(self._codes)
            raise

    def close(self):
        if not self._closed:
            self._closed = True
            self._pool.shutdown(wait=True, cancel_futures=False)
            with self._ownership_lock:
                self._owners.difference_update(self._codes)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def queries(self, plan):
        plan = checked(KnowledgeQueryPlan, plan)
        configs = {c.provider_code: c for c in self.configuration.providers}
        result = []
        for budget in plan.providers:
            if budget.provider_code not in configs:
                raise ValueError("plan names an unregistered provider")
            config = configs[budget.provider_code]
            if (
                budget.page_size > config.http.max_page_size
                or budget.max_pages > config.http.max_pages
            ):
                raise ValueError("plan conflicts with the pinned connector limits")
            result.append(provider_query(plan, config, self.policy))
        return tuple(result)

    def execution_checksum(self, plan):
        # The plan hash is semantic; execution additionally binds all provider config
        # and quota policy versions, without reading any environment values.
        from .models import ExecutionBinding

        return model_checksum(
            ExecutionBinding(
                plan=plan,
                queries=self.queries(plan),
                provider_proofs=tuple(q.identity.configuration for q in self.queries(plan)),
            )
        )

    @contextmanager
    def _slot(self, lock):
        while not lock.acquire(timeout=self.policy.poll_ms / 1000):
            self.governor.checkpoint()
        try:
            self.governor.checkpoint()
            yield
        finally:
            lock.release()

    def execute(
        self, plan: KnowledgeQueryPlan, *, cancel: threading.Event | None = None
    ) -> FederatedResult:
        plan = checked(KnowledgeQueryPlan, plan)
        queries = self.queries(plan)
        binding = self.execution_checksum(plan)
        if self._closed or not self._active.acquire(blocking=False):
            raise RuntimeError("executor is closed or its active-run bound is exhausted")
        try:
            run = RunBudget(
                self.clock.monotonic() + plan.total_timeout_ms / 1000,
                plan.total_max_requests,
                plan.total_max_hits,
                cancel or threading.Event(),
            )
            concurrency = threading.BoundedSemaphore(plan.max_concurrency)
            futures = [
                self._pool.submit(self._source, plan, query, budget, run, concurrency)
                for query, budget in zip(queries, plan.providers, strict=True)
            ]
            outcomes = tuple(f.result() for f in futures)
            return FederatedResult(
                binding=ExecutionBinding(
                    plan=plan,
                    queries=queries,
                    provider_proofs=tuple(q.identity.configuration for q in queries),
                ),
                plan_checksum=plan.checksum(),
                execution_checksum=binding,
                status=derive_federated_status(outcomes),
                providers=outcomes,
            )
        finally:
            self._active.release()

    def _source(self, plan, query, budget, run, concurrency):
        started = self.clock.now()
        start = self.clock.monotonic()
        source = SourceBudget(
            run,
            min(run.deadline, start + budget.timeout_ms / 1000),
            budget.max_requests,
            max_retries=budget.max_retries,
        )
        receipts, hits = [], []
        seen_ids, seen_pages, seen_cursors = set(), set(), set()
        reason, category = "complete", None
        with self.governor.bind(source):
            try:
                with (
                    self._slot(concurrency),
                    self._slot(self._provider_locks[budget.provider_code]),
                ):
                    connector = self.registry.get(budget.provider_code)
                    if connector.capabilities().scope() != query.identity.scope():
                        raise ValueError("connector configuration scope changed")
                    for subindex, text in enumerate(query.texts):
                        cursor = None
                        while True:
                            self.governor.checkpoint()
                            if len(receipts) >= budget.max_pages:
                                raise SearchStopped("page_limit")
                            with run.lock:
                                available = min(
                                    budget.max_hits - len(hits), run.max_hits - run.hits
                                )
                            if available <= 0:
                                raise SearchStopped("hit_budget")
                            # Keep page size fixed: changing it would invalidate 04B1 cursors.
                            request = StructuredQuery(
                                text=text, page_size=budget.page_size, cursor=cursor
                            )
                            page = checked(SearchPage, connector.search(request))
                            self.governor.checkpoint()
                            if (
                                page.scope() != query.identity.scope()
                                or page.query_checksum != request.checksum()
                            ):
                                raise ValueError("returned page is outside the requested scope")
                            page_index = len(receipts)
                            receipts.append(
                                PageReceipt(
                                    subquery_index=subindex,
                                    page_index=page_index,
                                    query_checksum=page.query_checksum,
                                    page_checksum=page.page_checksum,
                                    cursor_checksum=model_checksum(cursor) if cursor else None,
                                    requests=tuple(r.metadata for r in page.responses),
                                    response_checksums=tuple(
                                        r.response_checksum for r in page.responses
                                    ),
                                    retrieved_at=page.retrieved_at,
                                )
                            )
                            # Ignore rank changes when detecting the same logical page.
                            signature = tuple(sorted(r.record_id for r in page.records))
                            if (
                                signature in seen_pages
                                and page.records
                                and plan.stop_on_duplicate_page
                            ):
                                raise SearchStopped("duplicate_page")
                            seen_pages.add(signature)
                            for record in page.records:
                                if record.record_id in seen_ids:
                                    continue
                                candidate = self._candidate(
                                    record, page, subindex, page_index, query.identity.configuration
                                )
                                with run.lock:
                                    if run.hits >= run.max_hits or len(hits) >= budget.max_hits:
                                        raise SearchStopped("hit_budget")
                                    run.hits += 1
                                hits.append(candidate)
                                seen_ids.add(record.record_id)
                            if not page.records and plan.stop_on_empty:
                                break
                            if page.next_cursor is None:
                                break
                            cursor_key = digest(page.next_cursor.token.encode())
                            if cursor_key in seen_cursors:
                                raise SearchStopped("duplicate_page")
                            seen_cursors.add(cursor_key)
                            cursor = page.next_cursor
                    reason = "complete" if hits else "empty"
            except SearchStopped as error:
                reason = error.reason
            except ConnectorError as error:
                reason = (
                    "timeout"
                    if error.category.value == "timeout"
                    or (error.last_error and error.last_error.value == "timeout")
                    else "provider_error"
                )
                category = error.category.value
            except PrivacyError:
                run.cancel.set()
                raise
            except (ValueError, TypeError):
                reason, category = "provider_error", "security"
            except Exception:
                # Do not serialize exception text, which may contain a provider URL/body.
                reason, category = "provider_error", "internal"
        request_ids = {r.request_checksum for p in receipts for r in p.requests}
        extra = tuple(r for r in source.metadata if r.request_checksum not in request_ids)
        return ProviderOutcome(
            query=query,
            started_at=started,
            completed_at=self.clock.now(),
            elapsed_ms=max(0, int((self.clock.monotonic() - start) * 1000)),
            stop_reason=reason,
            error_category=category,
            wire_requests=source.requests,
            wire_receipts=tuple(source.wire_receipts),
            receipts=tuple(receipts),
            error_requests=extra,
            hits=tuple(hits),
            hit_count=len(hits),
        )

    @staticmethod
    def _candidate(record, page, subindex, page_index, configuration=None):
        # Copy only a safe discovery title and query-free URL. Never persist snippets,
        # response bodies, record IDs (which can be URLs), or contact-bearing queries.
        from .discovery import DiscoveryPolicy
        from .discovery import identifier as typed_identifier
        from .discovery import record_id as typed_record_id
        from .discovery import title as typed_title

        if configuration is None:
            raise ValueError("explicit configured provider proof required")
        title = typed_title(record.title or "Untitled discovery")
        safe_text(title)
        clean_url = typed_identifier("URL", record.url, configuration)
        if "://" not in record.record_id:
            safe_text(record.record_id)
        else:
            typed_identifier("URL", record.record_id, configuration)
        for identifier in record.identifiers:
            typed_identifier(identifier.identifier_type, identifier.value, configuration)
        identifiers = (Identifier(identifier_type="URL", value=clean_url),)
        record_id = (
            typed_record_id(record.record_id, configuration)
            if "://" not in record.record_id
            else None
        )
        raw_identifiers = tuple(
            Identifier(
                identifier_type=i.identifier_type,
                value=typed_identifier(i.identifier_type, i.value, configuration),
            )
            for i in record.identifiers
            if i.identifier_type != "URL"
        )
        return CandidateHit(
            metadata=HitMetadata(
                configuration=configuration,
                discovery_policy=DiscoveryPolicy(),
                provider_code=record.provider_code,
                provider_version=record.provider_version,
                adapter=record.adapter,
                config_checksum=record.config_checksum,
                record_id_checksum=digest(record.record_id.encode()),
                provider_record_id=record_id,
                discovery_identifiers=raw_identifiers,
                source_type=record.source_type,
                subquery_index=subindex,
                page_index=page_index,
                provider_rank=record.rank,
                page_checksum=page.page_checksum,
                query_checksum=page.query_checksum,
                request_checksums=tuple(r.metadata.request_checksum for r in page.responses),
                response_checksums=tuple(r.response_checksum for r in page.responses),
            ),
            title=title,
            identifiers=identifiers,
            discovered_at=page.retrieved_at,
        )
