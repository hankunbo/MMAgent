"""04B2 sole search write service; no Source identity, archive or publish access."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from db.knowledge import (
    JobStatus,
    KnowledgeSnapshot,
    KnowledgeStatus,
    SearchHit,
    SearchQuery,
    SearchRun,
    SourceIdentifierType,
    SourceIdentity,
    seal,
)
from db.knowledge.repository import KnowledgeConflictError, KnowledgeRepository
from db.knowledge.search_integrity import validate_search_records
from packages.knowledge_connectors.ports import digest
from packages.knowledge_search import FederatedSearch, KnowledgeQueryPlan
from packages.knowledge_search.models import FederatedResult


@dataclass(frozen=True)
class SearchReceipt:
    status: str
    cache_hit: bool
    records: tuple[KnowledgeSnapshot, ...]


class KnowledgeSearchService:
    def __init__(
        self,
        executor: FederatedSearch,
        repository: KnowledgeRepository,
        providers: tuple[KnowledgeSnapshot, ...],
        *,
        synthetic: bool = False,
    ):
        if type(synthetic) is not bool:
            raise TypeError("synthetic must be an explicit boolean")
        self.executor, self.repository, self.synthetic = executor, repository, synthetic
        self.providers = {}
        for snapshot in providers:
            snapshot = KnowledgeSnapshot.model_validate(snapshot)
            if (
                snapshot.status is not KnowledgeStatus.DRAFT
                or snapshot.withdrawn
                or snapshot.record.kind.value != "source_provider"
            ):
                raise ValueError("search requires registered Draft providers")
            if snapshot.record.configuration_proof is None:
                raise ValueError("registered provider configuration proof required")
            self.providers[snapshot.record.code] = snapshot.record
        # Fixed lock stripes bound memory and serialize retries of the same key.
        self._locks = tuple(threading.Lock() for _ in range(64))

    def search(
        self,
        plan: KnowledgeQueryPlan,
        *,
        idempotency_key: str,
        cancel: threading.Event | None = None,
    ) -> SearchReceipt:
        plan = KnowledgeQueryPlan.model_validate(plan)
        if type(idempotency_key) is not str or not 1 <= len(idempotency_key) <= 200:
            raise ValueError("bounded idempotency key required")
        key_hash = digest(idempotency_key.encode()).split(":")[1]
        binding = self.executor.execution_checksum(plan)
        key = "04B2:" + key_hash
        with self._locks[int(key_hash[:8], 16) % len(self._locks)]:
            for query in self.executor.queries(plan):
                ref = self.providers.get(query.identity.provider_code)
                if (
                    ref is None
                    or ref.version != query.identity.provider_version
                    or ref.configuration_proof != query.identity.configuration
                ):
                    raise KnowledgeConflictError("provider registration/version is missing")
                registered = self.repository.get(ref.reference())
                if (
                    registered.record != ref
                    or registered.status is not KnowledgeStatus.DRAFT
                    or registered.withdrawn
                ):
                    raise KnowledgeConflictError("exact registered provider is unavailable")
            cached = self.repository.lookup_search(key, binding)
            if cached is not None:
                return self._receipt(cached, True, binding)
            result = FederatedResult.model_validate(self.executor.execute(plan, cancel=cancel))
            if (
                result.binding.plan != plan
                or result.execution_checksum != binding
                or result.binding.queries != self.executor.queries(plan)
            ):
                raise ValueError("live result does not belong to this execution")
            records = []
            for outcome in result.providers:
                identity = outcome.query.identity
                provider = self.providers[identity.provider_code]
                suffix = digest((key + identity.provider_code).encode()).split(":")[1].upper()
                query = seal(
                    SearchQuery,
                    code="SOURCE:QUERY:" + suffix,
                    version="1.0.0",
                    synthetic=self.synthetic,
                    provider_ref=provider.reference(),
                    provider_snapshot=provider,
                    query_text=" | ".join(outcome.query.texts),
                    config_version=plan.query_plan_version,
                    query_plan=plan,
                    provider_query=outcome.query,
                )
                run = seal(
                    SearchRun,
                    code="SOURCE:SEARCH:" + suffix,
                    version="1.0.0",
                    synthetic=self.synthetic,
                    query_ref=query.reference(),
                    provider_ref=provider.reference(),
                    started_at=outcome.started_at,
                    completed_at=outcome.completed_at,
                    run_status=JobStatus.SUCCEEDED
                    if outcome.stop_reason in ("complete", "empty")
                    else JobStatus.FAILED,
                    response_checksum=digest(
                        "".join(p.page_checksum for p in outcome.receipts).encode()
                    ),
                    execution=outcome.persisted(),
                    execution_checksum=result.execution_checksum,
                    federated_status=result.status,
                )
                records.extend((query, run))
                for rank, hit in enumerate(outcome.hits, 1):
                    hit_suffix = (
                        digest((suffix + hit.metadata.record_id_checksum).encode())
                        .split(":")[1]
                        .upper()
                    )
                    records.append(
                        seal(
                            SearchHit,
                            code="SOURCE:HIT:" + hit_suffix,
                            version="1.0.0",
                            synthetic=self.synthetic,
                            run_ref=run.reference(),
                            provider_ref=provider.reference(),
                            rank=rank,
                            title=hit.title,
                            identifiers=tuple(
                                SourceIdentity(
                                    identifier_type=SourceIdentifierType(i.identifier_type),
                                    value=i.value,
                                )
                                for i in hit.identifiers
                            ),
                            discovered_at=hit.discovered_at,
                            discovery=hit.metadata,
                        )
                    )
            stored = self.repository.commit_search(tuple(records), key, binding)
            return self._receipt(stored, False, binding)

    @staticmethod
    def _receipt(records, cache_hit, binding):
        records = tuple(KnowledgeSnapshot.model_validate(s) for s in records)
        status = validate_search_records(tuple(s.record for s in records), binding)
        return SearchReceipt(status=status, cache_hit=cache_hit, records=records)
