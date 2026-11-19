"""One bounded 11D stage on the original Task; no queue or activation service."""

from datetime import UTC, datetime
from uuid import uuid4

from packages.grounding.models import digest, validate
from packages.tool_adapter.models import AdapterPlan, AdapterResult
from packages.tool_adapter.service import AdapterOnboarding
from packages.tool_discovery.discovery import inspect, merge, query

from .catalog import SyntheticCatalogOwner
from .future import SyntheticFutureRunOwner
from .models import ExtensionGrant, ExtensionPlan, ExtensionPolicy, ExtensionResult


class ExtensionAuthority:
    def __init__(self, grants=()):
        values = tuple(validate(ExtensionGrant, g) for g in grants)
        self._grants = {(g.run_id, g.task_id): g for g in values}
        if len(values) != len(self._grants) or len(values) > 128:
            raise ValueError("extension_grant_conflict_or_bound")
        self._revoked = set()

    def revoke(self, run_id, task_id):
        self._revoked.add((run_id, task_id))

    def current(self, run_id, task_id, stage_key, worker):
        g = self._grants.get((run_id, task_id))
        if g is None or (run_id, task_id) in self._revoked:
            raise ValueError("extension_current_grant_required")
        if (stage_key, worker) != (g.stage_key, g.worker):
            raise ValueError("extension_stage_identity_conflict")
        if g.expires_at <= datetime.now(UTC):
            raise ValueError("extension_grant_expired")
        return g


class OfflineExtension:
    def __init__(self, *, adapter, repository, authority, policy=None, catalog=None, future=None):
        if not isinstance(adapter, AdapterOnboarding) or type(authority) is not ExtensionAuthority:
            raise ValueError("extension_owner_required")
        if catalog is not None and type(catalog) is not SyntheticCatalogOwner:
            raise ValueError("extension_synthetic_catalog_owner_required")
        if future is not None and (
            type(future) is not SyntheticFutureRunOwner or future.catalog is not catalog
        ):
            raise ValueError("extension_future_owner_required")
        self.adapter, self.authority = adapter, authority
        self.policy = validate(ExtensionPolicy, policy or ExtensionPolicy())
        self.catalog, self.future = catalog, future
        self.repository = repository.with_owner(self)

    def read(self, run_id, key):
        return self.repository.read(run_id, key)

    def _prepare(self, run_id, key, *, stage_key, worker, lease_token):
        discovery = self.adapter.discovery
        receipt = discovery.gaps.read(run_id, key)
        if receipt is None:
            raise ValueError("extension_original_task_required")
        grant = self.authority.current(run_id, receipt.task_id, stage_key, worker)
        record = self.adapter.read(run_id, key)
        if record is None or record["status"] != "saved":
            raise ValueError("extension_adapter_unconfirmed")
        p = validate(AdapterPlan, record["plan"])
        result = validate(AdapterResult, record["result"])
        # Recheck original owner and full current source metadata under D authority.
        # The C grant is historical and is neither renewed nor consumed again.
        current = discovery.gaps.recheck(p.application)
        if current != p.application or discovery.read(run_id, key)[
            "result"
        ] != p.discovery.model_dump(mode="json"):
            raise ValueError("extension_original_material_changed")
        q = query(current, discovery.policy)
        if (
            q != p.query
            or self.adapter.manifest != p.manifest
            or self.adapter.policy != p.policy
            or digest(self.adapter.fake) != p.fake_protocol_checksum
        ):
            raise ValueError("extension_current_contract_changed")
        observations, remaining = [], discovery.policy.max_records
        for source in discovery.sources:
            if remaining <= 0:
                raise ValueError("extension_source_budget")
            records, truncated = source.read(q, limit=remaining)
            if truncated:
                raise ValueError("extension_source_budget")
            remaining -= len(records)
            observations.extend(inspect(r, source, q, discovery.policy) for r in records)
        observed = next(
            (c for c in merge(observations) if c.identity == p.candidate.identity), None
        )
        if observed != p.candidate:
            raise ValueError("extension_candidate_changed")
        runtime_context = self.repository._context(run_id)
        binding = self.future.binding(p) if self.future else None
        # Original owner recheck can itself call 05/06. Check the local catalog
        # and independent D authority AFTER all those calls and input-owner reads.
        # No admission or result evaluation follows these terminal checks here.
        fixture = self.catalog.fixture if self.catalog else None
        observation = self.catalog.current(record) if self.catalog else None
        if self.authority.current(run_id, receipt.task_id, stage_key, worker) != grant:
            raise ValueError("extension_current_grant_changed")
        if (fixture is not None or binding is not None) and result.outcome != "missing":
            raise ValueError("extension_unconfirmed_or_failed_not_fixture_eligible")
        return ExtensionPlan(
            task_id=receipt.task_id,
            adapter_record=record,
            adapter_plan=p,
            adapter_result=result,
            policy=self.policy,
            grant=grant,
            catalog_fixture=fixture,
            catalog_observation=observation,
            future_binding=binding,
            runtime_context=runtime_context,
            lease_token=lease_token,
        )

    def _recheck(self, plan):
        current = OfflineExtension._prepare(
            self,
            plan.grant.run_id,
            plan.adapter_plan.application.key,
            stage_key=plan.grant.stage_key,
            worker=plan.grant.worker,
            lease_token=plan.lease_token,
        )
        if current != plan:
            raise ValueError("extension_current_plan_changed")

    def _evaluate(self, plan):
        a = plan.adapter_result
        missing = list(a.missing)
        missing += [
            "formal_tool_extension_identity",
            "formal_catalog_publication_owner",
            "original_06_real_execution_assembly",
            "executed_counterfactual_pair",
        ]
        if self.catalog is None:
            missing.append("synthetic_catalog_fixture_not_installed")
        if self.future is None:
            missing.append("synthetic_new_run_owner_not_installed")
        return ExtensionResult(
            task_id=plan.task_id,
            plan_checksum=digest(plan),
            adapter_outcome=a.outcome,
            catalog_status=a.catalog_status,
            extension_job_status=a.extension_job_status,
            fake_catalog_ready=plan.catalog_fixture is not None,
            future_check=self.future.evaluate(plan) if self.future else None,
            missing=tuple(dict.fromkeys(missing)),
        )

    def consume(self, run_id, key, *, stage_key, worker):
        saved = self.read(run_id, key)
        if saved is not None:
            g = saved["plan"]["grant"]
            if (g["stage_key"], g["worker"]) != (stage_key, worker):
                raise ValueError("extension_stage_identity_conflict")
            return saved
        plan = self._prepare(
            run_id, key, stage_key=stage_key, worker=worker, lease_token="extension:" + uuid4().hex
        )
        reserved = self.repository._reserve(plan)
        if not reserved["created"]:
            return reserved["document"]
        return self.repository._commit(plan, self._evaluate(plan))

    def cancel(self, run_id, key, *, stage_key, worker):
        saved = self.read(run_id, key)
        if saved is None:
            raise ValueError("extension_reservation_missing")
        plan = validate(ExtensionPlan, saved["plan"])
        if (plan.grant.stage_key, plan.grant.worker) != (stage_key, worker):
            raise ValueError("extension_stage_identity_conflict")
        return self.repository._cancel(plan)

    def report(self, run_id, key):
        return dict(
            schema_version="synthetic-extension-report/1.0.0",
            synthetic=True,
            original_requirement=self.adapter.discovery.gaps.report(run_id, key),
            extension=self.read(run_id, key),
            historical_read_not_authorization=True,
            main_run_waits=False,
            recovery="same_stage_read_only_no_restart",
            memory_published=False,
            live_domain_acceptance="live_domain_acceptance_not_enabled",
        )
