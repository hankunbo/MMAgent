"""Current-owner synthetic 11C entry. Saved/reserved recovery never restarts work."""

from datetime import UTC, datetime
from uuid import uuid4

from packages.grounding.models import digest, validate
from packages.tool_discovery.discovery import inspect, merge, query
from packages.tool_discovery.models_v11 import DiscoveryResult
from packages.tool_discovery.service import Discovery

from .gates import FakeProtocol
from .models import AdapterManifest, AdapterPlan, AdapterPolicy, SyntheticGrant


class SyntheticAuthority:
    def __init__(self, grants=()):
        values = tuple(validate(SyntheticGrant, g) for g in grants)
        self._grants = {(g.run_id, g.task_id): g for g in values}
        if len(self._grants) != len(values) or len(values) > 128:
            raise ValueError("adapter_grant_conflict_or_bound")
        self._revoked = set()

    def revoke(self, run_id, task_id):
        self._revoked.add((run_id, task_id))

    def current(self, run_id, task_id, stage_key, worker):
        g = self._grants.get((run_id, task_id))
        if g is None or (run_id, task_id) in self._revoked:
            raise ValueError("adapter_current_grant_required")
        if (g.stage_key, g.worker) != (stage_key, worker):
            raise ValueError("adapter_stage_identity_conflict")
        if g.expires_at <= datetime.now(UTC):
            raise ValueError("adapter_grant_expired")
        return g


class AdapterOnboarding:
    def __init__(self, *, discovery, repository, authority, manifest, policy=None, fake=None):
        if not isinstance(discovery, Discovery) or type(authority) is not SyntheticAuthority:
            raise ValueError("adapter_owner_required")
        self.discovery, self.authority = discovery, authority
        self.manifest = validate(AdapterManifest, manifest)
        self.policy = validate(AdapterPolicy, policy or AdapterPolicy())
        if fake is not None and type(fake) is not FakeProtocol:
            raise ValueError("adapter_fake_protocol_required")
        self.fake = validate(FakeProtocol, fake or FakeProtocol(case="not_installed"))
        self.repository = repository.with_owner(self)

    def read(self, run_id, key):
        return self.repository.read(run_id, key)

    def _prepare(self, run_id, key, *, stage_key, worker, lease_token):
        receipt = self.discovery.gaps.read(run_id, key)
        if receipt is None:
            raise ValueError("adapter_original_task_required")
        # The installed current grant is required BEFORE candidate I/O or stage work.
        grant = self.authority.current(run_id, receipt.task_id, stage_key, worker)
        a = self.discovery.gaps.recheck(receipt.application)
        historical = self.discovery.read(run_id, key)
        if historical is None or historical["status"] != "saved":
            raise ValueError("adapter_discovery_unconfirmed")
        raw = historical["result"]
        if raw["schema_version"] != "tool-discovery-result/1.1.0":
            raise ValueError("adapter_legacy_input_contract")
        result = validate(DiscoveryResult, raw)
        if result.status != "discovered" or grant.candidate_identity not in result.selected:
            raise ValueError("adapter_candidate_history_missing")
        q = query(a, self.discovery.policy)
        if q != result.query:
            raise ValueError("adapter_current_input_changed")
        observations = []
        remaining = self.discovery.policy.max_records
        for source in self.discovery.sources:
            if remaining <= 0:
                raise ValueError("adapter_source_budget_exhausted")
            records, truncated = source.read(q, limit=remaining)
            if truncated:
                raise ValueError("adapter_source_budget_exhausted")
            remaining -= len(records)
            observations.extend(inspect(r, source, q, self.discovery.policy) for r in records)
        current = next(
            (c for c in merge(observations) if c.identity == grant.candidate_identity), None
        )
        prior = next(c for c in result.candidates if c.identity == grant.candidate_identity)
        if current is None or current != prior:
            raise ValueError("adapter_candidate_evidence_changed")
        if self.manifest.candidate not in current.records:
            raise ValueError("adapter_manifest_candidate_binding")
        return AdapterPlan(
            task_id=receipt.task_id,
            application=a,
            discovery=result,
            query=q,
            candidate=current,
            manifest=self.manifest,
            policy=self.policy,
            grant=grant,
            fake_protocol_checksum=digest(self.fake),
            lease_token=lease_token,
            runtime_context=self.repository._context(run_id),
        )

    def consume(self, run_id, key, *, stage_key, worker):
        saved = self.read(run_id, key)
        if saved is not None:
            g = saved["plan"]["grant"]
            if (stage_key, worker) != (g["stage_key"], g["worker"]):
                raise ValueError("adapter_stage_identity_conflict")
            return saved
        plan = self._prepare(
            run_id, key, stage_key=stage_key, worker=worker, lease_token="adapter:" + uuid4().hex
        )
        value = self.repository._reserve(plan)
        if not value["created"]:
            return value["document"]
        result = self.fake.evaluate(plan)
        return self.repository._commit(plan, result)

    def cancel(self, run_id, key, *, stage_key, worker):
        """Close this original stage only; never renews it or cancels a completed history."""
        saved = self.read(run_id, key)
        if saved is None:
            raise ValueError("adapter_reservation_missing")
        plan = validate(AdapterPlan, saved["plan"])
        if (stage_key, worker) != (plan.grant.stage_key, plan.grant.worker):
            raise ValueError("adapter_stage_identity_conflict")
        return self.repository._cancel(plan)

    def report(self, run_id, key):
        return dict(
            schema_version="synthetic-adapter-report/1.0.0",
            original_requirement=self.discovery.gaps.report(run_id, key),
            adapter=self.read(run_id, key),
            historical_read_not_authorization=True,
            recovery="same_stage_read_only_no_renewal_or_restart",
            live_domain_acceptance="live_domain_acceptance_not_enabled",
        )
