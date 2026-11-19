"""One bounded offline discovery per original Task. No daemon or implicit retry."""

import time

from packages.grounding.models import canonical, digest, validate
from packages.tool_gap_queue.service import GapQueue

from .discovery import inspect, merge, query
from .models import SOURCE_ORDER, DiscoveryPolicy
from .models_v11 import DiscoveryPlan, DiscoveryResult
from .sources import SyntheticSource
from .versions import contract_status


class Discovery:
    def __init__(self, *, gap_queue, repository, sources=(), policy=None):
        if not isinstance(gap_queue, GapQueue):
            raise ValueError("discovery_gap_owner_required")
        self.gaps = gap_queue
        self.policy = validate(DiscoveryPolicy, policy or DiscoveryPolicy())
        if len(sources) > self.policy.max_sources or any(
            type(s) is not SyntheticSource for s in sources
        ):
            raise ValueError("discovery_offline_sources_required")
        self.sources = tuple(sorted(sources, key=lambda s: (SOURCE_ORDER.index(s.kind), s.name)))
        if len({s.name for s in self.sources}) != len(self.sources):
            raise ValueError("discovery_source_name_conflict")
        self.repository = repository.with_owner(self) if repository is not None else None

    def bindings(self):
        return tuple(
            dict(name=s.name, kind=s.kind, fingerprint=s.fingerprint) for s in self.sources
        )

    def read(self, run_id, key):
        """Historical result/reservation; never authorizes candidate work."""
        return self.repository.read(run_id, key)

    def _prepare(self, receipt, *, claim_key, worker):
        from db.runtime.repository.case_pipeline import identity

        a = self.gaps.recheck(receipt.application)
        q = query(a, self.policy)
        ack_key = "discovery-ack:" + digest([receipt.task_id, claim_key])[7:55]
        ack_tag = receipt.task_id + ":ack:" + ack_key
        operation = dict(
            action="ack", parameters=dict(lease_token=receipt.lease_token, worker=worker)
        )
        plan = dict(
            schema_version="tool-discovery-plan/1.1.0",
            task_id=receipt.task_id,
            application_checksum=digest(a),
            query=q.model_dump(mode="json"),
            policy=self.policy.model_dump(mode="json"),
            sources=list(self.bindings()),
            claim_key=claim_key,
            ack_key=ack_key,
            ack_command_checksum=digest(operation),
            ack_event_id=identity("MESSAGE", ack_tag),
            ack_idempotency_key=identity("TASK", ack_tag),
            worker=worker,
            lease_token=receipt.lease_token,
        )
        return validate(DiscoveryPlan, plan).model_dump(mode="json")

    def consume(self, run_id, key, *, claim_key, worker):
        saved = self.read(run_id, key)
        if saved is not None:
            plan = saved["plan"]
            if (plan["claim_key"], plan["worker"]) != (claim_key, worker):
                raise ValueError("discovery_pending_identity_conflict")
            # Historical recovery is allowed after withdrawal. No owner/source I/O or claim here.
            return saved
        began = time.monotonic()
        lease = self.gaps.claim(run_id, key, command_key=claim_key, worker=worker)
        if lease is None or lease.state != "leased":
            return dict(status="not_leased", result=None, historical_read_not_authorization=True)
        plan = self._prepare(lease, claim_key=claim_key, worker=worker)
        reservation, created, remaining = self.repository._reserve(lease, plan)
        if not created:
            return reservation
        budget = min(self.policy.max_seconds, remaining - self.policy.commit_margin_seconds)
        deadline = began + budget  # Conservative: also counts claim/owner/transaction overhead.
        if budget <= 0 or time.monotonic() >= deadline:
            return reservation
        result = self._evaluate(lease.task_id, plan, deadline=deadline)
        # The commit repeats original owner reconstruction and current-claim check. If expired,
        # cancelled or withdrawn, the original reservation remains visible, never a false ack.
        return self.repository._commit(lease, plan, result)

    def _evaluate(self, task_id, plan, *, deadline):
        from .models_v11 import DiscoveryQuery

        if plan["sources"] != list(self.bindings()) or plan["policy"] != self.policy.model_dump(
            mode="json"
        ):
            raise ValueError("discovery_source_or_policy_changed")
        q = validate(DiscoveryQuery, plan["query"])
        began = time.monotonic()
        observations, reports = [], []
        examined, used_bytes = 0, 0
        exhausted, technical = False, False
        for source in self.sources:
            left = self.policy.max_records - examined
            if left <= 0 or time.monotonic() >= deadline:
                exhausted = True
                break
            try:
                records, truncated = source.read(q, limit=left)
            except OSError as exc:
                reports.append(
                    dict(
                        name=source.name,
                        kind=source.kind,
                        status="technical_failure",
                        reason=type(exc).__name__,
                        records=0,
                    )
                )
                technical = True
                continue
            count = 0
            for record in records:
                size = len(canonical(record))
                if (
                    time.monotonic() >= deadline
                    or used_bytes + size > self.policy.max_metadata_bytes
                ):
                    exhausted = True
                    break
                used_bytes += size
                observations.append(inspect(record, source, q, self.policy))
                examined += 1
                count += 1
            exhausted |= truncated
            reports.append(
                dict(
                    name=source.name,
                    kind=source.kind,
                    status="bounded" if truncated or count < len(records) else "read",
                    records=count,
                )
            )
            if exhausted:
                break
        candidates = merge(observations)
        selected = tuple(c.identity for c in candidates if c.disposition == "discovered")[
            : self.policy.top_k
        ]
        status = (
            "technical_failure"
            if technical
            else "budget_exhausted"
            if exhausted
            else "discovered"
            if selected
            else "incomplete"
            if candidates
            else "no_candidates"
        )
        return DiscoveryResult(
            task_id=task_id,
            plan_checksum=digest(plan),
            query=q,
            policy=self.policy,
            status=status,
            candidates=candidates,
            selected=selected,
            sources=tuple(reports),
            records_examined=examined,
            elapsed_seconds=time.monotonic() - began,
            memory=dict(status="not_requested", extension_authority=False),
        )

    def acknowledge_saved(self, run_id, key):
        """Read atomic result + original ack fact; no new work even after source withdrawal."""
        saved = self.read(run_id, key)
        if saved is None:
            return None
        saved["ack_history"] = self.gaps.read_command(run_id, key, saved["plan"]["ack_key"])
        if saved["ack_history"] is not None:
            saved["ack_history"] = saved["ack_history"].model_dump(mode="json")
        return saved

    def memory_diagnostics(self, run_id):
        """Optional original retrieval/recheck path; does not enter ranking or write results."""
        return self.gaps.memory_diagnostics(run_id)

    def report(self, run_id, key):
        original = self.gaps.report(run_id, key)
        record = self.read(run_id, key)
        return dict(
            schema_version="tool-discovery-report/1.1.0",
            input_contract_status=contract_status(record),
            original_requirement=original,
            discovery=record,
            historical_read_not_authorization=True,
            recovery=(
                "read_same_task_and_command; reserved_does_not_restart; explicit_fail_or_cancel"
            ),
            onboarding_prerequisites=[
                "real_candidate_identity",
                "formal_extension_task",
                "source_and_target_catalog",
                "approved_policy",
                "complete_original_v1_contract",
            ],
            live_domain_acceptance="live_domain_acceptance_not_enabled",
        )
