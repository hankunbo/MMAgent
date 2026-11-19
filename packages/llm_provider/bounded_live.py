"""One durable budget for probes and all roles; bounded, cached Knowledge calls."""

from packages.evidence_details.live_permission import validate_live_config
from packages.grounding.models import digest

from .provider import ProviderError, RequestBudget


class BoundedLiveBudget(RequestBudget):
    def __init__(
        self,
        *,
        binding,
        config,
        journal,
        clock=None,
        allocation=None,
        diagnostic_plan=None,
        bounded_phase=False,
    ):
        validate_live_config(config)
        from packages.grounding.models import validate

        from .live_allocation import read_allocation
        from .public_compatibility import CompatibilityPlan

        self.diagnostic_plan = (
            validate(CompatibilityPlan, diagnostic_plan) if diagnostic_plan else None
        )
        if self.diagnostic_plan and self.diagnostic_plan.run_id != binding.run_id:
            raise ProviderError("scope")
        self.allocation = read_allocation(allocation) if allocation is not None else None
        seconds = 1200 if self.allocation and self.allocation.stage == "public" else 2700
        p = binding.profile
        narrowed = bool(bounded_phase and p.seconds < seconds)
        if bounded_phase:
            if self.allocation or self.diagnostic_plan or not 0 < p.seconds <= seconds:
                raise ProviderError("configuration")
            seconds = p.seconds
        if (p.max_steps, p.max_no_progress, p.max_retries, p.seconds) != (20, 2, 1, seconds):
            raise ProviderError("configuration")
        self.recovery = getattr(config, "recovery", None)
        self.config_checksum = digest(config)
        self.batch_public_plan = None
        if (
            getattr(config, "review_batching", None)
            and self.allocation
            and self.allocation.stage == "public"
        ):
            from .public_batches import public_plan

            self.batch_public_plan = public_plan(binding.run_id, config)
        self.binding = binding
        self.cancel_signal = None
        super().__init__(
            run_id=binding.run_id,
            deadline=binding.deadline,
            max_requests=40,
            journal=journal,
            clock=clock,
        )
        with journal.session():
            expected = dict(
                schema_version="bounded-live-limits/1.1.0"
                if narrowed
                else "bounded-live-limits/1.0.0",
                config_checksum=digest(self.allocation)
                if self.allocation
                else self.config_checksum,
                max_steps=20,
                max_no_progress=2,
                max_searches=8,
                max_concurrency=1,
                max_retries=1,
                seconds=seconds,
                deadline=binding.deadline.isoformat(),
            )
            previous = [
                e
                for e in journal.read()
                if e.get("schema_version")
                in {"bounded-live-limits/1.0.0", "bounded-live-limits/1.1.0"}
            ]
            if previous and previous != [expected]:
                raise ProviderError("configuration")
            if not previous:
                journal.append(expected)
            if self.allocation:
                saved = [
                    e
                    for e in journal.read()
                    if e.get("schema_version") == self.allocation.schema_version
                ]
                event = self.allocation.model_dump(mode="json")
                if saved and saved != [event]:
                    raise ProviderError("configuration")
                if not saved:
                    journal.append(event)
            if self.diagnostic_plan:
                if (
                    self.allocation is None
                    or self.allocation.stage != "public"
                    or self.allocation.configuration_checksums
                    != tuple(c.config_checksum for c in self.diagnostic_plan.candidates)
                ):
                    raise ProviderError("configuration")
                event = dict(
                    schema_version="public-comparison-owner-plan/1.0.0",
                    plan=self.diagnostic_plan.model_dump(mode="json"),
                )
                saved = [
                    e for e in journal.read() if e.get("schema_version") == event["schema_version"]
                ]
                if saved and saved != [event]:
                    raise ProviderError("configuration")
                if not saved:
                    journal.append(event)
            if self.batch_public_plan:
                event = dict(
                    schema_version="public-batch-owner-plan/1.0.0", plan=self.batch_public_plan
                )
                saved = [
                    e for e in journal.read() if e.get("schema_version") == event["schema_version"]
                ]
                if saved and saved != [event]:
                    raise ProviderError("configuration")
                if not saved:
                    journal.append(event)

    def before_wire(self, config_checksum, scope, role, body):
        if self.batch_public_plan and (
            scope.case_code != "CASE:PUBLIC_PROBE"
            or scope.run_id != self.run_id
            or config_checksum != self.config_checksum
            or role != "review"
            or digest(body) not in {w for s in self.batch_public_plan["steps"] for w in s["wires"]}
        ):
            raise ProviderError("configuration")
        if self.diagnostic_plan:
            if scope.case_code != "CASE:PUBLIC_PROBE" or scope.run_id != self.run_id:
                raise ProviderError("scope")
            candidate = next(
                (
                    c
                    for c in self.diagnostic_plan.candidates
                    if c.config_checksum == config_checksum
                ),
                None,
            )
            if candidate is None or not any(
                r.role == role and r.wire_checksum == digest(body) for r in candidate.requests
            ):
                raise ProviderError("configuration")

    def public_format_transition(self, record, event, target_config):
        from .http_recovery import format_rejected

        if (
            not self.diagnostic_plan
            or not format_rejected(event)
            or record.scope.case_code != "CASE:PUBLIC_PROBE"
            or event.get("config_checksum") != record.config_checksum
        ):
            return False
        candidates = self.diagnostic_plan.candidates
        old = next(
            (i for i, c in enumerate(candidates) if c.config_checksum == record.config_checksum),
            None,
        )
        new = next(
            (i for i, c in enumerate(candidates) if c.config_checksum == target_config), None
        )
        return (
            old is not None
            and new is not None
            and new == old + 1
            and any(
                r.role == record.role
                and r.wire_checksum == event.get("wire_checksum")
                and r.input_checksum == record.input_checksum
                and r.prompt_checksum == record.prompt_checksum
                and r.schema_checksum == event.get("schema_checksum")
                for r in candidates[old].requests
            )
        )

    def record_response(self, record, reply):
        # HTTP worker already rejects secret echoes. This is untrusted evidence, never acceptance.
        documents = getattr(self.journal, "documents", None)
        if documents is not None and not record.simulated and reply.content:
            from packages.object_store import checksum

            with documents.session():
                documents.write(
                    "live-response:" + record.request_id,
                    (
                        dict(
                            schema_version="bounded-live-response/1.0.0",
                            request_id=record.request_id,
                            raw_model_text=reply.content.decode("utf-8"),
                            raw_checksum=checksum(reply.content),
                            validation_error=record.error,
                            untrusted_model_output=True,
                            accepted_medical_evidence=False,
                        ),
                    ),
                )

    def check_control(self):
        from agents.planner.service import Planner

        cancelled = any(e.get("phase") == "live_cancelled" for e in self.journal.read())
        if self.cancel_signal is not None and self.cancel_signal.is_set() and not cancelled:
            self.journal.append(dict(phase="live_cancelled"))
            cancelled = True
        if cancelled:
            raise ProviderError("cancelled")
        documents = getattr(self.journal, "documents", None)
        if documents is None or getattr(documents, "sessions", None) is None:
            raise ProviderError("agent_control_owner_unavailable")
        owner = Planner(runtime=documents.sessions, owner=None, clock=lambda: self.clock())
        try:
            owner.require_agent_work(self.binding)
        except ValueError as error:
            if str(error) in {
                "stopped",
                "cancelled",
                "deadline",
                "step_budget",
                "no_new_information",
                "agent_control_requires_confirmation",
                "agent_control_binding",
            }:
                raise ProviderError(str(error)) from error
            raise  # Unknown/malformed/missing Control is a technical error, never permission.

    def before_reserve(self, config_checksum):
        self.check_control()
        allowed = (
            self.allocation.configuration_checksums if self.allocation else (self.config_checksum,)
        )
        if config_checksum not in allowed:
            raise ProviderError("configuration")
        if self.allocation:
            from .live_allocation import prior_case_used

            limit = (
                min(12, 40 - prior_case_used(self.allocation))
                if self.allocation.stage == "public"
                else 40 - self.allocation.public_used - prior_case_used(self.allocation)
            )
            if self.used >= limit:
                raise ProviderError("budget")
        summary = self.summary()
        if summary.pending_unknown or summary.api_requests_unknown:
            raise ProviderError("transport")
        from .http_recovery import observation_for, retryable

        records = self.records()
        for record in records:
            if self.diagnostic_plan and record.error is not None:
                event = observation_for(self.journal, record.request_id)
                if self.public_format_transition(record, event, config_checksum):
                    continue
                if record.error not in {"timeout", "transport", "rate_limit"}:
                    raise ProviderError(record.error)
            if (
                self.recovery
                and record.retry_index >= 1
                and record.error in {"transport", "rate_limit"}
                and not any(
                    r.invocation_id == record.invocation_id and r.error is None for r in records
                )
            ):
                raise ProviderError(record.error)
            if record.error in {"timeout", "transport", "rate_limit"} and record.api_requests != 0:
                if self.recovery:
                    event = observation_for(self.journal, record.request_id)
                    if (
                        not retryable(event)
                        or event.get("config_checksum") != record.config_checksum
                    ):
                        raise ProviderError(record.error)
                elif record.error in {"timeout", "transport"}:
                    raise ProviderError("transport")


class BoundedKnowledge:
    """Wrap an existing installed Knowledge tool; never launches a second search stack.

    Cache key includes the exact structured query and tool/config version. A claimed
    search whose confirmation is missing is UNKNOWN, including across reconstruction.
    """

    def __init__(self, *, documents, budget, tool, tool_reference):
        self.documents, self.budget, self.tool = documents, budget, tool
        self.reference = tool_reference

    def search(self, query):
        key = "live-search:" + digest(dict(query=query, tool=self.reference))
        with self.documents.session():
            if self.budget.clock() >= self.budget.deadline:
                raise ProviderError("deadline")
            self.budget.before_reserve(self.budget.config_checksum)
            prior = self.documents.read(key)
            if prior:
                return prior[0]["result"]
            claims = [e for e in self.budget.journal.read() if e.get("phase") == "search_reserved"]
            if any(e["key"] == key for e in claims):
                raise ValueError("search_unknown_no_repeat")
            if len(claims) >= 8:
                raise ProviderError("budget")
            self.budget.journal.append(
                dict(phase="search_reserved", key=key, query_checksum=digest(query))
            )
        result = self.tool(query, deadline=self.budget.deadline)
        with self.documents.session():
            self.documents.write(key, (dict(result=result),))
        return result
