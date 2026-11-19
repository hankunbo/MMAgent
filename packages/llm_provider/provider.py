"""Shared provider, bounded retry and journal. No Agent owns an HTTP client."""

import json
import threading
import time
from dataclasses import replace
from datetime import UTC, datetime
from typing import Protocol

from packages.grounding.models import canonical, digest, validate

from .config import prompt, read_provider_config
from .models import RequestRecord, RequestSummary, RolePayload, Scope, Usage, read_request_record
from .transport import DeterministicFakeTransport, HTTPTransport, Reply, wire_request


class ProviderError(ValueError):
    def __init__(self, code, *, fields=()):
        super().__init__(code)
        self.code = code
        self.fields = tuple(fields)


class LLMProvider(Protocol):
    def generate_structured(
        self,
        *,
        role,
        input_payload,
        output_schema,
        scope,
        budget,
        deadline,
        images=(),
        admit=None,
        before_request=None,
    ): ...


class MemoryJournal:
    """Explicit Fake journal; durable assembly uses RuntimeAgentJournal."""

    durable = False

    def __init__(self):
        self.events = []
        self.lock = threading.RLock()

    def session(self):
        return self.lock

    def validate_binding(self, run_id, deadline):
        pass  # Explicit synthetic journal has no Runtime owner.

    def read(self):
        return tuple(self.events)

    def append(self, event):
        self.events.append(event)


class RequestBudget:
    """One run, one shared request budget across roles AND every rank/Provider retry.

    A persisted reservation with no result is consumed, never silently retried/reset.
    Journal is installed by the owner. Restart loads reservations from that same journal.
    """

    def __init__(self, *, run_id, deadline, max_requests, journal, clock=None):
        if deadline.tzinfo is None or not 1 <= max_requests <= 100:
            raise ProviderError("configuration")
        self.run_id, self.deadline, self.max_requests = run_id, deadline, max_requests
        self.journal = journal
        self.clock = clock or (lambda: datetime.now(UTC))
        self.lock = threading.Lock()
        journal.validate_binding(run_id, deadline)
        binding = dict(run_id=run_id, deadline=deadline.isoformat(), max_requests=max_requests)
        with journal.session():
            events = journal.read()
            if events:
                if events[0] != {"schema_version": "agent-budget/1.0.0", "binding": binding}:
                    raise ProviderError("scope")
            else:
                journal.append({"schema_version": "agent-budget/1.0.0", "binding": binding})

    @property
    def used(self):
        return sum(e.get("phase") == "reserved" for e in self.journal.read())

    @property
    def effective_deadline(self):
        """A composition may narrow request time without rewriting the run binding."""
        return self.deadline

    def reserve(self, scope, deadline, config_checksum):
        with self.lock, self.journal.session():
            self.before_reserve(config_checksum)
            if scope.run_id != self.run_id:
                raise ProviderError("scope")
            until = min(deadline, self.deadline, self.effective_deadline)
            if self.clock() >= until:
                raise ProviderError("deadline")
            if self.used >= self.max_requests:
                raise ProviderError("budget")
            n = self.used + 1
            request_id = "request:" + digest([self.run_id, n])[7:]
            self.journal.append(
                dict(
                    schema_version="agent-request-reservation/1.0.0",
                    phase="reserved",
                    request_id=request_id,
                    scope_checksum=digest(scope),
                    config_checksum=config_checksum,
                )
            )
            return request_id, n, until

    def before_reserve(self, config_checksum):
        """Optional stricter assembly guard, under the same durable reservation lock."""

    def before_wire(self, config_checksum, scope, role, body):
        """Optional owner-installed frozen public experiment binding."""

    def check_control(self):
        """Bounded live assemblies override; legacy Fake behavior is unchanged."""

    def record_response(self, record, reply):
        """Only an explicitly installed bounded research budget retains untrusted output."""

    def record(self, record, *, response_confirmed=None):
        with self.lock, self.journal.session():
            self.journal.append(dict(phase="completed", record=record.model_dump(mode="json")))
            if not record.simulated:
                self.journal.append(
                    dict(
                        phase="response_confirmation",
                        request_id=record.request_id,
                        confirmed=response_confirmed,
                    )
                )

    def transport_counts(self):
        """Local record completion, remote response, usage, and billing are distinct facts."""
        records = [r for r in self.records() if not r.simulated]
        confirmations = {
            e["request_id"]: e["confirmed"]
            for e in self.journal.read()
            if e.get("phase") == "response_confirmation"
        }
        pending = self.used - len(self.records())
        return dict(
            reserved_attempts=self.used,
            local_completed_records=len(self.records()),
            dispatched_observed=sum(r.api_requests or 0 for r in records),
            dispatch_unknown=sum(r.api_requests is None for r in records) + pending,
            remote_response_confirmed=sum(confirmations.get(r.request_id) is True for r in records),
            remote_response_unknown=sum(confirmations.get(r.request_id) is None for r in records)
            + pending,
            usage_observed=sum(r.usage.status == "observed" for r in records),
            usage_unknown=sum(r.usage.status == "unknown" for r in records) + pending,
            cost="unknown",
        )

    def summary(self):
        records = self.records()
        pending = self.used - len(records)
        return RequestSummary(
            run_id=self.run_id,
            reservations=self.used,
            completed=len(records),
            pending_unknown=pending,
            api_requests_observed=sum(r.api_requests or 0 for r in records),
            api_requests_unknown=sum(r.api_requests is None for r in records) + pending,
            simulated_attempts=sum(r.simulated for r in records),
            usage_unknown=sum(r.usage.status == "unknown" for r in records),
        )

    def records(self):
        return tuple(
            read_request_record(e["record"])
            for e in self.journal.read()
            if e.get("phase") == "completed"
        )


def strict_json(raw):
    def pairs(items):
        obj = {}
        for k, v in items:
            if k in obj:
                raise ValueError("duplicate_json_key")
            obj[k] = v
        return obj

    return json.loads(
        raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError())
    )


class StructuredProvider:
    def __init__(
        self,
        config,
        *,
        transport=None,
        live_opt_in=False,
        clock=None,
        sleep=time.sleep,
        egress=None,
    ):
        try:
            self.config = read_provider_config(config)
        except (ValueError, TypeError):
            raise ProviderError("configuration") from None
        config = self.config
        if config.mode == "live" and not live_opt_in:
            raise ProviderError("configuration")
        if transport is None:
            transport = (
                DeterministicFakeTransport()
                if config.mode == "fake"
                else HTTPTransport(
                    network_route=getattr(config, "network_route", None),
                    http_client=getattr(config, "http_client", None),
                )
            )
        if (config.mode == "fake") != transport.simulated:
            raise ProviderError("configuration")
        if isinstance(transport, HTTPTransport) and getattr(
            transport, "network_route", None
        ) != getattr(config, "network_route", None):
            raise ProviderError("configuration")
        if isinstance(transport, HTTPTransport) and transport.http_client != getattr(
            config, "http_client", None
        ):
            raise ProviderError("configuration")
        self.transport = transport
        # Opt-in platform boundary; old CLI configuration/wire and grants stay unchanged.
        self.egress = egress
        self.clock = clock or (lambda: datetime.now(UTC))
        self.sleep = sleep
        self.semaphore = threading.BoundedSemaphore(config.max_concurrency)

    def generate_structured(
        self,
        *,
        role,
        input_payload,
        output_schema,
        scope,
        budget,
        deadline,
        images=(),
        admit=None,
        before_request=None,
    ):
        try:
            scope = validate(Scope, scope)
            version = (
                input_payload.get("schema_version")
                if isinstance(input_payload, dict)
                else getattr(input_payload, "schema_version", None)
                if isinstance(input_payload, RolePayload)
                else None
            )
            if version in {
                "input-scope-expert-input/1.0.0",
                "input-scope-expert-input/1.1.0",
                "input-scope-expert-input/1.2.0",
            }:
                from .input_scope import (
                    ScopeExpertPayload,
                    ScopeExpertPayloadV2,
                    ScopeExpertPayloadV3,
                )

                cls = {
                    "input-scope-expert-input/1.0.0": ScopeExpertPayload,
                    "input-scope-expert-input/1.1.0": ScopeExpertPayloadV2,
                    "input-scope-expert-input/1.2.0": ScopeExpertPayloadV3,
                }[version]
                projected = validate(cls, input_payload)
            elif version == "input-scope-vision-input/1.0.0":
                from .input_scope import ScopeVisionPayload

                projected = validate(ScopeVisionPayload, input_payload)
            elif version == "targeted-vision-input/1.0.0":
                from packages.evidence_acquisition.models import TargetedVisionPayload

                projected = validate(TargetedVisionPayload, input_payload)
            else:
                projected = validate(RolePayload, input_payload)
            if projected.role != role or projected.material.scope != scope:
                raise ValueError()
        except (ValueError, TypeError):
            raise ProviderError("scope") from None
        return self._generate(
            role=role,
            input_payload=input_payload,
            output_schema=output_schema,
            scope=scope,
            budget=budget,
            deadline=deadline,
            images=images,
            admit=admit,
            before_request=before_request,
            projected=projected,
        )

    def generate_review_batch(
        self,
        *,
        input_payload,
        scope,
        budget,
        deadline,
        admit=None,
        before_request=None,
        capture=None,
    ):
        from packages.evidence_details.review_batches import ReviewBatchPayload, proposal_type

        if getattr(self.config, "review_batching", None) is None:
            raise ProviderError("configuration")
        projected = validate(ReviewBatchPayload, input_payload)
        scope = validate(Scope, scope)
        if projected.material.scope != scope:
            raise ProviderError("scope")
        if len(projected.review_tasks.tasks) > self.config.review_batching.batch_size:
            raise ProviderError("configuration")
        return self._generate(
            role="review",
            input_payload=input_payload,
            output_schema=proposal_type(self.config),
            scope=scope,
            budget=budget,
            deadline=deadline,
            images=(),
            admit=admit,
            before_request=before_request,
            projected=projected,
            capture=capture,
        )

    def generate_memory_review(
        self, *, input_payload, scope, budget, deadline, admit=None, before_request=None
    ):
        """Explicit 10D Fake-only domain dispatch; old RolePayload remains frozen."""
        from packages.memory_review.models import Advice, ReviewInput

        if not self.transport.simulated or self.config.mode != "fake":
            raise ProviderError("configuration")
        scope = validate(Scope, scope)
        from packages.memory_review.task_models import qualification

        projected = qualification(ReviewInput, input_payload)
        if projected.scope != scope.model_dump(mode="json"):
            raise ProviderError("scope")
        return self._generate(
            role="expert",
            input_payload=projected,
            output_schema=Advice,
            scope=scope,
            budget=budget,
            deadline=deadline,
            images=(),
            admit=admit,
            before_request=before_request,
            projected=None,
            memory_review=True,
        )

    def generate_definitions(self, *, payload, scope, budget, before_request, capture=None):
        from packages.evidence_reuse.definitions import (
            ConfirmedInterpretationProposal,
            DefinitionProposal,
            EvidenceDefinitionProposal,
            OwnerBoundDefinitionProposal,
            QuotedDefinitionProposal,
            ScopedDefinitionProposal,
        )

        if next(r for r in self.config.roles if r.role == "source").prompt_version not in {
            "definition-draft-2.0.0",
            "definition-draft-2.1.0",
            "definition-draft-2.2.0",
            "definition-draft-2.3.0",
            "definition-draft-3.0.0",
            "definition-draft-4.0.0",
            "definition-draft-4.1.0",
            "definition-draft-5.0.0",
            "definition-draft-6.0.0",
            "definition-draft-6.1.0",
            "definition-draft-6.2.0",
            "definition-draft-6.3.0",
            "definition-draft-6.4.0",
            "definition-draft-6.5.0",
            "definition-draft-6.6.0",
            "definition-draft-6.7.0",
            "definition-draft-6.8.0",
        }:
            raise ProviderError("configuration")
        return self._generate(
            role="source",
            input_payload=payload,
            output_schema=OwnerBoundDefinitionProposal
            if next(r for r in self.config.roles if r.role == "source").prompt_version
            in {
                "definition-draft-6.0.0",
                "definition-draft-6.1.0",
                "definition-draft-6.2.0",
                "definition-draft-6.3.0",
                "definition-draft-6.4.0",
                "definition-draft-6.5.0",
                "definition-draft-6.6.0",
                "definition-draft-6.7.0",
                "definition-draft-6.8.0",
            }
            else ConfirmedInterpretationProposal
            if next(r for r in self.config.roles if r.role == "source").prompt_version
            == "definition-draft-5.0.0"
            else EvidenceDefinitionProposal
            if next(r for r in self.config.roles if r.role == "source").prompt_version
            in {"definition-draft-4.0.0", "definition-draft-4.1.0"}
            else ScopedDefinitionProposal
            if next(r for r in self.config.roles if r.role == "source").prompt_version
            == "definition-draft-3.0.0"
            else QuotedDefinitionProposal
            if next(r for r in self.config.roles if r.role == "source").prompt_version
            == "definition-draft-2.3.0"
            else DefinitionProposal,
            scope=scope,
            budget=budget,
            deadline=budget.deadline,
            images=(),
            admit=None,
            before_request=before_request,
            projected=None,
            capture=capture,
        )

    def _generate(
        self,
        *,
        role,
        input_payload,
        output_schema,
        scope,
        budget,
        deadline,
        images,
        admit,
        before_request,
        projected,
        memory_review=False,
        capture=None,
    ):
        if not self.transport.simulated and not budget.journal.durable:
            raise ProviderError("configuration")
        setting, text = prompt(self.config, role)
        if memory_review:
            text = (
                "Review this synthetic Memory binding and source checksums. Return only "
                "candidate_supported, candidate_rejected or needs_more_cases with reason codes. "
                "No publication, current patient evidence or clinical approval is authorized."
            )
            setting = setting.model_copy(
                update={
                    "prompt_version": "memory-review-1.0.0",
                    "prompt_checksum": digest(text),
                    "images": False,
                }
            )
        if budget.max_requests > self.config.max_requests:
            raise ProviderError("configuration")
        if len(canonical(input_payload)) > setting.max_input_bytes:
            raise ProviderError("schema")
        if images and (not setting.images or not self.config.image_capable):
            raise ProviderError("unsupported_image")
        if role == "vision" and not images:
            raise ProviderError("unsupported_image")
        expected_images = tuple(
            o.image
            for o in (projected.material.observations if projected is not None else ())
            if o.image is not None
        )
        if images and (role != "vision" or tuple(b for b, _ in images) != expected_images):
            raise ProviderError("scope")
        from .images import validate_image

        for binding, raw in images:
            try:
                validate_image(binding, raw)
            except ValueError:
                raise ProviderError("scope") from None
        until = min(deadline, budget.deadline, budget.effective_deadline)
        remaining = (until - self.clock()).total_seconds()
        if remaining <= 0:
            raise ProviderError("deadline")
        if not self.semaphore.acquire(timeout=remaining):
            raise ProviderError("deadline")
        try:
            vision_binding = None
            wire_schema = output_schema.model_json_schema()
            from . import vision_correction
            from .vision_content import PROMPT
            from .vision_content import prepare as prepare_vision_content
            from .vision_projection import PROMPT as COMPACT_PROMPT
            from .vision_projection import project as project_vision

            compact_vision = role == "vision" and setting.prompt_version in {
                COMPACT_PROMPT,
                vision_correction.PROMPT,
            }
            if role == "vision" and setting.prompt_version in {
                PROMPT,
                COMPACT_PROMPT,
                vision_correction.PROMPT,
            }:
                binding, code, wire_schema = prepare_vision_content(scope, images, output_schema)
                vision_binding = (binding, code)
                text += "\nOWNER_BINDING_CODE: " + code
            from . import definition_identity

            definition_identity_enabled = (
                role == "source" and setting.prompt_version in definition_identity.PROMPTS
            )
            wire_payload = input_payload
            if definition_identity_enabled:
                wire_payload, wire_schema = definition_identity.project(input_payload, wire_schema)
            from . import definition_format, definition_passages

            definition_format_enabled = role == "source" and setting.prompt_version in {
                definition_format.PROMPT,
                definition_passages.PROMPT,
            }
            if definition_format_enabled:
                wire_payload = definition_format.project(wire_payload)
            passage_selection = (
                role == "source" and setting.prompt_version == definition_passages.PROMPT
            )
            if passage_selection:
                wire_payload = definition_passages.project(wire_payload)
                wire_schema = definition_passages.SelectedDefinitionProposal.model_json_schema()
            body = wire_request(
                self.config.transport,
                setting,
                text,
                project_vision(input_payload) if compact_vision else wire_payload,
                wire_schema,
                images,
                scope_binding=digest(scope) if not self.transport.simulated else None,
                wire_policy=getattr(self.config, "wire_policy", None),
                wire_mode=getattr(self.config, "wire_mode", None),
            )
            from .source_snippets import project_wire

            body = project_wire(body, self.config, role, projected)
            budget.before_wire(digest(self.config), scope, role, body)
            invocation_id = None
            from .definition_correction import enabled as definition_feedback_enabled

            format_budget = (
                getattr(self.config, "format_correction_budget", 0)
                if role == "review"
                or definition_feedback_enabled(self.config, role)
                or vision_correction.enabled(self.config, role)
                else 0
            )
            correction_sent = False
            for retry in range(self.config.transport.max_retries + 1 + format_budget):
                if retry and getattr(self.config, "max_extra_requests", None) is not None:
                    if (
                        sum(r.retry_index > 0 for r in budget.records())
                        >= self.config.max_extra_requests
                    ):
                        raise ProviderError("budget")
                if before_request is not None:
                    before_request()
                egress_request = None
                egress_checks = []
                if self.egress is not None:
                    from packages.object_store import checksum
                    from packages.platform_security import EgressRequest

                    egress_request = EgressRequest(
                        scope=scope,
                        role=role,
                        config_checksum=digest(self.config),
                        payload_checksum=digest(input_payload),
                        wire_checksum=digest(body),
                        wire_bytes=len(canonical(body)),
                        image_checksums=tuple(checksum(raw) for _, raw in images),
                    )
                    egress_checks.append(self.egress.check(egress_request))
                request_id, n, until = budget.reserve(scope, until, digest(self.config))
                if invocation_id is None:
                    invocation_id = request_id
                start = self.clock()
                timeout = min(
                    self.config.transport.timeout_seconds, (until - start).total_seconds()
                )
                if timeout <= 0:
                    reply = Reply(error="deadline", api_requests=0)
                else:
                    # Reservation and dispatch are distinct. A stopped reservation never sends.
                    try:
                        budget.check_control()
                        if egress_request is not None:
                            egress_checks.append(self.egress.check(egress_request))
                    except Exception as control_error:
                        reply = Reply(
                            error="configuration", api_requests=0, response_confirmed=False
                        )
                        # Preserve the original technical/control exception after recording no-send.
                        dispatch_error = control_error
                    else:
                        dispatch_error = None
                    try:
                        if dispatch_error is None:
                            reply = self.transport.send(
                                self.config.transport,
                                body,
                                timeout=timeout,
                                max_bytes=setting.max_output_bytes + 65536,
                                **(
                                    dict(effective_http_timeout=max(timeout / 2, timeout - 1.0))
                                    if getattr(self.config, "review_batching", None)
                                    and isinstance(self.transport, HTTPTransport)
                                    else {}
                                ),
                                **(
                                    dict(timing_events=True)
                                    # Diagnostics belong to the transport boundary, not a
                                    # particular medical prompt generation. In particular,
                                    # legacy batch Review must retain a safe failure reason.
                                    # This changes no wire body, deadline or retry decision.
                                    if isinstance(self.transport, HTTPTransport)
                                    else {}
                                ),
                            )
                    except Exception:
                        reply = Reply(error="transport", api_requests=None)
                http_observation = (
                    self.transport.observation(reply)
                    if isinstance(self.transport, HTTPTransport)
                    else None
                )
                if getattr(self.config, "recovery", None) and http_observation is not None:
                    with budget.journal.session():
                        budget.journal.append(
                            dict(
                                http_observation,
                                request_id=request_id,
                                config_checksum=digest(self.config),
                                prompt_checksum=setting.prompt_checksum,
                            )
                        )
                error = reply.error
                if getattr(self.config, "wire_policy", None):
                    from .http_recovery import format_rejected

                    if format_rejected(http_observation):
                        error = "configuration"
                result = None
                owner_error = None
                rejection_fields = ()
                if egress_request is not None and timeout > 0 and dispatch_error is None:
                    try:
                        egress_checks.append(self.egress.check(egress_request))
                    except Exception as exc:
                        # Account for the completed attempt, but never accept or retry it.
                        error = "configuration"
                        owner_error = exc
                schema_failures = []
                if len(reply.content) > setting.max_output_bytes:
                    error = "truncated"
                if self.clock() >= until:
                    error = "deadline"
                if error is None:
                    try:
                        from packages.evidence_details.review_batches import ReviewBatchProposal

                        if vision_binding is not None:
                            from .vision_content import decode as decode_vision_content

                            result, proof = decode_vision_content(reply.content, *vision_binding)
                            with budget.journal.session():
                                budget.journal.append(dict(proof, request_id=request_id))
                        elif passage_selection:
                            result, proof = definition_passages.decode(
                                strict_json(reply.content), input_payload
                            )
                            with budget.journal.session():
                                budget.journal.append(dict(proof, request_id=request_id))
                        elif issubclass(output_schema, ReviewBatchProposal):
                            from .review_response import decode

                            result, _ = decode(reply.content, self.config)
                        else:
                            result = validate(output_schema, strict_json(reply.content))
                        if definition_identity_enabled:
                            schema_failures = definition_identity.failures(result, input_payload)
                            if schema_failures:
                                error = "schema"
                    except Exception as schema_error:
                        from .review_correction import fields

                        schema_failures = fields(
                            schema_error,
                            rating_constraints=setting.prompt_version
                            in {"review-direction-2.9.0", "review-direction-2.10.0"},
                        )
                        if definition_format_enabled:
                            try:
                                raw_definition = strict_json(reply.content)
                            except ValueError:
                                raw_definition = None
                            schema_failures = definition_format.fields(schema_error, raw_definition)
                        error = "schema"
                    if error is None and admit is not None:
                        try:
                            admit(result)
                        except Exception as exc:
                            error = "scope"
                            if isinstance(exc, ProviderError) and exc.code == "scope":
                                rejection_fields = exc.fields
                            if not self.transport.simulated:
                                if isinstance(exc, ProviderError) and exc.code == "scope":
                                    if (
                                        self.config.schema_version
                                        in {
                                            "agent-provider-config/7.0.0",
                                            "agent-provider-config/7.1.0",
                                        }
                                        and role == "source"
                                        and exc.fields == ("quote",)
                                    ):
                                        schema_failures = [
                                            {"path": ["quote"], "type": "quote_not_verbatim"}
                                        ]
                                    diagnostic = dict(
                                        category="business_binding_mismatch",
                                        fields=list(exc.fields),
                                    )
                                else:
                                    error = "configuration"
                                    owner_error = exc
                                    diagnostic = dict(
                                        category="owner_validation_error",
                                        exception_type=type(exc).__name__,
                                    )
                                reply = replace(
                                    reply,
                                    diagnostic={
                                        **(reply.diagnostic or {}),
                                        "candidate_validation": diagnostic,
                                    },
                                )
                if self.clock() >= until:
                    error = "deadline"
                record = RequestRecord(
                    request_id=request_id,
                    invocation_id=invocation_id,
                    retry_index=retry,
                    role=role,
                    scope=scope,
                    provider="deterministic_fake"
                    if self.transport.simulated
                    else self.config.transport.provider,
                    model="offline_fixture"
                    if self.transport.simulated
                    else self.config.transport.model,
                    endpoint=self.config.transport.endpoint,
                    config_checksum=digest(self.config),
                    prompt_version=setting.prompt_version,
                    prompt_checksum=setting.prompt_checksum,
                    input_checksum=digest(input_payload),
                    output_checksum=digest(result)
                    if result is not None and error is None
                    else digest(reply.content.hex())
                    if reply.content
                    else None,
                    remote_request_checksum=reply.request_checksum,
                    attempt=n,
                    simulated=self.transport.simulated,
                    api_requests=0 if self.transport.simulated else reply.api_requests,
                    usage=Usage() if self.transport.simulated else reply.usage,
                    latency_seconds=max(0.0, (self.clock() - start).total_seconds()),
                    error=error,
                    deadline=until,
                )
                budget.record(record, response_confirmed=reply.response_confirmed)
                if egress_request is not None:
                    with budget.journal.session():
                        budget.journal.append(
                            dict(
                                schema_version="platform-egress-check/1.0.0",
                                request_id=request_id,
                                request_checksum=digest(egress_request),
                                approved_checks=[
                                    dict(
                                        reference=a.reference.model_dump(mode="json"),
                                        policy_checksum=a.policy_checksum,
                                        data_class=a.data_class,
                                    )
                                    for a in egress_checks
                                ],
                                response_eligible=error is None,
                                current_authorization=False,
                            )
                        )
                if capture is not None:
                    capture(record, reply)
                budget.record_response(record, reply)
                if error == "schema" and (
                    vision_correction.enabled(self.config, role) or definition_format_enabled
                ):
                    with budget.journal.session():
                        budget.journal.append(
                            dict(
                                schema_version="source-definition-contract-diagnostic/1.0.0"
                                if definition_format_enabled
                                else "vision-contract-diagnostic/1.0.0",
                                request_id=request_id,
                                failures=schema_failures,
                                candidate_accepted=False,
                            )
                        )
                if timeout > 0 and dispatch_error is not None:
                    raise dispatch_error
                if reply.diagnostic is not None:
                    with budget.journal.session():
                        budget.journal.append(
                            dict(
                                schema_version="agent-transport-diagnostic/1.0.0",
                                request_id=request_id,
                                value=reply.diagnostic,
                            )
                        )
                if owner_error is not None:
                    raise owner_error
                if error is None:
                    budget.check_control()  # Keep confirmed response/usage, deny new acceptance.
                    return result
                from . import definition_correction
                from .review_correction import corrected_wire, eligible

                definition_feedback = definition_correction.eligible(
                    self.config, scope, role, retry, error, reply, http_observation
                )
                if (
                    definition_feedback
                    or vision_correction.eligible(
                        self.config,
                        scope,
                        role,
                        retry,
                        error,
                        reply,
                        http_observation,
                        schema_failures,
                    )
                    or eligible(
                        self.config,
                        scope,
                        role,
                        retry,
                        error,
                        reply,
                        http_observation,
                        schema_failures,
                    )
                ) and (self.transport.simulated or http_observation is not None):
                    corrected = (
                        definition_correction.corrected_wire(body, schema_failures)
                        if definition_feedback
                        else corrected_wire(body, schema_failures, role=role)
                    )
                    budget.check_control()
                    budget.before_wire(digest(self.config), scope, role, corrected)
                    with budget.journal.session():
                        budget.journal.append(
                            dict(
                                schema_version=(
                                    "source-definition-schema-correction/1.0.0"
                                    if definition_feedback
                                    else "source-quote-correction/1.0.0"
                                    if role == "source"
                                    else "vision-state-correction/1.0.0"
                                    if role == "vision"
                                    else "review-schema-correction/1.0.0"
                                ),
                                invocation_id=invocation_id,
                                failed_request_id=record.request_id,
                                failed_record_checksum=digest(record),
                                input_checksum=digest(input_payload),
                                failures=schema_failures,
                                original_wire_checksum=digest(body),
                                corrected_wire_checksum=digest(corrected),
                                max_corrections_this_invocation=1,
                            )
                        )
                    body = corrected
                    correction_sent = True
                    continue
                if (
                    error not in {"rate_limit", "timeout", "transport"}
                    or retry >= self.config.transport.max_retries
                    or (format_budget and correction_sent)
                ):
                    raise ProviderError(error, fields=rejection_fields if error == "scope" else ())
                if getattr(self.config, "recovery", None):
                    from .http_recovery import retry_delay, retryable

                    if not retryable(http_observation):
                        raise ProviderError(error)
                    delay = retry_delay(
                        http_observation, now=self.clock(), default=self.config.retry_delay_seconds
                    )
                    # Do not occupy this process for a long remote backoff.
                    if delay > 5:
                        raise ProviderError(error)
                else:
                    # Legacy behavior remains frozen.
                    if (
                        not self.transport.simulated
                        and reply.api_requests != 0
                        and error != "rate_limit"
                    ):
                        raise ProviderError(error)
                    delay = self.config.retry_delay_seconds
                if self.clock().timestamp() + delay >= until.timestamp():
                    raise ProviderError("deadline")
                budget.check_control()
                self.sleep(delay)
                budget.check_control()
            raise ProviderError("no_result")
        finally:
            self.semaphore.release()


class OpenAIResponsesProvider(StructuredProvider):
    def __init__(self, config, **options):
        if config.transport.provider != "openai" or config.transport.endpoint != "responses":
            raise ProviderError("configuration")
        super().__init__(config, **options)


class OpenAICompatibleChatProvider(StructuredProvider):
    def __init__(self, config, **options):
        if (
            config.transport.provider != "openai_compatible"
            or config.transport.endpoint != "chat_completions"
        ):
            raise ProviderError("configuration")
        super().__init__(config, **options)


class DeterministicFakeProvider(StructuredProvider):
    def __init__(self, config, replies=(), **options):
        if config.mode != "fake":
            raise ProviderError("configuration")
        super().__init__(config, transport=DeterministicFakeTransport(replies), **options)
