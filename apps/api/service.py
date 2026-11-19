"""Thin local service facade. Original owners retain facts, writes and recovery."""

from datetime import UTC, datetime
from threading import Event as Cancellation

from apps.api.auth import require_role
from apps.api.errors import Fault, translate
from apps.api.models import RunView
from db.audit.repository import AuditEvent
from db.runtime.repository.case_pipeline import identity
from packages.grounding.models import digest
from packages.planner_control.models import Command


def installed(value):
    if value is None:
        raise Fault("CAPABILITY_NOT_READY")
    return value


class ResearchAPI:
    def __init__(self, dependencies, baseline):
        self.d, self.baseline = dependencies, baseline

    def case(self, principal, case_code):
        # Administrative installation does not grant access to patient resources.
        require_role(principal, "researcher", "read_only", "offline_reviewer", "restricted_agent")
        rt = installed(self.d.runtime)
        if case_code not in principal.case_codes:
            if not principal.dataset_codes or self.d.visible is None:
                raise Fault("SCOPE_FORBIDDEN")
            visible = self.d.visible.read(case_code)
            if visible.dataset_ref.code not in principal.dataset_codes:
                raise Fault("SCOPE_FORBIDDEN")
        case = rt.get_case_bundle(case_code)
        if case.case_code != case_code:
            raise ValueError("case_reader_scope")
        return case

    def run(self, principal, run_id):
        row = installed(self.d.runtime).read_run(run_id)
        self.case(principal, row.context.request.scope.case_code)
        if principal.synthetic != row.context.synthetic:
            raise Fault("SCOPE_FORBIDDEN")
        return row

    def import_case(self, principal, body):
        require_role(principal, "researcher")
        if body.dataset_code not in principal.dataset_codes:
            raise Fault("SCOPE_FORBIDDEN")
        source = self.d.case_sources.get(body.source_code)
        if source is None:
            raise Fault("OBJECT_NOT_FOUND")
        if source.dataset_code != body.dataset_code:
            raise Fault("SCOPE_FORBIDDEN")
        if (
            source.checksum != body.expected_version
            or source.sample.fingerprint() != source.checksum
        ):
            raise Fault("VERSION_CONFLICT")
        key = digest(dict(scope=body.dataset_code, key=body.idempotency_key))
        value = installed(self.d.preparation).register(source.sample, key=key)
        visible = value.visible
        if visible.dataset_ref.code != body.dataset_code:
            raise ValueError("import_dataset_scope")
        return self.d.preparation.read_visible(visible.case.case_code)

    def register_artifact(self, principal, case_code, body):
        from packages.llm_provider.images import validate_image
        from packages.object_store import checksum

        require_role(principal, "researcher")
        self.case(principal, case_code)
        source = self.d.artifact_sources.get(body.source_code)
        if source is None:
            raise Fault("OBJECT_NOT_FOUND")
        a = source.artifact
        if a.scope.case_code != case_code:
            raise Fault("SCOPE_FORBIDDEN")
        if body.expected_version != digest(a):
            raise Fault("VERSION_CONFLICT")
        if a.media_type not in self.baseline.allowed_media_types:
            raise Fault("MEDIA_TYPE_UNSUPPORTED")
        raw = installed(self.d.objects).read_bounded(a.uri, self.baseline.max_artifact_bytes)
        if len(raw) != source.byte_size or checksum(raw) != a.checksum:
            raise Fault("VERSION_CONFLICT")
        b = validate_image(source.image_binding, raw)
        if (
            b.reference.code != a.asset_code
            or b.reference.checksum != a.checksum
            or b.media_type != a.media_type
            or tuple(a.dimensions[:2]) != (b.height, b.width)
        ):
            raise ValueError("visible_image_binding")
        key = identity("TASK", digest(dict(case=case_code, key=body.idempotency_key)))
        return self.d.runtime.register_visible_artifact(a, key)

    def create_run(self, principal, body):
        require_role(principal, "researcher")
        self.case(principal, body.case_code)
        if body.mode != "create_only":
            raise Fault("CAPABILITY_NOT_READY")
        service = installed(self.d.research)
        if service.synthetic != principal.synthetic:
            raise Fault("SCOPE_FORBIDDEN")
        r = body.request
        request = service.research.read_request(r.code, r.version, r.checksum)
        if request.scope.case_code != body.case_code or request.lock.run_id != body.run_id:
            raise Fault("SCOPE_FORBIDDEN")
        if body.expected_version != request.checksum:
            raise Fault("VERSION_CONFLICT")
        context = service.create(r.code, r.version, r.checksum, body.idempotency_key)
        # Return the immutable creation receipt, independent of later runtime progress.
        return dict(
            run_id=context.run_id,
            case_code=body.case_code,
            context_checksum=context.checksum,
            synthetic=context.synthetic,
            created=True,
            creation_status="CREATED",
            queued=False,
            executed=False,
            confirmed=True,
            dispatch="not_installed",
            task_id=None,
        )

    def dispatch_start(self, principal, run_id, body):
        from packages.task_runtime.models import DispatchRequest

        require_role(principal, "researcher")
        row = self.run(principal, run_id)
        if body.case_code != row.context.request.scope.case_code:
            raise Fault("SCOPE_FORBIDDEN")
        if body.context_checksum != row.context.checksum:
            raise Fault("VERSION_CONFLICT")
        return installed(self.d.dispatch).submit(
            DispatchRequest(
                run_id=run_id,
                case_code=body.case_code,
                context_checksum=body.context_checksum,
                idempotency_key=body.idempotency_key,
                expected_revision=body.expected_revision,
            )
        )

    def dispatch_read(self, principal, run_id):
        self.run(principal, run_id)
        value = installed(self.d.dispatch).read(run_id)
        if value is None:
            raise Fault("OBJECT_NOT_FOUND")
        return value

    def dispatch_cancel(self, principal, run_id, body):
        require_role(principal, "researcher")
        row = self.run(principal, run_id)
        if body.case_code != row.context.request.scope.case_code:
            raise Fault("SCOPE_FORBIDDEN")
        if body.context_checksum != row.context.checksum:
            raise Fault("VERSION_CONFLICT")
        dispatch = installed(self.d.dispatch)
        current = dispatch.read(run_id)
        if current is None:
            raise Fault("OBJECT_NOT_FOUND")
        # Cancellation idempotence lives at the Task command; expected_revision is Task revision.
        if (
            dispatch.history(run_id, body.idempotency_key) is None
            and current.revision != body.expected_revision
        ):
            raise Fault("VERSION_CONFLICT")
        return dispatch.command(
            run_id, "cancel", body.idempotency_key, expected_revision=body.expected_revision
        )

    def extension_page(self, run_id, *, after="", limit=20):
        rows = self.d.runtime.extension_receipts(run_id, after=after, limit=limit)
        items = [
            dict(
                task_id=r.task_id,
                state=r.state,
                requirement_key=r.application.key,
                version=r.schema_version,
                available_next_run=False,
                historical_read_not_authorization=True,
            )
            for r in rows
        ]
        return dict(
            items=items,
            next_after=items[-1]["task_id"] if len(items) == limit else None,
            main_run_waits=False,
            formal_job="not_installed",
        )

    def run_view(self, row):
        c = row.context
        try:
            extensions = self.extension_page(c.run_id)
        except Exception as error:
            # Main facts remain readable; this is explicitly unavailable, never an empty list.
            extensions = dict(
                availability="owner_error", error_code=translate(error).code, main_run_waits=False
            )
        return RunView(
            run_id=c.run_id,
            case_code=c.request.scope.case_code,
            status=row.status,
            context_version=c.schema_version,
            context_checksum=c.checksum,
            synthetic=c.synthetic,
            extensions=extensions,
        )

    def control(self, principal, run_id, body, *, cancel):
        require_role(principal, "researcher")
        row = self.run(principal, run_id)
        if body.case_code != row.context.request.scope.case_code:
            raise Fault("SCOPE_FORBIDDEN")
        if body.context_checksum != row.context.checksum:
            raise Fault("VERSION_CONFLICT")
        planner = installed(self.d.planners.get(run_id))
        state = planner.read(run_id)
        pending = state.pending
        signal = Cancellation()
        if cancel:
            signal.set()
        if pending:
            command = pending.command
            if (command.key, command.expected_revision) != (
                body.idempotency_key,
                body.expected_revision,
            ):
                raise Fault("VERSION_CONFLICT")
            # An unclaimed intent may be cancelled, but HTTP must never resume its launch.
            if not cancel and not pending.launch_claimed:
                raise Fault("CAPABILITY_NOT_READY")
            if cancel:
                # Original owners re-read under their locks; this snapshot cannot prove no claim.
                planner.observe_recovery_cancel(run_id, command, cancel=signal)
        elif cancel:
            prior = next((e for e in state.events if e.command.key == body.idempotency_key), None)
            if prior is not None and state.cancelled:
                command = prior.command
                if command.expected_revision != body.expected_revision:
                    raise Fault("VERSION_CONFLICT")
            else:
                command = Command(
                    key=body.idempotency_key,
                    expected_revision=body.expected_revision,
                    action="cancel",
                )
        else:
            event = next((e for e in state.events if e.command.key == body.idempotency_key), None)
            if event is None:
                raise Fault("DATA_NOT_READY")
            command = event.command
            if command.expected_revision != body.expected_revision:
                raise Fault("VERSION_CONFLICT")
        result = planner.step(run_id, command, cancel=signal)
        current = planner.read(run_id)
        return dict(
            result=result.model_dump(mode="json"),
            control=planner.report(run_id).model_dump(mode="json"),
            cancelled=current.cancelled,
            revision=current.revision,
            confirmed=current.pending is None,
            retry="reconcile_same_key" if current.pending else "none",
        )

    def offline_append(self, principal, run_id, body):
        require_role(principal, "offline_reviewer")
        row = self.run(principal, run_id)
        if body.case_code != row.context.request.scope.case_code:
            raise Fault("SCOPE_FORBIDDEN")
        if body.expected_version != row.context.checksum:
            raise Fault("VERSION_CONFLICT")
        reader, writer = installed(self.d.audit_reader), installed(self.d.audit_writer)
        key = identity(
            "TASK", digest(dict(run=run_id, actor=principal.actor_code, key=body.idempotency_key))
        )
        event_id = identity("MESSAGE", key)
        payload = dict(
            schema_version="research-offline-note/1.0.0",
            finding=body.finding,
            context_checksum=body.expected_version,
            actor_code=principal.actor_code,
            synthetic=principal.synthetic,
        )
        old = reader.get_event(event_id)
        if old is not None:
            if old.payload != payload or old.run_id != run_id or old.case_code != body.case_code:
                raise Fault("IDEMPOTENCY_CONFLICT")
            return old
        event = AuditEvent(
            event_id=event_id,
            event_type="OFFLINE_REVIEW_RECORDED",
            actor="PLANNER",
            case_code=body.case_code,
            run_id=run_id,
            correlation_code="SOURCE:RESEARCH_HTTP_OFFLINE",
            idempotency_key=key,
            occurred_at=datetime.now(UTC),
            payload=payload,
        )
        try:
            return writer.append_event(event)
        except Exception:
            # Concurrent same-key writers may differ only in their server timestamp.
            old = reader.get_event(event_id)
            if old is not None and old.payload == payload:
                return old
            raise

    def extension(self, principal, run_id, key, body=None):
        row = self.run(principal, run_id)
        binding = installed(self.d.extensions.get(run_id))
        owner = binding.owner
        if body is None:
            value = owner.read(run_id, key)
            if value is None:
                raise Fault("OBJECT_NOT_FOUND")
        else:
            require_role(principal, "researcher")
            if body.case_code != row.context.request.scope.case_code:
                raise Fault("SCOPE_FORBIDDEN")
            if body.idempotency_key != binding.stage_key:
                raise Fault("IDEMPOTENCY_CONFLICT")
            record = owner.adapter.read(run_id, key)
            if record is None:
                raise Fault("DATA_NOT_READY")
            if digest(record) != body.expected_version:
                raise Fault("VERSION_CONFLICT")
            value = owner.consume(run_id, key, stage_key=binding.stage_key, worker=binding.worker)
        # Never expose the saved grant/lease as an authorization token.
        return dict(
            schema_version="synthetic-extension-http-view/1.0.0",
            task_id=value["task_id"],
            status=value["status"],
            result=value["result"],
            budget_charged=value["budget_charged"],
            available_next_run=False,
            historical_read_not_authorization=True,
            main_run_waits=False,
            current_qualification="not_asserted",
            recovery="same_stage_read_only_no_restart",
        )
