"""FastAPI application factory; no global app, sockets, worker or implicit credentials."""

from typing import Annotated, Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.exceptions import HTTPException

from apps.api.assembly import Baseline, Dependencies
from apps.api.auth import require_role
from apps.api.errors import MATRIX, Fault, translate
from apps.api.models import (
    ArtifactImport,
    CaseImport,
    ControlWrite,
    Envelope,
    ErrorDetail,
    ErrorEnvelope,
    ExpertRequest,
    ExtensionConsume,
    OfflineAppend,
    OfflineVote,
    RunCreate,
)
from apps.api.readers import render_bound_json
from apps.api.service import ResearchAPI, installed
from packages.grounding.models import Code, Token
from packages.task_runtime.expert_models import ExpertDispatchRequest


def failure(trace, error):
    status, retry, message = MATRIX[error.code]
    data = ErrorEnvelope(
        trace_id=trace,
        error=ErrorDetail(
            error_code=error.code, message=message, retry=retry, commit_state=error.commit_state
        ),
    )
    headers = {"X-Trace-ID": trace, "Cache-Control": "no-store"}
    if status == 401:
        headers["WWW-Authenticate"] = "Bearer"
    return JSONResponse(status_code=status, content=data.model_dump(mode="json"), headers=headers)


class Limits:
    def __init__(self, app, baseline):
        self.app, self.baseline = app, baseline

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        trace = uuid4().hex
        scope.setdefault("state", {})["trace_id"] = trace
        headers = dict(scope["headers"])
        if scope["method"] in {"POST", "PUT", "PATCH"}:
            if headers.get(b"content-type", b"").split(b";")[0] != b"application/json":
                return await failure(trace, Fault("MEDIA_TYPE_UNSUPPORTED"))(scope, receive, send)
        chunks, size = [], 0
        while True:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                return  # No domain invocation has occurred.
            size += len(msg.get("body", b""))
            if size > self.baseline.max_request_bytes:
                return await failure(trace, Fault("REQUEST_TOO_LARGE"))(scope, receive, send)
            chunks.append(msg.get("body", b""))
            if not msg.get("more_body", False):
                break
        consumed = False

        async def replay():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()

        async def traced(msg):
            if msg["type"] == "http.response.start":
                msg["headers"] = [
                    *(
                        pair
                        for pair in msg.get("headers", [])
                        if pair[0].lower() not in {b"x-trace-id", b"cache-control"}
                    ),
                    (b"x-trace-id", trace.encode()),
                    (b"cache-control", b"no-store"),
                ]
            await send(msg)

        await self.app(scope, replay, traced)


def create_app(dependencies: Dependencies, baseline: Baseline | None = None):
    baseline = baseline or Baseline.load()
    if baseline.default_page_size > baseline.max_page_size:
        raise ValueError("api_page_baseline")
    api = ResearchAPI(dependencies, baseline)
    errors = {status: {"model": ErrorEnvelope} for status, _, _ in MATRIX.values()}
    app = FastAPI(
        title="MMAgent Research Offline API",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        responses=errors,
    )
    app.add_middleware(Limits, baseline=baseline)
    bearer = HTTPBearer(auto_error=False)

    def principal(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]):
        return dependencies.identities.authenticate(
            None if credentials is None else f"{credentials.scheme} {credentials.credentials}"
        )

    Auth = Annotated[object, Depends(principal)]
    Page = Annotated[int, Query(ge=1, le=baseline.max_page_size)]

    def response(request, value, version, semantics="historical_read"):
        data = Envelope(
            trace_id=request.state.trace_id,
            semantics=semantics,
            contract_version=version,
            data=jsonable_encoder(value),
        )
        if len(data.model_dump_json().encode()) > baseline.max_response_bytes:
            raise Fault("OWNER_FAILURE")
        return data

    def safe_html(value):
        rendered, sha = render_bound_json(jsonable_encoder(value))
        if len(rendered.encode()) > baseline.max_response_bytes:
            raise Fault("OWNER_FAILURE")
        return HTMLResponse(
            rendered,
            headers={
                "X-Source-Checksum": sha,
                "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.exception_handler(RequestValidationError)
    async def invalid(request, error):
        return failure(request.state.trace_id, Fault("REQUEST_INVALID"))

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        code = "METHOD_NOT_ALLOWED" if error.status_code == 405 else "OBJECT_NOT_FOUND"
        return failure(request.state.trace_id, Fault(code))

    @app.exception_handler(Fault)
    @app.exception_handler(Exception)
    async def owner_error(request, error):
        return failure(
            getattr(request.state, "trace_id", uuid4().hex),
            translate(error, write=request.method == "POST"),
        )

    @app.get("/capabilities", response_model=Envelope)
    def capabilities(request: Request, actor: Auth):
        return response(
            request,
            {
                "baseline": baseline.model_dump(mode="json"),
                "identity": {
                    "actor_code": actor.actor_code,
                    "role": actor.role,
                    "synthetic": actor.synthetic,
                },
                "case_import": "assembled" if dependencies.preparation else "not_ready",
                "run_create": "assembled" if dependencies.research else "not_ready",
                "run_read": "read_only" if dependencies.runtime else "not_ready",
                "planner": "explicit_per_run",
                "graph": "explicit_per_version",
                "dispatch": "synthetic_explicit_per_run" if dependencies.dispatch else "not_ready",
                "formal_graph_commit": "not_ready",
                "expert_request": "synthetic_explicit_per_run"
                if dependencies.expert_requests
                else "not_ready",
                "replay_execution": "not_ready",
                "offline_audit": "append_only" if dependencies.audit_writer else "not_ready",
            },
            baseline.version,
        )

    @app.post("/cases", response_model=Envelope, status_code=201)
    def import_case(body: CaseImport, request: Request, actor: Auth):
        value = api.import_case(actor, body)
        return response(request, value, value.schema_version, "synchronous_result")

    @app.get("/cases/{case_code}", response_model=Envelope)
    def read_case(case_code: Code, request: Request, actor: Auth):
        require_role(actor, "researcher", "read_only", "offline_reviewer")
        case = api.case(actor, case_code)
        return response(
            request,
            dict(
                case_code=case.case_code,
                encounter_code=case.encounter_code,
                goals=case.goals,
                laterality=case.laterality,
            ),
            "CaseBundle/1.0",
        )

    @app.post("/cases/{case_code}/artifacts", response_model=Envelope, status_code=201)
    def register_artifact(case_code: Code, body: ArtifactImport, request: Request, actor: Auth):
        value = api.register_artifact(actor, case_code, body)
        return response(request, value, "Artifact/1.0", "synchronous_result")

    @app.get("/cases/{case_code}/artifacts", response_model=Envelope)
    def artifacts(
        case_code: Code,
        request: Request,
        actor: Auth,
        after: Code | Literal[""] = "",
        limit: Page = baseline.default_page_size,
    ):
        require_role(actor, "researcher", "read_only", "offline_reviewer")
        case = api.case(actor, case_code)
        rows = sorted(
            (a for a in case.artifacts if a.asset_code > after), key=lambda a: a.asset_code
        )
        # Visible metadata only; no object URI, source note or download route.
        items = [
            dict(
                asset_code=a.asset_code,
                artifact_type_code=a.artifact_type_code,
                checksum=a.checksum,
                media_type=a.media_type,
                qc_status=a.qc_status,
            )
            for a in rows[:limit]
        ]
        return response(
            request,
            dict(items=items, next_after=items[-1]["asset_code"] if len(rows) > limit else None),
            "Artifact/1.0",
        )

    @app.post("/runs", response_model=Envelope, status_code=201)
    def create_run(body: RunCreate, request: Request, actor: Auth):
        return response(
            request, api.create_run(actor, body), "research-run/1.0.0", "synchronous_result"
        )

    @app.get("/runs", response_model=Envelope)
    def runs(
        case_code: Code,
        request: Request,
        actor: Auth,
        after: Code | Literal[""] = "",
        limit: Page = baseline.default_page_size,
    ):
        api.case(actor, case_code)
        rows = dependencies.runtime.list_runs(case_code, after=after, limit=limit)
        if any(row.context.synthetic != actor.synthetic for row in rows):
            raise Fault("SCOPE_FORBIDDEN")
        return response(
            request,
            dict(
                items=[api.run_view(row) for row in rows],
                next_after=rows[-1].context.run_id if len(rows) == limit else None,
            ),
            "research-run/1.0.0",
        )

    @app.get("/runs/{run_id}", response_model=Envelope)
    def read_run(run_id: Code, request: Request, actor: Auth):
        return response(request, api.run_view(api.run(actor, run_id)), "research-run/1.0.0")

    @app.post("/runs/{run_id}/cancel", response_model=Envelope)
    def cancel(run_id: Code, body: ControlWrite, request: Request, actor: Auth):
        value = api.control(actor, run_id, body, cancel=True)
        return response(
            request,
            value,
            "planner-control/1.0.0",
            "synchronous_result" if value["confirmed"] else "unconfirmed",
        )

    @app.post("/runs/{run_id}/resume", response_model=Envelope)
    def resume(run_id: Code, body: ControlWrite, request: Request, actor: Auth):
        value = api.control(actor, run_id, body, cancel=False)
        return response(
            request,
            value,
            "planner-control/1.0.0",
            "synchronous_result" if value["confirmed"] else "unconfirmed",
        )

    @app.post("/runs/{run_id}/start", response_model=Envelope)
    def start(run_id: Code, body: ControlWrite, request: Request, actor: Auth):
        value = api.dispatch_start(actor, run_id, body)
        return response(request, value, "research-dispatch/1.0.0", "synchronous_result")

    @app.get("/runs/{run_id}/dispatch", response_model=Envelope)
    def dispatch_read(run_id: Code, request: Request, actor: Auth):
        return response(request, api.dispatch_read(actor, run_id), "research-dispatch/1.0.0")

    @app.post("/runs/{run_id}/dispatch/cancel", response_model=Envelope)
    def dispatch_cancel(run_id: Code, body: ControlWrite, request: Request, actor: Auth):
        # Task cancellation request is durable; it is not a claim of tool termination.
        value = api.dispatch_cancel(actor, run_id, body)
        return response(request, value, "research-dispatch/1.0.0", "synchronous_result")

    @app.post("/runs/{run_id}/replays", response_model=Envelope)
    def replay(run_id: Code, body: ControlWrite, actor: Auth):
        require_role(actor, "researcher")
        api.run(actor, run_id)
        raise Fault("CAPABILITY_NOT_READY")

    @app.get("/runs/{run_id}/planner", response_model=Envelope)
    def planner(run_id: Code, request: Request, actor: Auth):
        api.run(actor, run_id)
        value = installed(dependencies.planners.get(run_id)).report(run_id)
        return response(request, value, "planner-control-report/1.0.0")

    @app.get("/runs/{run_id}/tool-results", response_model=Envelope)
    def tool_results(
        run_id: Code,
        request: Request,
        actor: Auth,
        kind: Literal["attempts", "confirmed"] = "attempts",
    ):
        api.run(actor, run_id)
        if kind == "confirmed":
            value = installed(dependencies.executions.get(run_id)).read(run_id)
            return response(request, value, value.schema_version)
        value = dependencies.runtime.read_memory_execution(run_id)
        return response(request, value, value["schema_version"])

    @app.get("/runs/{run_id}/report", response_model=Envelope)
    def report(
        run_id: Code,
        request: Request,
        actor: Auth,
        kind: Literal["research", "planner"] = "research",
    ):
        api.run(actor, run_id)
        if kind == "planner":
            value = installed(dependencies.planners.get(run_id)).report(run_id)
            return response(request, value, "planner-control-report/1.0.0")
        value = installed(dependencies.reports.get(run_id)).read(run_id)
        return response(request, value, value["version"])

    @app.get("/runs/{run_id}/report.html", response_class=HTMLResponse)
    def report_html(
        run_id: Code,
        request: Request,
        actor: Auth,
        kind: Literal["research", "planner"] = "research",
    ):
        value = report(run_id, request, actor, kind)
        return safe_html(value.data)

    @app.get("/cases/{case_code}/graphs/{asset_code}/{part}", response_model=Envelope)
    def graph(
        case_code: Code,
        asset_code: Code,
        part: Literal["graph", "records", "scores", "html"],
        version: str,
        request: Request,
        actor: Auth,
    ):
        require_role(actor, "researcher", "read_only", "offline_reviewer")
        api.case(actor, case_code)
        reader = installed(dependencies.graphs.get(version))
        saved = reader.read(case_code, asset_code, version)
        if part == "html":
            return safe_html(saved)
        value = saved[part]
        if value is None:
            raise Fault("DATA_NOT_READY")
        return response(request, value, value.get("schema_version", value.get("schema", version)))

    @app.get("/runs/{run_id}/extensions", response_model=Envelope)
    def extensions(
        run_id: Code,
        request: Request,
        actor: Auth,
        after: Code | Literal[""] = "",
        limit: Page = baseline.default_page_size,
    ):
        api.run(actor, run_id)
        return response(
            request, api.extension_page(run_id, after=after, limit=limit), "tool-gap-receipt/1.0.0"
        )

    @app.get("/runs/{run_id}/requirements/{key}", response_model=Envelope)
    def requirement(run_id: Code, key: Token, request: Request, actor: Auth):
        api.run(actor, run_id)
        owner = installed(dependencies.gaps)
        if owner.read(run_id, key) is None:
            raise Fault("OBJECT_NOT_FOUND")
        return response(request, owner.report(run_id, key), "tool-gap-report/1.0.0")

    @app.get("/runs/{run_id}/extensions/{key}", response_model=Envelope)
    def extension(run_id: Code, key: Token, request: Request, actor: Auth):
        value = api.extension(actor, run_id, key)
        return response(request, value, value["schema_version"])

    @app.get("/runs/{run_id}/extensions/{key}/stages/{stage}", response_model=Envelope)
    def extension_stage(
        run_id: Code,
        key: Token,
        stage: Literal["discovery", "adapter"],
        request: Request,
        actor: Auth,
    ):
        api.run(actor, run_id)
        owner = installed(dependencies.discovery if stage == "discovery" else dependencies.adapter)
        saved = owner.read(run_id, key)
        if saved is None:
            raise Fault("OBJECT_NOT_FOUND")
        value = dict(
            status=saved["status"],
            task_id=saved["task_id"],
            result=saved["result"],
            budget_charged=saved["budget_charged"],
            historical_read_not_authorization=True,
            available_next_run=False,
            main_run_waits=False,
        )
        version = (
            "tool-discovery-report/1.1.0"
            if stage == "discovery"
            else "synthetic-adapter-report/1.0.0"
        )
        return response(request, value, version)

    @app.get("/runs/{run_id}/tool-onboarding-jobs", response_model=Envelope)
    @app.get("/runs/{run_id}/formal-graph", response_model=Envelope)
    def missing_formal_owner(run_id: Code, actor: Auth):
        api.run(actor, run_id)
        raise Fault("CAPABILITY_NOT_READY")

    @app.post("/runs/{run_id}/extensions/{key}/consume", response_model=Envelope)
    def consume(run_id: Code, key: Token, body: ExtensionConsume, request: Request, actor: Auth):
        value = api.extension(actor, run_id, key, body)
        return response(request, value, value["schema_version"], "historical_read")

    @app.get("/runs/{run_id}/validations/{key}", response_model=Envelope)
    def validation(run_id: Code, key: Token, request: Request, actor: Auth):
        row = api.run(actor, run_id)
        value = installed(dependencies.validations.get(run_id)).read(key=key, historical=True)
        if (value["run_id"], value["case_code"]) != (run_id, row.context.request.scope.case_code):
            raise ValueError("validation_scope")
        return response(request, value, value["schema_version"])

    @app.get("/runs/{run_id}/counterfactuals/{candidate}", response_model=Envelope)
    def counterfactual(run_id: Code, candidate: Token, request: Request, actor: Auth):
        row = api.run(actor, run_id)
        value = installed(dependencies.counterfactuals.get(run_id)).read(
            candidate_id=candidate, historical=True
        )
        if value is None:
            raise Fault("OBJECT_NOT_FOUND")
        if (value["run_id"], value["case_code"]) != (run_id, row.context.request.scope.case_code):
            raise ValueError("counterfactual_scope")
        return response(request, value, value["schema_version"])

    @app.get("/runs/{run_id}/memory", response_model=Envelope)
    def memory(run_id: Code, request: Request, actor: Auth):
        api.run(actor, run_id)
        owner = installed(dependencies.gaps)
        if owner.memory is None:
            raise Fault("CAPABILITY_NOT_READY")
        value = owner.memory_diagnostics(run_id)
        if value["status"] == "technical_failure":
            raise Fault("OWNER_FAILURE")
        return response(request, value, "memory-diagnostics-owner-read/1.0.0")

    @app.get("/runs/{run_id}/expert-options", response_model=Envelope)
    def expert_options(run_id: Code, request: Request, actor: Auth):
        api.run(actor, run_id)
        port = installed(dependencies.expert_requests.get(run_id))
        if port.run_id != run_id:
            raise Fault("SCOPE_FORBIDDEN")
        value = port.options()
        return response(request, value, value["schema_version"])

    @app.post("/runs/{run_id}/expert", response_model=Envelope)
    def request_expert(
        run_id: Code, body: ExpertRequest | ExpertDispatchRequest, request: Request, actor: Auth
    ):
        require_role(actor, "researcher", "restricted_agent")
        row = api.run(actor, run_id)
        if not isinstance(body, ExpertDispatchRequest):
            raise Fault("CAPABILITY_NOT_READY")
        if body.case_code != row.context.request.scope.case_code:
            raise Fault("SCOPE_FORBIDDEN")
        port = installed(dependencies.expert_requests.get(run_id))
        if port.run_id != run_id:
            raise Fault("SCOPE_FORBIDDEN")
        value = port.submit(body, actor_code=actor.actor_code)
        return response(request, value, value["schema_version"], "synchronous_result")

    @app.get("/runs/{run_id}/expert/{key}", response_model=Envelope)
    def expert(run_id: Code, key: Token, request: Request, actor: Auth):
        from agents.expert_agent.models import read_result
        from packages.contracts import ExpertAssessment

        row = api.run(actor, run_id)
        raw = installed(dependencies.expert_history.get(run_id)).read(key)
        if raw is None:
            raise Fault("OBJECT_NOT_FOUND")
        raw = raw[0]
        if raw.get("contract_type") == "ExpertAssessment" and raw.get("schema_version") == "1.0":
            from packages.grounding.models import validate

            value = validate(ExpertAssessment, raw)
            if (
                value.scope.run_id != run_id
                or value.scope.case_code != row.context.request.scope.case_code
            ):
                raise ValueError("expert_scope")
            version = "ExpertAssessment/1.0"
        elif raw.get("schema_version") in {
            "research-review-result/1.0.0",
            "research-review-result/2.0.0",
            "research-review-result/3.0.0",
        }:
            value = read_result(raw)
            version = value["schema_version"]
            if value["run_id"] != run_id:
                raise ValueError("expert_scope")
        else:
            raise Fault("VERSION_UNSUPPORTED")
        return response(request, value, version)

    @app.get("/runs/{run_id}/audit", response_model=Envelope)
    def audit(
        run_id: Code,
        request: Request,
        actor: Auth,
        after: Annotated[int, Query(ge=0)] = 0,
        limit: Page = baseline.default_page_size,
    ):
        require_role(actor, "researcher", "read_only", "offline_reviewer")
        row = api.run(actor, run_id)
        events = installed(dependencies.audit_reader).list_run(
            row.context.request.scope.case_code, run_id, after=after, limit=limit
        )
        items = [
            dict(
                event_id=e.event_id,
                sequence=e.event_sequence,
                event_type=e.event_type,
                actor=e.actor,
                run_id=e.run_id,
                case_code=e.case_code,
                payload_checksum=e.payload_checksum,
                occurred_at=e.occurred_at,
            )
            for e in events
        ]
        return response(
            request,
            dict(
                items=items, next_after=events[-1].event_sequence if len(events) == limit else None
            ),
            "AuditEventRecord/1.0",
        )

    @app.post("/runs/{run_id}/offline-audit", response_model=Envelope, status_code=201)
    def append_audit(run_id: Code, body: OfflineAppend, request: Request, actor: Auth):
        event = api.offline_append(actor, run_id, body)
        return response(
            request,
            dict(
                event_id=event.event_id, payload_checksum=event.payload_checksum, run_unchanged=True
            ),
            "AuditEventRecord/1.0",
            "synchronous_result",
        )

    @app.get("/runs/{run_id}/offline-reviews/{board}", response_model=Envelope)
    def read_offline_review(run_id: Code, board: Token, request: Request, actor: Auth):
        require_role(actor, "researcher", "read_only", "offline_reviewer")
        row = api.run(actor, run_id)
        port = installed(dependencies.offline_boards.get((run_id, board)))
        if (port.run_id, port.case_code) != (run_id, row.context.request.scope.case_code):
            raise Fault("SCOPE_FORBIDDEN")
        value = port.read()
        return response(request, value, value["schema_version"])

    @app.post("/runs/{run_id}/offline-reviews/{board}", response_model=Envelope)
    def vote_offline_review(
        run_id: Code, board: Token, body: OfflineVote, request: Request, actor: Auth
    ):
        require_role(actor, "offline_reviewer")
        row = api.run(actor, run_id)
        port = installed(dependencies.offline_boards.get((run_id, board)))
        if (port.run_id, port.case_code) != (run_id, row.context.request.scope.case_code):
            raise Fault("SCOPE_FORBIDDEN")
        value = port.vote(actor, body)
        return response(request, value, value["schema_version"], "synchronous_result")

    return app
