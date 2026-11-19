"""Install original owners, export bounded observations, and run existing workers."""

import json
import time
from dataclasses import dataclass, field
from typing import Annotated, Literal

from pydantic import Field, model_validator

from apps.api.app import create_app
from apps.api.assembly import Baseline, Dependencies
from apps.api.service import ResearchAPI
from packages.grounding.models import Value, digest
from packages.planner_control.models import Reference


class MetricBlock(Value):
    schema_version: Literal["platform-metric-block/1.0.0"] = "platform-metric-block/1.0.0"
    run_id: str
    case_code: str
    facet: Literal["llm", "coverage", "memory", "tool_qc", "validation", "extension", "expert"]
    source: Reference
    measurements: dict[str, Annotated[float, Field(allow_inf_nan=False)] | bool | None]

    @model_validator(mode="after")
    def names(self):
        allowed = {
            "llm": {
                "reserved_attempts",
                "local_completed_records",
                "dispatched_observed",
                "dispatch_unknown",
                "remote_response_confirmed",
                "remote_response_unknown",
                "usage_observed",
                "usage_unknown",
                "cost",
                "retry_records",
                "latency_seconds",
            },
            "coverage": {"observation_ratio", "interpretation_ratio", "missing_count"},
            "memory": {"hits", "filtered", "selected"},
            "tool_qc": {"passed", "failed", "unknown"},
            "validation": {"supported", "refuted", "conflicts", "unknown"},
            "extension": {"queued", "running", "failed", "available_next_run"},
            "expert": {"requested", "accepted", "rejected", "unavailable"},
        }
        if not self.measurements.keys() <= allowed[self.facet]:
            raise ValueError("platform_metric_names")
        return self


FACETS = ("llm", "coverage", "memory", "tool_qc", "validation", "extension", "expert")


class Profile(Value):
    schema_version: Literal["local-platform-profile/1.0.0"] = "local-platform-profile/1.0.0"
    workers: Literal[1] = 1
    max_ticks: int = Field(default=16, ge=1, le=64)
    shutdown_seconds: int = Field(default=10, ge=1, le=60)
    keep_alive_seconds: int = Field(default=5, ge=1, le=30)
    concurrency: int = Field(default=8, ge=1, le=32)
    live_enabled: Literal[False] = False


@dataclass(frozen=True)
class Installation:
    dependencies: Dependencies
    profile: Profile = Profile()
    metric_readers: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.dependencies.dispatch is not None:
            owner = self.dependencies.dispatch.owner
            if owner is None or owner.planners != self.dependencies.planners:
                raise ValueError("platform_owner_binding")
            for run_id, planner in owner.planners.items():
                row = self.dependencies.runtime.read_run(run_id)
                state = planner.read(run_id)
                if (
                    not row.context.synthetic
                    or state.binding.run_context_checksum != row.context.checksum
                ):
                    raise ValueError("platform_synthetic_binding")

    def snapshot(self, principal, run_id):
        deps = self.dependencies
        row = ResearchAPI(deps, Baseline.load()).run(principal, run_id)
        planner = deps.planners.get(run_id)
        state = planner.read(run_id) if planner else None
        receipt = deps.dispatch.read(run_id) if deps.dispatch else None
        facets = {}
        for name in FACETS:
            reader = self.metric_readers.get((run_id, name))
            if reader is None:
                facets[name] = dict(status="not_installed", measurements=None)
                continue
            block = MetricBlock.model_validate(reader.read(run_id))
            if (block.run_id, block.case_code, block.facet) != (
                run_id,
                row.context.request.scope.case_code,
                name,
            ):
                raise ValueError("platform_metric_scope")
            facets[name] = dict(status="observed", value=block.model_dump(mode="json"))
        return dict(
            schema_version="platform-observation/1.0.0",
            run_id=run_id,
            case_code=row.context.request.scope.case_code,
            context_checksum=row.context.checksum,
            control_checksum=digest(state) if state else None,
            graph=state.graph.model_dump(mode="json") if state else None,
            locks=[r.model_dump(mode="json") for r in state.binding.locks] if state else [],
            phase=state.phase.value if state else None,
            cancelled=state.cancelled if state else None,
            step_revision=state.revision if state else None,
            planner_steps=state.steps if state else None,
            planner_retries=state.retries if state else None,
            no_progress=state.no_progress if state else None,
            fallback_events=sum(e.after.value == "fallback" for e in state.events)
            if state
            else None,
            last_event_at=state.events[-1].at.isoformat() if state and state.events else None,
            queue_wait_seconds=None,  # original dispatch receipt has no enqueue timestamp
            action_candidates=len(state.candidates) if state else None,
            tool_reservations=state.reserved if state else None,
            tool_started=state.started if state else None,
            tool_confirmed=state.confirmed if state else None,
            dispatch_state=receipt.state if receipt else None,
            claims=receipt.claims if receipt else None,
            dispatch_failures=receipt.failures if receipt else None,
            deadline=state.binding.deadline.isoformat() if state else None,
            available_readers={
                "report": run_id in deps.reports,
                "validation": run_id in deps.validations,
                "counterfactual": run_id in deps.counterfactuals,
                "expert": run_id in deps.expert_history,
                "expansion": deps.gaps is not None,
            },
            facets=facets,
            current_authorization=False,
            clinical_completion=False,
        )

    def app(self):
        from fastapi import Request
        from fastapi.responses import JSONResponse

        app = create_app(self.dependencies)

        @app.get("/platform/runs/{run_id}/operations")
        def operations(run_id: str, request: Request):
            actor = self.dependencies.identities.authenticate(request.headers.get("authorization"))
            return JSONResponse(self.snapshot(actor, run_id))

        @app.get("/platform/runs/{run_id}/operations.html")
        def operations_html(run_id: str, request: Request):
            from fastapi.responses import HTMLResponse

            from apps.api.readers import render_bound_json

            actor = self.dependencies.identities.authenticate(request.headers.get("authorization"))
            rendered, sha = render_bound_json(self.snapshot(actor, run_id))
            return HTMLResponse(
                rendered,
                headers={
                    "X-Source-Checksum": sha,
                    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
                },
            )

        return SafeEvents(app)


class BudgetMetricsReader:
    """Read the original request journal, not Control.api_requests (legacy fixed zero)."""

    def __init__(self, budget, case_code):
        self.budget, self.case_code = budget, case_code

    def read(self, run_id):
        if run_id != self.budget.run_id:
            raise ValueError("platform_metric_scope")
        events = self.budget.journal.read()
        records = self.budget.records()
        if any(r.scope.run_id != run_id or r.scope.case_code != self.case_code for r in records):
            raise ValueError("platform_metric_scope")
        measurements = self.budget.transport_counts()
        measurements["cost"] = None  # original budget explicitly says unknown
        measurements.update(
            retry_records=sum(r.retry_index > 0 for r in records),
            latency_seconds=sum(r.latency_seconds for r in records),
        )
        return MetricBlock(
            run_id=run_id,
            case_code=self.case_code,
            facet="llm",
            source=Reference(
                code="METRIC:AGENT_BUDGET", version="1.0.0", checksum=digest(events), role="audit"
            ),
            measurements=measurements,
        )


class SafeEvents:
    """Fixed HTTP metadata only: never raw path/query/body/header or exception text."""

    def __init__(self, app, sink=None):
        self.app = app
        self.sink = sink or (lambda event: print(json.dumps(event), flush=True))

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        start, status, trace = time.monotonic(), 500, None

        async def observe(message):
            nonlocal status, trace
            if message["type"] == "http.response.start":
                status = message["status"]
                trace = (
                    dict(message.get("headers", []))
                    .get(b"x-trace-id", b"")
                    .decode("ascii", errors="ignore")
                )
            await send(message)

        try:
            await self.app(scope, receive, observe)
        finally:
            self.sink(
                dict(
                    schema_version="platform-http-event/1.0.0",
                    trace_id=trace,
                    http_status=status,
                    elapsed_seconds=round(time.monotonic() - start, 6),
                )
            )
