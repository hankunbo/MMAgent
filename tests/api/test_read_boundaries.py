"""HTTP must distinguish absent installations, absent data and broken owner reads."""

from dataclasses import replace
from types import SimpleNamespace

from apps.api.assembly import Dependencies
from apps.api.readers import ConfirmedExecutionReader
from packages.grounding.models import digest
from tests.api.support import HTTP, error, identities


def test_http_missing_memory_is_not_technical_failure_or_empty():
    context = SimpleNamespace(
        run_id="RUN:ANON",
        synthetic=True,
        checksum=digest("context"),
        request=SimpleNamespace(scope=SimpleNamespace(case_code="CASE:ANON")),
        schema_version="research-run/1.0.0",
    )
    runtime = SimpleNamespace(
        read_run=lambda _: SimpleNamespace(context=context, status="CREATED"),
        get_case_bundle=lambda _: SimpleNamespace(case_code="CASE:ANON"),
        extension_receipts=lambda *a, **k: (),
    )
    deps = Dependencies(identities(cases=["CASE:ANON"]), runtime=runtime)
    path = "/runs/RUN:ANON"
    for suffix in ("/memory", "/planner", "/formal-graph", "/tool-onboarding-jobs", "/report"):
        error(HTTP(deps).get(path + suffix), "CAPABILITY_NOT_READY", 503)
    empty = SimpleNamespace(memory=object(), memory_diagnostics=lambda _: dict(status="empty"))
    assert HTTP(replace(deps, gaps=empty)).get(path + "/memory").json()["data"] == {
        "status": "empty"
    }
    broken = SimpleNamespace(
        memory=object(), memory_diagnostics=lambda _: dict(status="technical_failure")
    )
    error(HTTP(replace(deps, gaps=broken)).get(path + "/memory"), "OWNER_FAILURE", 503)
    error(HTTP(deps, "restricted_agent").get(path + "/audit"), "SCOPE_FORBIDDEN", 403)
    assert HTTP(deps, "restricted_agent").get(path).status_code == 200
    expert = replace(deps, expert_history={"RUN:ANON": SimpleNamespace(read=lambda _: None)})
    error(HTTP(expert).get(path + "/expert/missing"), "OBJECT_NOT_FOUND", 404)
    plan = SimpleNamespace(document=SimpleNamespace(run_id="RUN:ANON"))
    executor = SimpleNamespace(recover=lambda *a: dict(status="ACTIVE", result=None))
    bound = replace(deps, executions={"RUN:ANON": ConfirmedExecutionReader(executor, plan, "same")})
    r = HTTP(bound).get(path + "/tool-results", params={"kind": "confirmed"})
    assert error(r, "COMMIT_UNKNOWN", 503)["error"]["commit_state"] == "unconfirmed"
