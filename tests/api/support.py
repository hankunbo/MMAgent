import asyncio

import httpx

from apps.api.app import create_app
from apps.api.assembly import Dependencies
from apps.api.auth import LocalIdentities, Principal
from db import manage
from db.runtime.repository.research_api import ResearchAPIRepository


def identities(*, cases=(), datasets=()):
    return LocalIdentities(
        {
            role: Principal(
                "SOURCE:SYNTHETIC_" + role.upper(),
                role,
                frozenset(cases),
                frozenset(datasets),
                synthetic=True,
            )
            for role in ("researcher", "read_only", "restricted_agent", "offline_reviewer")
        }
    )


class HTTP:
    def __init__(self, dependencies, role="researcher"):
        self.app, self.role = create_app(dependencies), role

    def request(self, method, path, **kwargs):
        async def send():
            transport = httpx.ASGITransport(app=self.app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://synthetic.local"
            ) as c:
                kwargs.setdefault("headers", {"Authorization": "Bearer " + self.role})
                return await c.request(method, path, **kwargs)

        return asyncio.run(send())

    def get(self, path, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path, body, **kwargs):
        return self.request("POST", path, json=body, **kwargs)


def dependencies(driver, request, **extra):
    return Dependencies(
        identities=identities(cases=[request.scope.case_code]),
        runtime=ResearchAPIRepository(manage.test_target("runtime")),
        research=driver.research,
        **extra,
    )


def create_body(req, key="api-create"):
    return dict(
        api_version="research-http/1.0.0",
        idempotency_key=key,
        case_code=req.scope.case_code,
        run_id=req.lock.run_id,
        request=dict(code=req.code, version=req.version, checksum=req.checksum),
        expected_version=req.checksum,
        mode="create_only",
    )


def error(response, code, status):
    assert response.status_code == status, response.text
    value = response.json()
    assert value["error"]["error_code"] == code, value
    assert not value["error"]["automatic_retry"]
    assert value["trace_id"] and response.headers["x-trace-id"] == value["trace_id"]
    return value
