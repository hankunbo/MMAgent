"""Actual HTTP serialization, validation, authentication and error boundary."""

import pytest

from apps.api.assembly import Dependencies
from apps.api.errors import Fault
from tests.api.support import HTTP, error, identities


def test_http_auth_openapi_limits_and_unready():
    h = HTTP(Dependencies(identities=identities()))
    error(h.get("/capabilities", headers={"X-Role": "admin"}), "AUTHENTICATION_REQUIRED", 401)
    error(
        h.get("/capabilities", headers={"Authorization": "Bearer forged"}),
        "AUTHENTICATION_REQUIRED",
        401,
    )
    result = h.get("/capabilities").json()
    assert result["data"]["identity"]["synthetic"]
    assert result["data"]["dispatch"] == "not_ready"
    document = h.get("/openapi.json").json()
    assert document["openapi"].startswith("3.")
    assert "202" not in document["paths"]["/runs"]["post"]["responses"]
    assert document["paths"]["/runs"]["post"]["security"]
    error(h.post("/runs", {"actor": "admin", "grant": "forged"}), "REQUEST_INVALID", 422)
    error(h.post("/runs", {"note": "SECRET_NOTE" * 10000}), "REQUEST_TOO_LARGE", 413)
    error(
        h.request("POST", "/runs", content="x", headers={"Content-Type": "text/plain"}),
        "MEDIA_TYPE_UNSUPPORTED",
        415,
    )
    error(h.get("/unknown"), "OBJECT_NOT_FOUND", 404)
    error(h.request("DELETE", "/runs"), "METHOD_NOT_ALLOWED", 405)
    error(h.get("/runs", params={"case_code": "CASE:ANON", "limit": 101}), "REQUEST_INVALID", 422)
    error(
        h.get("/runs", params={"case_code": "CASE:ANON", "after": "invalid cursor"}),
        "REQUEST_INVALID",
        422,
    )
    error(h.get("/runs", params={"case_code": "CASE:ANON"}), "CAPABILITY_NOT_READY", 503)


@pytest.mark.parametrize(
    "exception,code,status",
    [
        (Fault("BUDGET_EXHAUSTED"), "BUDGET_EXHAUSTED", 429),
        (Fault("DATA_NOT_READY"), "DATA_NOT_READY", 409),
        (TimeoutError("secret DSN postgresql://password"), "OWNER_TIMEOUT", 504),
        (OSError("SECRET_NOTE key=private"), "OWNER_FAILURE", 503),
        (ValueError("arbitrary corrupt owner"), "OWNER_FAILURE", 503),
    ],
)
def test_owner_failure_http_redaction(exception, code, status):
    class BrokenRuntime:
        def get_case_bundle(self, code):
            raise exception

    h = HTTP(Dependencies(identities(cases=["CASE:ANON"]), runtime=BrokenRuntime()))
    r = h.get("/cases/CASE:ANON")
    error(r, code, status)
    assert not any(s in r.text for s in ("SECRET_NOTE", "postgresql", "password", "Traceback"))
