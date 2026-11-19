"""Deterministic OpenAPI, error matrix and anonymous examples; never opens a listener."""

import argparse
import json
from pathlib import Path

from apps.api.app import create_app
from apps.api.assembly import Dependencies
from apps.api.auth import LocalIdentities
from apps.api.errors import MATRIX
from apps.api.models import API_VERSION

ROOT = Path(__file__).resolve().parents[2]


def documents():
    app = create_app(Dependencies(LocalIdentities({})))
    request = dict(
        api_version=API_VERSION,
        idempotency_key="anonymous-create-1",
        case_code="CASE:ANON",
        run_id="RUN:ANON",
        request=dict(code="RESEARCH:ANON", version="1.0.0", checksum="sha256:" + "a" * 64),
        expected_version="sha256:" + "a" * 64,
        mode="create_only",
    )
    return {
        "docs/api/openapi.json": app.openapi(),
        "docs/api/error_matrix.json": {
            code: dict(http_status=status, retry=retry, message=message, automatic_retry=False)
            for code, (status, retry, message) in MATRIX.items()
        },
        "docs/api/examples.json": dict(
            synthetic=True,
            illustrative_not_authorization=True,
            create_run=request,
            created_receipt=dict(
                api_version=API_VERSION,
                trace_id="anonymous-trace",
                semantics="synchronous_result",
                contract_version="research-run/1.0.0",
                data=dict(
                    run_id="RUN:ANON",
                    case_code="CASE:ANON",
                    context_checksum="sha256:" + "b" * 64,
                    synthetic=True,
                    created=True,
                    creation_status="CREATED",
                    queued=False,
                    executed=False,
                    confirmed=True,
                    dispatch="not_installed",
                    task_id=None,
                ),
                live_domain_acceptance="live_domain_acceptance_not_enabled",
            ),
            commit_unknown=dict(
                api_version=API_VERSION,
                trace_id="anonymous-trace",
                error=dict(
                    error_code="COMMIT_UNKNOWN",
                    message=MATRIX["COMMIT_UNKNOWN"][2],
                    retry="reconcile_same_key",
                    automatic_retry=False,
                    commit_state="unconfirmed",
                ),
            ),
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for rel, value in documents().items():
        path = ROOT / rel
        expected = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected)
        elif not path.exists() or path.read_text() != expected:
            raise SystemExit("API schema snapshot mismatch: " + rel)
    print("12A OpenAPI/error matrix/anonymous examples verified")


if __name__ == "__main__":
    main()
