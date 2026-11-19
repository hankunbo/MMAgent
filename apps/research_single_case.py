"""Unique K research CLI. Deployment is explicit; default invocation only reads readiness."""

import argparse
import importlib.util
import json
import signal
from pathlib import Path
from threading import Event

import psycopg

from packages.research_runtime.deployment import assemble, load_configuration


def credential_boundary():
    # Existing credential parser, fixed path; no request-provided resolver or DSN.
    path = Path("/opt/data/test/MMAgent/.local/07-p2-services-v2/deployment_common.py")
    spec = importlib.util.spec_from_file_location("research_deployment_boundary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/research/fairclip_single_case.json")
    parser.add_argument("--key")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--recover", action="store_true")
    args = parser.parse_args(argv)
    config = load_configuration(Path(args.config))
    if config["installed_bundle"] is None:
        result = dict(
            deployment_ready=False,
            case_preparation_pending=True,
            execution_authorized=False,
            reasons=["actual_DB_06_review_and_GPU_installation_pending"],
            public_entrypoint="python -m apps.research_single_case",
        )
        print(json.dumps(result))
        return 2
    deployment = assemble(config, credential_boundary())
    if not (args.execute or args.recover):
        print(
            json.dumps(
                dict(
                    deployment_ready=True, case_preparation_pending=True, execution_authorized=False
                )
            )
        )
        return 0
    if args.key is None:
        parser.error("--key is required for execution or recovery")
    cancellation = Event()
    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, lambda *_: cancellation.set())
    try:
        if args.recover:
            request = deployment.requests.for_key(args.key)
            result = deployment.service.run_registered(
                request, key=args.key, cancel=cancellation, recover_only=True
            )
        else:
            sample = deployment.prepare_sample()
            result = deployment.service.run(sample, key=args.key, cancel=cancellation)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return (
            3 if not result["terminal"] else 0 if result["technical_status"] == "SUCCEEDED" else 2
        )
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, psycopg.Error):
        # No credentials, hidden inputs or arbitrary external exception text cross the CLI.
        print(
            json.dumps(
                dict(
                    deployment_ready=False,
                    execution_authorized=False,
                    reason="research_deployment_or_run_rejected",
                )
            )
        )
        raise SystemExit(2) from None
