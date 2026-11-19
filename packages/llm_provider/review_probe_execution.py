"""Append-only execution of public Review probes; this is not a patient runner."""

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from apps.vectorengine_single_case import write_once
from packages.evidence_details.review_batches import check_batch
from packages.grounding.models import digest

from .curl_client import verify_binary
from .network_route import route_opener
from .review_probe import DIRECTIONAL_VERSION, RUNGS, build, check
from .review_response import decode
from .transport import HTTPTransport


def status(output):
    output = Path(output)
    if (output / "summary.json").exists():
        return json.loads((output / "summary.json").read_bytes())
    manifest = json.loads((output / "plan.json").read_bytes())
    entries = []
    for entry in manifest["entries"]:
        folder = output / entry["name"]
        result = folder / "result.json"
        entries.append(
            json.loads(result.read_bytes())
            if result.exists()
            else dict(
                name=entry["name"],
                status="claimed_unconfirmed_no_replay"
                if (folder / "claim.json").exists()
                else "not_sent",
            )
        )
    return dict(
        status="incomplete" if (output / "execution.claim").exists() else "prepared",
        entries=entries,
        automatic_replay=False,
        clinical_qualification=False,
    )


def execute(output, load_secret, *, transport_factory=HTTPTransport):
    output = Path(output)
    if (output / "execution.claim").exists():
        raise ValueError("review_probe_already_claimed_use_status_no_replay")
    manifest = check(output)
    if (output / "cancel").exists():
        raise ValueError("review_probe_cancelled_before_credentials")
    provider_config = manifest.get("provider_binding", {}).get("path")
    directional_material = manifest["schema_version"] == DIRECTIONAL_VERSION
    config, *_ = build(
        manifest["model"],
        manifest["nonce"],
        RUNGS[0][0],
        0,
        provider_config=provider_config,
        directional_material=directional_material,
    )
    route_opener(config.network_route, config.transport.base_url)
    if getattr(config, "http_client", None) is not None:
        verify_binary(config.http_client)
    # Exclusive process-level claim prevents concurrent launches or blind crash recovery.
    with (output / "execution.claim").open("x") as handle:
        handle.write(digest(manifest) + "\n")
    started = time.monotonic()
    results, stopped = [], set()
    env_name = config.transport.api_key_env
    previous = os.environ.get(env_name)
    try:
        os.environ[env_name] = load_secret()
        for entry in manifest["entries"]:
            if entry["rung"] in stopped:
                continue
            if (output / "cancel").exists() or time.monotonic() - started > 900:
                break
            config, state, plan, body, counts = build(
                manifest["model"],
                manifest["nonce"],
                entry["rung"],
                entry["repetition"],
                provider_config=provider_config,
                directional_material=directional_material,
            )
            if (
                digest(config) != entry["checksums"]["config"]
                or digest(body) != entry["checksums"]["wire"]
            ):
                raise ValueError("review_probe_configuration_changed_before_dispatch")
            route_opener(config.network_route, config.transport.base_url)
            folder = output / entry["name"]
            write_once(
                folder / "claim.json",
                dict(at=datetime.now(UTC).isoformat(), wire_checksum=digest(body)),
            )
            transport = transport_factory(
                network_route=config.network_route, http_client=getattr(config, "http_client", None)
            )
            clock = time.monotonic()
            reply = transport.send(
                config.transport,
                body,
                timeout=100,
                effective_http_timeout=90,
                max_bytes=131072,
                timing_events=True,
            )
            contract = dict(passed=False)
            # Original parser/owner enforce exact IDs, missing semantics and citation membership.
            if reply.content and os.environ[env_name].encode() not in reply.content:
                with (folder / "response.txt").open("xb") as handle:
                    handle.write(reply.content)
                if not reply.error:
                    try:
                        candidate, _ = decode(reply.content, config)
                        check_batch(state, plan, plan.batches[0], candidate)
                        contract = dict(
                            passed=True,
                            items=len(candidate.items),
                            candidate_checksum=digest(candidate),
                        )
                    except (ValueError, TypeError, KeyError) as error:
                        contract = dict(passed=False, error_type=type(error).__name__)
            secret_echo = bool(reply.content and os.environ[env_name].encode() in reply.content)
            result = dict(
                name=entry["name"],
                rung=entry["rung"],
                repetition=entry["repetition"],
                at=datetime.now(UTC).isoformat(),
                seconds=time.monotonic() - clock,
                actual_requests=reply.api_requests,
                error="secret_response" if secret_echo else reply.error,
                contract=contract,
                finish_reason=reply.finish_reason,
                response_confirmed=reply.response_confirmed,
                http=transport.observation(reply),
                diagnostic=reply.diagnostic,
                usage=reply.usage.model_dump(mode="json"),
                inventory=counts,
            )
            write_once(folder / "result.json", result)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if reply.error or not contract["passed"]:
                stopped.add(entry["rung"])
    finally:
        if previous is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = previous
    confirmed = {r["name"] for r in results}
    qualified = [
        rung
        for rung, _, _ in RUNGS
        if sum(r["rung"] == rung and r["contract"]["passed"] for r in results) == 3
    ]
    summary = dict(
        schema_version="public-review-load-result/1.0.0",
        results=results,
        actual_requests=sum(r["actual_requests"] or 0 for r in results),
        request_count_uncertain=any(r["actual_requests"] is None for r in results),
        not_sent=[e["name"] for e in manifest["entries"] if e["name"] not in confirmed],
        qualified_rungs=qualified,
        three_rung_gate_passed=len(qualified) == 3,
        retries=0,
        clinical_qualification=False,
        patient_data=False,
        database_writes=0,
        goal_restarted=False,
        cancelled=(output / "cancel").exists(),
    )
    write_once(output / "summary.json", summary)
    return summary
