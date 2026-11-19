"""Fixed non-secret timing vocabulary; no remote text or request payload."""

import json
import math

PHASES = {"dispatch_boundary", "response_headers", "response_body", "worker_end"}


def read(raw):
    result = []
    for line in (raw or b"").splitlines():
        if not line.startswith(b"TIMING "):
            continue
        try:
            value = json.loads(line[7:])
            phase, seconds = value["phase"], value["seconds"]
            if (
                phase in PHASES
                and type(seconds) in {int, float}
                and math.isfinite(seconds)
                and seconds >= 0
            ):
                result.append(dict(phase=phase, seconds=seconds))
        except (ValueError, TypeError, KeyError):
            continue
    return result[:4]


def diagnostic(raw, *, timeout_origin=None):
    return dict(
        transport_timing=dict(
            schema_version="http-phase-timing/1.0.0",
            phases=read(raw),
            timeout_origin=timeout_origin,
            dispatch_is_not_server_receipt=True,
            before_headers_includes="connect_upload_queue_inference_not_separable",
        )
    )
