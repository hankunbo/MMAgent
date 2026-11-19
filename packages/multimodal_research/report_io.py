"""Publish bounded report mirrors completely; persistent owners remain authoritative.

Only explicit owner-checked export recovery may bypass a proven truncated mirror,
and it writes one deterministic alternate, never replaces historical bytes.
"""

import json
import os
import tempfile
from pathlib import Path

from packages.grounding.models import canonical
from packages.object_store import checksum


def mirror_state(path, payload, *, kind):
    if not path.exists():
        return "absent"
    raw = path.read_bytes()
    if raw == payload:
        return "exact"
    if kind == "json":
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        else:
            if value == json.loads(payload):
                return "exact"
            raise ValueError("local_report_content_conflict")
    if len(raw) < len(payload) and payload.startswith(raw):
        return "partial"
    raise ValueError("local_report_content_conflict")


def publish(path, payload, *, kind, recover=None):
    path = Path(path)
    state = mirror_state(path, payload, kind=kind)
    if state == "partial":
        if recover is None:
            raise ValueError("local_report_partial_requires_recovery")
        recover()
        path = path.with_name(path.stem + ".recovered-" + checksum(payload)[7:23] + path.suffix)
        state = mirror_state(path, payload, kind=kind)
        if state == "partial":
            raise ValueError("local_report_recovery_mirror_incomplete")
    if state == "exact":
        return path
    fd, name = tempfile.mkstemp(prefix=".report-publish-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            if stream.write(payload) != len(payload):
                raise OSError("local_report_short_write")
            stream.flush()
            os.fsync(stream.fileno())
        # Link creates the complete final file atomically and never clobbers.
        try:
            os.link(temporary, path)
        except FileExistsError:
            if mirror_state(path, payload, kind=kind) != "exact":
                raise ValueError("local_report_publish_conflict") from None
    finally:
        temporary.unlink(missing_ok=True)
    return path


def publish_json(path, value, *, recover=None):
    return publish(path, canonical(value) + b"\n", kind="json", recover=recover)


def publish_html(path, page, *, recover=None):
    return publish(path, page.encode("utf-8"), kind="html", recover=recover)


def authorize_recovery(campaign, final):
    from .stages import local_current, review_for_export

    if not getattr(campaign, "local_export_recovery", False):
        raise ValueError("local_report_partial_requires_recovery")
    # A caller flag alone never substitutes for original owners or current use.
    local_current(campaign, None)
    if review_for_export(campaign) != final:
        raise ValueError("local_report_recovery_binding")


def review_mirror(campaign, final, detail):
    path = getattr(campaign, "review_report_path", None)
    if path is None:
        path = campaign.phase_path(None) / "review-report.json"
    payload = canonical(detail) + b"\n"
    state = mirror_state(path, payload, kind="json")
    if state != "partial":
        return None
    authorize_recovery(campaign, final)
    raw = path.read_bytes()
    if not payload.startswith(raw) or len(raw) >= len(payload):
        raise ValueError("local_report_recovery_binding")
    return dict(
        schema_version="local-review-mirror-recovery/1.0.0",
        run_id=final["run_id"],
        acceptance_key=final["acceptance_key"],
        mirror="review-report.json",
        original_checksum=checksum(raw),
        original_bytes=len(raw),
        owner_report_checksum=checksum(payload),
        reason="incomplete_exact_owner_prefix",
        original_preserved=True,
        added_requests=0,
    )
