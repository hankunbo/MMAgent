"""Completed read-only failures may be gaps, never authorizations to replay them."""

LOCAL = {"schema", "refusal", "truncated", "unsupported_image", "no_result", "timeout"}


def reason(events):
    reserved = [e["request_id"] for e in events if e.get("phase") == "reserved"]
    completed = [e["record"] for e in events if e.get("phase") == "completed"]
    if (
        not reserved
        or len(set(reserved)) != len(reserved)
        or len(completed) != len(reserved)
        or {r["request_id"] for r in completed} != set(reserved)
        or any(r.get("role") != "vision" or r.get("error") not in LOCAL for r in completed)
    ):
        return None
    return completed[-1]["error"]
