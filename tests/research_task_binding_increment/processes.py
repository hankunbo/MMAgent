"""Exact owned-process observations; exited zombies are not running servers."""

import os
from pathlib import Path


def process_state(pid):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return None
    # comm may contain spaces/parentheses. The final ')' terminates comm.
    end = raw.rindex(")")
    fields = raw[end + 2 :].split()
    if len(fields) < 20 or fields[0] not in set("RSDZTtXxKWPI"):
        raise ValueError("owned_process_state_unknown")
    return dict(
        pid=pid,
        name=raw[raw.index("(") + 1 : end],
        state=fields[0],
        ppid=int(fields[1]),
        start_ticks=fields[19],
    )


def classify(original, current):
    if current is None:
        return "absent"
    if current["pid"] != original["pid"] or current["start_ticks"] != original["start_ticks"]:
        return "pid_reused"
    if current["state"] in {"Z", "X", "x"}:
        return "terminated"
    return "running"


def observe_owned(original, *, reap=False):
    current = process_state(original["pid"])
    result = dict(
        original=original, current=current, disposition=classify(original, current), reaped=False
    )
    # Reap only this recorded exact child after it is already a zombie. No signals,
    # broad waitpid(-1), process scans, or authority to stop/delete other instances.
    if (
        reap
        and result["disposition"] == "terminated"
        and current["state"] == "Z"
        and current["ppid"] == os.getpid()
    ):
        try:
            pid, _ = os.waitpid(original["pid"], os.WNOHANG)
        except ChildProcessError:
            pid = 0
        result["reaped"] = pid == original["pid"]
        result["after_reap"] = process_state(original["pid"])
    return result
