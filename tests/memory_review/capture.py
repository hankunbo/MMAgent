"""Direct subprocess exit/log capture. Never calls historical collectors."""

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def capture(output, name, command):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    log = output / (name + ".log")
    if log.exists():
        raise SystemExit("refusing to overwrite evidence")
    start, stamp = time.monotonic(), datetime.now(UTC).isoformat()
    sessions = {}
    with log.open("wb") as stream:
        process = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            while process.poll() is None:
                for marker in Path(".local").glob("pytest-session-*/owner.json"):
                    try:
                        owner = json.loads(marker.read_text())
                        if owner.get("pid") == process.pid:
                            sessions[str(marker.parent.resolve())] = owner
                    except (OSError, ValueError):
                        pass
                time.sleep(0.2)
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGINT)
            process.wait()
        direct_exit = process.wait()
    text = log.read_text(errors="replace")
    result = dict(
        command=command,
        direct_exit=direct_exit,
        started_at=stamp,
        ended_at=datetime.now(UTC).isoformat(),
        elapsed_seconds=time.monotonic() - start,
        log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
        summaries=[
            line for line in text.splitlines() if re.search(r"=+ .* (passed|failed|error)", line)
        ],
        passed=re.findall(r"^(.+::\S+) PASSED", text, re.M),
        failed=re.findall(r"^(.+::\S+) (?:FAILED|ERROR)", text, re.M),
    )
    paths = set(
        re.findall(
            r"/opt/data/test/MMAgent/\.local/(?:pytest-session|db-test-pytest)-[A-Za-z0-9_-]+", text
        )
    )
    paths.update(sessions)
    result["owned_sessions"] = sessions
    result["owned_paths"] = {p: dict(absent=not Path(p).exists()) for p in sorted(paths)}
    remaining = []
    for commandline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            raw = commandline.read_bytes()
            if b"postgres" in raw and any(p.encode() in raw for p in paths):
                remaining.append(int(commandline.parent.name))
        except OSError:
            pass
    result["owned_pg_processes_remaining"] = remaining
    result["cleanup_verified_in_log"] = (
        "cleanup=verified" in text or "cleanup verified" in text or "cleanup: verified" in text
    )
    (output / (name + ".exit")).write_text(str(direct_exit) + "\n")
    (output / (name + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return direct_exit


if __name__ == "__main__":
    raise SystemExit(capture(sys.argv[1], sys.argv[2], sys.argv[3:]))
