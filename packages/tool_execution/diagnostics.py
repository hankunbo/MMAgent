"""Bounded, checksum-bound local diagnostics; never an execution authorization.

Read the existing runner log, not caller text. Public output contains only
allowlisted fault categories, never the traceback, environment or credentials.
"""

import hashlib
import re
from pathlib import Path


def explain_local_failure(workspace, *, attempt_id, step):
    if not re.fullmatch(r"[a-f0-9]{32}", attempt_id) or not re.fullmatch(
        r"step_[a-f0-9]{64}", step["step_id"]
    ):
        raise ValueError("diagnostic_identity")
    history = step["status_history"]
    states = [v if isinstance(v, str) else v["status"] for v in history]
    result = dict(
        schema_version="tool-failure-diagnostic/1.0.0",
        attempt_id=attempt_id,
        step_id=step["step_id"],
        status=step["result"]["status"],
        process_started="RUNNING" in states,
        exit_code=step.get("exit_code"),
        error_code=step.get("error_code"),
        elapsed_seconds=step.get("elapsed_seconds"),
        log_verified=False,
        category="execution_failure",
        stage="execution",
        retry_authorized=False,
    )
    if "RUNNING" not in states:
        result.update(stage="prelaunch", category="not_started")
    elif result["status"] == "SUCCEEDED":
        result.update(stage="completed", category="none")
    path = Path(workspace) / attempt_id / step["step_id"] / "stderr.log"
    if not path.exists():
        return result
    if any(p.is_symlink() for p in (path, *path.parents)) or path.stat().st_size > 1048576:
        raise ValueError("diagnostic_log_bounds")
    raw = path.read_bytes()
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    if digest != step.get("stderr_checksum"):
        raise ValueError("diagnostic_log_checksum")
    result.update(log_verified=True, log_checksum=digest, log_bytes=len(raw))
    if b"AttributeError: module 'platform' has no attribute 'system'" in raw:
        result.update(category="python_stdlib_shadowing", stage="dependency_import")
    elif b"ModuleNotFoundError:" in raw or b"ImportError:" in raw:
        result.update(category="dependency_import_error", stage="dependency_import")
    elif b"CUDA out of memory" in raw:
        result.update(category="gpu_memory_exhausted", stage="inference")
    return result
