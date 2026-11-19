"""New FairVLMed input's original Runtime lease and scoped nested exclusion."""

import fcntl
import json
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace

from apps.segman_resource_lock import INPUT_VERSION as INPUT_VERSION
from apps.segman_resource_lock import (
    INPUT_VERSIONS,
    PROFILE,
    execution_compatibility,
    implementation,
)
from db.runtime.repository.case_pipeline import CasePipelineRepository
from packages.grounding.models import digest

ROOT = Path(__file__).resolve().parents[2]
_ACTIVE = ContextVar("generic_case_execution_lease", default=None)


def validate_profile(spec):
    selected = spec.get("schema_version") in INPUT_VERSIONS
    if selected:
        current = implementation(ROOT)
        if spec.get("execution_lock_profile") != PROFILE or (
            spec.get("execution_lock_implementation") != current
            and not execution_compatibility(ROOT, spec, current)
        ):
            raise ValueError("generic_execution_lock_configuration_changed")
    elif any(k in spec for k in ("execution_lock_profile", "execution_lock_implementation")):
        raise ValueError("generic_legacy_execution_lock_extension")
    return selected


def current(key=None):
    active = _ACTIVE.get()
    if active is not None:
        if key is not None and key != active.spec["key"]:
            raise ValueError("generic_execution_lock_scope")
        if json.loads((active.root / "input.json").read_bytes()) != active.spec:
            raise ValueError("generic_execution_lock_input_changed")
        validate_profile(active.spec)
        for guard in active.guards:
            guard()
    return active


def target():
    from apps.research_single_case import credential_boundary

    return credential_boundary().resolve_target("runtime")


def capability():
    """Zero-dispatch probe of the exact shared resource owner and host bridge."""
    from apps.segman_resource_lock import resource_lock

    repository = CasePipelineRepository(target())
    with repository.guarded_session("RESOURCE:SEGMAN_COMPONENT", "native-inference") as guard:
        guard()
        with resource_lock(ROOT) as local:
            guard()
            return dict(local, available=True, runtime_resource_lease=True)


@contextmanager
def run_lock(spec, directory):
    if not validate_profile(spec):
        with (directory / "operation.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        return
    if _ACTIVE.get() is not None:
        raise ValueError("generic_execution_lock_nested_run")
    repository = CasePipelineRepository(target())
    key = "generic-case-execution:" + digest(str(directory.resolve()))[7:]
    with repository.guarded_session("CASE:GENERIC_EXECUTION", key) as guard:
        active = SimpleNamespace(
            spec=spec, root=directory.resolve(), guards=(guard,), repository=repository
        )
        token = _ACTIVE.set(active)
        try:
            current(spec["key"])
            yield
        finally:
            _ACTIVE.reset(token)


@contextmanager
def scoped_lock(path, *, resource=None):
    active = current()
    if active is None:
        with path.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        return
    if not path.resolve().is_relative_to(active.root):
        raise ValueError("generic_execution_lock_path_scope")
    if resource is None:
        # Reentrant within the whole-case lease held by this execution context.
        yield
        return
    if resource != "segman-component":
        raise ValueError("generic_execution_resource_scope")
    with active.repository.guarded_session(
        "RESOURCE:SEGMAN_COMPONENT", "native-inference"
    ) as guard:
        nested = SimpleNamespace(**dict(vars(active), guards=active.guards + (guard,)))
        token = _ACTIVE.set(nested)
        try:
            current()
            yield
        finally:
            _ACTIVE.reset(token)


def wait_process(process, *, timeout, stop, extra_guard=None):
    """Poll the original case/resource lease; interrupted intent is never retried."""
    import subprocess
    import time

    if _ACTIVE.get() is None and extra_guard is None:
        return process.wait(timeout=timeout)
    deadline = time.monotonic() + timeout
    while True:
        try:
            current()
            if extra_guard is not None:
                extra_guard()
        except BaseException:
            # Cleanup only; preserve the actual exception, never call it a pass.
            stop(process)
            raise
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout)
        try:
            return process.wait(timeout=min(1, remaining))
        except subprocess.TimeoutExpired:
            if time.monotonic() >= deadline:
                raise
