"""Fixed local research processes. Path checks are admission, not confinement."""

import ctypes
import fcntl
import hashlib
import json
import os
import re
import selectors
import signal
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from packages.grounding.models import digest

from .models import ExecutionError
from .process import identity

ROOT = Path("/opt/data/test")
# One project-wide reservation across all attempts, policies and GPU UUIDs.
LOCK_ROOT = ROOT / "MMAgent/.local/trusted-local-gpu-v1"


def path_checked(value, *, project=True):
    path = Path(value)
    if not path.is_absolute() or str(path) != str(value) or path.resolve() != path:
        raise ExecutionError("trusted_path_rejected")
    if project and (path == ROOT or not path.is_relative_to(ROOT)):
        raise ExecutionError("trusted_path_outside_project_root")
    return path


def verify_file(item, *, deadline, limit, project=True):
    """Streaming, bounded digest comparison to installed review, never a new approval."""
    path = path_checked(item.path, project=project)
    if item.byte_size > limit:
        raise ExecutionError("trusted_resource_budget")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        leaf = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
    finally:
        os.close(descriptor)
    with os.fdopen(leaf, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != item.byte_size:
            raise ExecutionError("trusted_resource_size_drift")
        hasher, size = hashlib.sha256(), 0
        while block := stream.read(1048576):
            size += len(block)
            if size > limit or time.monotonic() > deadline:
                raise ExecutionError("trusted_resource_budget")
            hasher.update(block)
        after = os.fstat(stream.fileno())
    if (
        size != item.byte_size
        or "sha256:" + hasher.hexdigest() != item.checksum
        or (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    ):
        raise ExecutionError("trusted_resource_digest_drift")


def journal(path, document):
    with path.open("x") as stream:
        json.dump(document, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(path.parent)


def sync_directory(path):
    fd = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def private_directory(path):
    path_checked(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.stat()
    if info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise ExecutionError("trusted_directory_not_private")


class NvidiaObservation:
    """Read-only driver observations. UUID visibility is not a security boundary."""

    def read(self, policy):
        verify_file(
            policy.nvidia_smi,
            deadline=time.monotonic() + 5,
            limit=67108864,
            project=False,
        )

        def query(fields):
            try:
                result = subprocess.run(
                    [policy.nvidia_smi.path, fields, "--format=csv,noheader,nounits"],
                    shell=False,
                    env={},
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                raise ExecutionError("gpu_observation_unavailable") from None
            if result.returncode or len(result.stdout) > 65536:
                raise ExecutionError("gpu_observation_unavailable")
            return result.stdout.decode("ascii").splitlines()

        devices = {}
        for row in query("--query-gpu=uuid,memory.total,memory.used"):
            uuid, total, used = [v.strip() for v in row.split(",")]
            devices[uuid] = (int(total), int(used))
        processes = []
        for row in query("--query-compute-apps=gpu_uuid,pid"):
            uuid, pid = [v.strip() for v in row.split(",")]
            processes.append((uuid, int(pid)))
        return devices, processes


class TrustedLocalProcessRunner:
    def __init__(self, workspace, policy):
        self.workspace = path_checked(workspace)
        private_directory(self.workspace)
        self.policy = policy
        self.observation = NvidiaObservation()
        self._leased = False

    def check_capabilities(self):
        if sys.platform != "linux":
            raise ExecutionError("trusted_linux_required")
        if self.policy.cpu_count > len(os.sched_getaffinity(0)):
            raise ExecutionError("cpu_affinity_capacity_unavailable")
        return self.policy.capabilities.model_dump(mode="json")

    @contextmanager
    def slot(self):
        private_directory(LOCK_ROOT)
        fd = os.open(LOCK_ROOT / "global.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "a+b") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ExecutionError("concurrency_limit") from None
            try:
                if (LOCK_ROOT / "active.json").exists() or (
                    LOCK_ROOT / "quarantined.json"
                ).exists():
                    raise ExecutionError("trusted_resource_quarantined_no_replay")
                self._leased = True
                yield
            finally:
                self._leased = False
                fcntl.flock(stream, fcntl.LOCK_UN)

    def review_for(self, step):
        review = next(
            (r for r in self.policy.reviews if r.implementation_ref == step.implementation_ref),
            None,
        )
        if review is None or (
            review.manifest_checksum != step.deployment.manifest.checksum
            or review.allowlist_checksum != step.deployment.allowlist.checksum
        ):
            raise ExecutionError("trusted_review_binding_missing")
        return review

    def verify_review(self, review):
        deadline = time.monotonic() + self.policy.closure_timeout_seconds
        verify_file(review.review_record, deadline=deadline, limit=1048576)
        roots = [path_checked(r) for r in review.inventory_roots]
        environment = path_checked(review.environment_root)
        if environment not in roots or not Path(review.executable.path).is_relative_to(environment):
            raise ExecutionError("trusted_environment_binding_invalid")
        expected = {f.path for f in review.files}
        actual = set()
        for root in roots:
            if not root.is_dir():
                raise ExecutionError("trusted_inventory_missing")
            for parent, directories, filenames in os.walk(root, followlinks=False):
                if time.monotonic() > deadline:
                    raise ExecutionError("trusted_resource_budget")
                for name in directories + filenames:
                    p = Path(parent) / name
                    path_checked(p)
                for name in filenames:
                    actual.add(str(Path(parent) / name))
                    if len(actual) > 16384:
                        raise ExecutionError("trusted_inventory_limit")
        if actual != expected:
            raise ExecutionError("trusted_inventory_drift")
        for item in review.files:
            verify_file(item, deadline=deadline, limit=self.policy.max_resource_bytes)

    def gpu_ready(self):
        devices, processes = self.observation.read(self.policy)
        selected = self.policy.selected_gpu
        if not set(self.policy.gpu_pool) <= devices.keys():
            raise ExecutionError("gpu_uuid_unavailable")
        total, used = devices[selected]
        if self.policy.version == "trusted-local-policy/1.0.0":
            if any(uuid == selected for uuid, _ in processes):
                raise ExecutionError("gpu_busy_no_external_process_changes")
            if total - used < self.policy.vram_budget_mib:
                raise ExecutionError("gpu_budget_unavailable")
        # 1.1 retains the project-wide slot and real device observation, but does
        # not reserve the whole card or reject based on estimated free VRAM.
        # Allocation failures remain real process failures, never medical negatives.
        return dict(uuid=selected, total_mib=total, used_mib=used, compute_processes=processes)

    def prepare(self, attempt, step, config):
        if not self._leased:
            raise ExecutionError("trusted_project_lease_required")
        if not re.fullmatch(r"[0-9a-f]{32}", attempt) or not re.fullmatch(
            r"step_[0-9a-f]{64}", step.step_id
        ):
            raise ExecutionError("trusted_attempt_identity_invalid")
        review = self.review_for(step)
        self.verify_review(review)
        base = self.workspace / attempt / step.step_id
        base.mkdir(parents=True, mode=0o700)
        root = base / "root"
        root.mkdir(mode=0o700)
        for name in ("in", "out", "tmp", "cache", "home"):
            (root / name).mkdir(mode=0o700)
        journal(base / "trusted-policy.json", self.policy.model_dump(mode="json"))
        journal(base / "review.json", review.model_dump(mode="json"))
        return base, root, {f.path: f.checksum for f in review.files}

    def run(self, base, root, argv, config, timeout, cancel, on_start):
        if not self._leased:
            raise ExecutionError("trusted_project_lease_required")
        # Prepared paths cannot be used to launch a different command through this port.
        from .trusted_models import TrustedToolReview

        review = TrustedToolReview.model_validate_json((base / "review.json").read_text())
        if review not in self.policy.reviews or argv[:2] != [
            review.executable.path,
            review.entrypoint.path,
        ]:
            raise ExecutionError("trusted_launch_binding_invalid")
        self.verify_review(review)
        gpu = self.gpu_ready()
        journal(base / "gpu-before.json", gpu)
        environment = {
            "HOME": str(root / "home"),
            "TMPDIR": str(root / "tmp"),
            "TMP": str(root / "tmp"),
            "TEMP": str(root / "tmp"),
            "XDG_CACHE_HOME": str(root / "cache"),
            "TORCH_HOME": str(root / "cache"),
            "HF_HOME": str(root / "cache"),
            "CUDA_CACHE_PATH": str(root / "cache"),
            "MPLCONFIGDIR": str(root / "cache"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": self.policy.selected_gpu,
            "OMP_NUM_THREADS": str(config.cpu_count),
            "OPENBLAS_NUM_THREADS": str(config.cpu_count),
            "LC_ALL": "C.UTF-8",
        }
        spec = dict(
            argv=argv,
            cwd=str(root),
            environment=environment,
            cpus=sorted(os.sched_getaffinity(0))[: config.cpu_count],
            file_bytes=config.max_output_bytes,
        )
        journal(base / "launch.json", spec)
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0):
            raise ExecutionError("subreaper_unavailable")
        journal(LOCK_ROOT / "active.json", dict(base=str(base), policy=digest(self.policy)))
        gate_read, gate_write = os.pipe()
        proc, proc_identity = None, None
        selector = selectors.DefaultSelector()
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        error, started = None, time.monotonic()
        try:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    str(Path(__file__).with_name("_trusted_child.py")),
                    str(base / "launch.json"),
                    str(gate_read),
                ],
                cwd=base,
                env={},
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                close_fds=True,
                pass_fds=(gate_read,),
                start_new_session=True,
            )
            os.close(gate_read)
            gate_read = None
            proc_identity = identity(proc.pid)
            on_start(proc_identity)
            os.write(gate_write, b"1")
            os.close(gate_write)
            gate_write = None
            for name in buffers:
                stream = getattr(proc, name)
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, name)
            while proc.poll() is None or selector.get_map():
                if cancel.is_set():
                    error = "cancelled"
                    break
                if time.monotonic() - started > min(timeout, self.policy.timeout_seconds):
                    error = "timeout"
                    break
                for key, _ in selector.select(0.02):
                    chunk = os.read(key.fd, 8192)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    remaining = self.policy.log_bytes - len(buffers[key.data])
                    buffers[key.data].extend(chunk[:remaining])
                    if len(chunk) > remaining:
                        error = "log_size_limit"
                if error:
                    break
                if proc.poll() is not None:
                    # Drain the bounded pipes after the leader exits. Descendants
                    # holding them open stay subject to the same wall deadline.
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            if error is None and proc.poll() not in (None, 0):
                error = "process_nonzero"
        finally:
            for gate in (gate_read, gate_write):
                if gate is not None:
                    os.close(gate)
            clean = self._cleanup(proc, proc_identity)
            selector.close()
            if proc is not None:
                proc.stdout.close()
                proc.stderr.close()
            for name, buf in buffers.items():
                with (base / (name + ".log")).open("xb") as stream:
                    stream.write(buf)
            journal(
                base / "trusted-cleanup.json",
                dict(
                    group_cleanup_observed=clean,
                    identity=proc_identity,
                    exit_code=proc.returncode if proc is not None else None,
                    escaped_descendants_not_excluded=True,
                ),
            )
            if clean:
                (LOCK_ROOT / "active.json").unlink()
                sync_directory(LOCK_ROOT)
            else:
                journal(
                    LOCK_ROOT / "quarantined.json", dict(base=str(base), reason="cleanup_unknown")
                )
                raise ExecutionError("process_cleanup_unconfirmed")
        return dict(
            exit_code=proc.returncode,
            error=error,
            stdout=bytes(buffers["stdout"]),
            stderr=bytes(buffers["stderr"]),
        )

    def _cleanup(self, proc, observed):
        if proc is None or observed is None:
            return False
        try:
            if proc.poll() is None and identity(proc.pid) != observed:
                return False
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=self.policy.cleanup_seconds)
            deadline = time.monotonic() + self.policy.cleanup_seconds
            while time.monotonic() < deadline:
                try:
                    while os.waitpid(-proc.pid, os.WNOHANG)[0]:
                        pass
                except ChildProcessError:
                    pass
                try:
                    os.killpg(proc.pid, 0)
                except ProcessLookupError:
                    # Driver query also fails closed if cleanup cannot be observed.
                    _, jobs = self.observation.read(self.policy)
                    if self.policy.version == "trusted-local-policy/1.1.0":
                        # The owned process group is gone. Unrelated compute jobs
                        # do not make our cleanup uncertain on a shared device.
                        # This is not proof against escaped descendants or PID
                        # namespace differences (see capabilities.cleanup).
                        return not any(
                            uuid == self.policy.selected_gpu and pid == observed["pid"]
                            for uuid, pid in jobs
                        )
                    return not any(uuid == self.policy.selected_gpu for uuid, _ in jobs)
                time.sleep(0.02)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return False
        return False
