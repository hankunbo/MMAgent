"""A closed Linux deployment, not a general sandbox for downloaded code."""

import ctypes
import fcntl
import json
import os
import platform
import selectors
import signal
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from packages.object_store import checksum

from .models import ExecutionError


def read_file(path, limit):
    path = Path(path)
    if not path.is_absolute() or any(p in ("..", ".") for p in path.parts):
        raise ExecutionError("path_rejected")
    # Reject symlinks in every component, including the parent of a leaf.
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
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ExecutionError("file_type_or_size_rejected")
        data = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if len(data) > limit or (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ExecutionError("file_changed_during_read")
    return data


def write_copy(root, target, data, *, executable=False):
    target = Path(target)
    if not target.is_absolute() or ".." in target.parts or str(target).startswith("/out/"):
        raise ExecutionError("deployment_path_rejected")
    path = root / str(target).lstrip("/")
    # The host attempt stays private. Only NEW directories inside its chroot
    # must be traversable by the dropped uid, even under the service's umask 077.
    parent = root
    for component in path.parent.relative_to(root).parts:
        parent = parent / component
        try:
            parent.mkdir(mode=0o755)
        except FileExistsError:
            if not stat.S_ISDIR(parent.lstat().st_mode):
                raise ExecutionError("deployment_parent_rejected") from None
        else:
            parent.chmod(0o755)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o555 if executable else 0o444)
    return path


def identity(pid):
    raw = Path(f"/proc/{pid}/stat").read_text()
    return dict(
        pid=pid,
        start_ticks=raw[raw.rfind(")") + 2 :].split()[19],
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    )


class LinuxProcessRunner:
    def __init__(self, workspace, policy):
        requested = Path(workspace).absolute()
        if requested.resolve() != requested:
            raise ExecutionError("workspace_symlink_or_traversal_rejected")
        self.workspace = requested
        self.workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
        info = self.workspace.stat()
        if info.st_uid != os.geteuid() or info.st_mode & 0o022:
            raise ExecutionError("workspace_must_be_service_owned_and_private")
        self.policy = policy

    def check_capabilities(self):
        if platform.machine() != "x86_64" or sys.platform != "linux" or os.geteuid() != 0:
            raise ExecutionError("isolation_requires_linux_x86_64_chroot_setuid_seccomp")
        # Availability is separate from the real child confinement tests.
        required = (1 << 18) | (1 << 7) | (1 << 6)
        caps = int(
            next(
                x.split()[1]
                for x in Path("/proc/self/status").read_text().splitlines()
                if x.startswith("CapEff:")
            ),
            16,
        )
        if caps & required != required:
            raise ExecutionError("chroot_or_privilege_drop_capability_missing")
        return dict(
            chroot_privilege=True,
            seccomp="tested_at_each_launch_fail_closed",
            gpu="unavailable",
            aggregate_tree_memory="unavailable",
            mount_namespace="not_used",
            network="seccomp_socket_syscalls_denied",
        )

    @contextmanager
    def slot(self):
        descriptor = os.open(
            self.workspace / "execution.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
        )
        with os.fdopen(descriptor, "a+b") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ExecutionError("concurrency_limit") from None
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def prepare(self, attempt, step, config):
        self.check_capabilities()
        if step.deployment.profile.runtime.gpu != "not_required":
            raise ExecutionError("gpu_isolation_unavailable")
        if set(config.environment_variables) != set(self.policy.environment):
            raise ExecutionError("environment_binding_mismatch")
        # Even a mistaken deployment owner cannot forward authentication or Python
        # loader controls. Empty is the standard fixture profile.
        if any(k not in {"LANG", "LC_ALL", "OMP_NUM_THREADS"} for k in self.policy.environment):
            raise ExecutionError("environment_variable_forbidden")
        if config.cpu_count > len(os.sched_getaffinity(0)):
            raise ExecutionError("cpu_affinity_capacity_unavailable")
        base = self.workspace / attempt / step.step_id
        base.mkdir(parents=True, mode=0o700)
        root = base / "root"
        root.mkdir(mode=0o755)
        root.chmod(0o755)
        (root / "out").mkdir(mode=0o777)
        (root / "out").chmod(0o777)
        copied = {}
        for item in self.policy.runtime_files:
            data = read_file(item.source, 67108864)
            if checksum(data) != item.checksum:
                raise ExecutionError("runtime_closure_changed")
            write_copy(root, item.target, data, executable=item.executable)
            copied[item.target] = item.checksum
        return base, root, copied

    def run(self, base, root, argv, config, timeout, cancel, on_start):
        gate_read, gate_write = os.pipe()
        spec = dict(
            gate_fd=gate_read,
            root=str(root),
            argv=argv,
            environment=dict(self.policy.environment),
            uid=self.policy.sandbox_uid,
            cpus=sorted(os.sched_getaffinity(0))[: config.cpu_count],
            memory_bytes=config.memory_bytes,
            cpu_seconds=self.policy.cpu_seconds,
            output_bytes=config.max_output_bytes,
            process_limit=self.policy.process_limit,
            supervisor_pid=os.getpid(),
        )
        path = base / "launch.json"
        try:
            path.write_text(json.dumps(spec))
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.prctl(36, 1, 0, 0, 0):
                raise ExecutionError("subreaper_unavailable")
        except BaseException:
            os.close(gate_read)
            os.close(gate_write)
            raise
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    str(Path(__file__).with_name("_sandbox_child.py")),
                    str(path),
                ],
                shell=False,
                cwd=base,
                env={},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                close_fds=True,
                pass_fds=(gate_read,),
                start_new_session=True,
            )
        except BaseException:
            os.close(gate_read)
            os.close(gate_write)
            raise
        os.close(gate_read)
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        selector = selectors.DefaultSelector()
        error, started = None, time.monotonic()
        try:
            on_start(identity(process.pid))
            try:
                os.write(gate_write, b"1")
            except BrokenPipeError:
                pass  # setup failed before gate release; collect the real child exit
            os.close(gate_write)
            gate_write = None
            started = time.monotonic()
            for name in buffers:
                stream = getattr(process, name)
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, name)
            while process.poll() is None or selector.get_map():
                if cancel.is_set():
                    error = "cancelled"
                    break
                if time.monotonic() - started > timeout:
                    error = "timeout"
                    break
                for key, _ in selector.select(0.02):
                    chunk = os.read(key.fd, 8192)
                    if not chunk:
                        selector.unregister(key.fileobj)
                    else:
                        buf = buffers[key.data]
                        remaining = self.policy.log_bytes - len(buf)
                        buf.extend(chunk[:remaining])
                        if len(chunk) > remaining:
                            error = "log_size_limit"
                if error:
                    break
                if process.poll() is not None:
                    # Descendants may hold pipes or keep writing after parent exit.
                    self._kill_group(process.pid)
            if error is None and process.wait() != 0:
                error = (
                    "isolation_setup_failed"
                    if process.returncode == 125
                    else "resource_limit"
                    if process.returncode in (-24, -25)
                    else "process_nonzero"
                )
        finally:
            if gate_write is not None:
                os.close(gate_write)
            self._kill_group(process.pid)
            process.wait()
            # Reap only children in this owned process group, never PG or other jobs.
            deadline = time.monotonic() + 5
            while True:
                try:
                    child, _ = os.waitpid(-process.pid, os.WNOHANG)
                except ChildProcessError:
                    break
                if child == 0:
                    if time.monotonic() >= deadline:
                        raise ExecutionError("process_cleanup_unconfirmed")
                    time.sleep(0.01)
            selector.close()
            process.stdout.close()
            process.stderr.close()
            # Bounded bytes back the existing StepResult checksums. Keep these in
            # this backend's attempt, as the trusted GPU backend already does.
            for name, buffer in buffers.items():
                with (base / (name + ".log")).open("xb") as stream:
                    stream.write(buffer)
                    stream.flush()
                    os.fsync(stream.fileno())
        return dict(
            exit_code=process.returncode,
            error=error,
            elapsed=time.monotonic() - started,
            stdout=bytes(buffers["stdout"]),
            stderr=bytes(buffers["stderr"]),
        )

    @staticmethod
    def _kill_group(pid):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
