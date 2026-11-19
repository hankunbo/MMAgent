"""Immutable local data-analysis records, lock + write-ahead dispatch journal."""

import fcntl
import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

from packages.dataset_adapters.patient_storage import encoded
from packages.object_store import checksum
from packages.tool_execution.process import read_file


def read(path):
    return json.loads(read_file(Path(path).absolute(), 16777216))


def write_once(path, value):
    raw = encoded(value)
    write_bytes_once(path, raw)


def write_bytes_once(path, raw):
    path = Path(path)
    if path.exists():
        if read_file(path.absolute(), 16777216) != raw:
            raise ValueError("qualitative_immutable_record_conflict")
        return
    # A partial write must never look like an acknowledged request/result.
    tmp = path.with_name(".tmp-" + uuid.uuid4().hex)
    try:
        with tmp.open("xb") as f:
            os.chmod(tmp, 0o600)
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        tmp.unlink(missing_ok=True)


def seal(value):
    from packages.grounding.models import digest

    return dict(value, checksum=digest(value))


def unseal(value):
    from packages.grounding.models import digest

    body = {k: v for k, v in value.items() if k != "checksum"}
    if value.get("checksum") != digest(body):
        raise ValueError("qualitative_record_checksum")
    return body


@contextmanager
def locked(directory):
    directory = Path(directory).absolute()
    if any(p.is_symlink() for p in (directory, *directory.parents)):
        raise ValueError("qualitative_symlink_directory")
    # The project volume is NFS and has no flock service. Pin execution to this
    # boot + PID namespace, then use the local filesystem lock. Another host
    # cannot acquire execution by merely using the same shared output path.
    host = dict(
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        pid_namespace=os.readlink("/proc/self/ns/pid"),
    )
    write_once(directory / "execution-host.json", seal(host))
    lock_root = Path(tempfile.gettempdir()) / ("mmagent-qualitative-locks-" + str(os.getuid()))
    lock_root.mkdir(mode=0o700, exist_ok=True)
    if lock_root.is_symlink() or lock_root.stat().st_uid != os.getuid():
        raise ValueError("qualitative_local_lock_owner")
    lock_file = lock_root / checksum(str(directory).encode())[7:]
    descriptor = os.open(lock_file, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def file_bindings(paths, root):
    return {str(p.relative_to(root)): checksum(read_file(p.absolute(), 16777216)) for p in paths}
