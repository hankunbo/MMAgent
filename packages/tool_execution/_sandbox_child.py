"""Trusted Linux bootstrap. Runs in a fresh process, never preexec_fn or a runner DB client."""

import ctypes
import errno
import json
import os
import resource
import signal
import sys


def restrict_syscalls():
    class Filter(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_ushort),
            ("jt", ctypes.c_ubyte),
            ("jf", ctypes.c_ubyte),
            ("k", ctypes.c_uint),
        ]

    class Program(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(Filter))]

    # x86_64 only; kill on a foreign ABI, including x32. No networking, namespace,
    # process group escape, tracing or signalling host processes.
    instructions = [
        (0x20, 0, 0, 4),
        (0x15, 1, 0, 0xC000003E),
        (0x06, 0, 0, 0x80000000),
        (0x20, 0, 0, 0),
        (0x35, 0, 1, 0x40000000),
        (0x06, 0, 0, 0x80000000),
    ]
    denied = (
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        62,
        101,
        109,
        112,
        155,
        161,
        165,
        166,
        169,
        175,
        176,
        200,
        234,
        246,
        248,
        249,
        250,
        272,
        288,
        298,
        299,
        307,
        308,
        310,
        311,
        313,
        321,
        322,
        323,
        425,
        426,
        427,
        434,
        435,
        438,
    )
    for number in denied:
        instructions.extend([(0x15, 0, 1, number), (0x06, 0, 0, 0x50000 | errno.EPERM)])
    instructions.append((0x06, 0, 0, 0x7FFF0000))
    array = (Filter * len(instructions))(*(Filter(*i) for i in instructions))
    program = Program(len(instructions), array)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) or libc.prctl(22, 2, ctypes.byref(program), 0, 0):
        raise OSError(ctypes.get_errno(), "seccomp_unavailable")


def main():
    with open(sys.argv[1]) as stream:
        spec = json.load(stream)
    libc = ctypes.CDLL(None, use_errno=True)
    # Kernel kills this leader if supervisor disappears. Parent still marks the
    # attempt uncertain; this is not proof all descendants stopped after a crash.
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0):
        raise OSError("parent_death_signal_unavailable")
    if os.getppid() != spec["supervisor_pid"]:
        raise OSError("supervisor_lost")
    os.sched_setaffinity(0, spec["cpus"])
    for limit, value in (
        (resource.RLIMIT_AS, spec["memory_bytes"]),
        (resource.RLIMIT_CPU, spec["cpu_seconds"]),
        (resource.RLIMIT_FSIZE, spec["output_bytes"]),
        (resource.RLIMIT_NPROC, spec["process_limit"]),
        (resource.RLIMIT_NOFILE, 64),
        (resource.RLIMIT_CORE, 0),
    ):
        resource.setrlimit(limit, (value, value + 1 if limit == resource.RLIMIT_CPU else value))
    os.chroot(spec["root"])
    os.chdir("/out")
    os.setgroups([])
    os.setgid(spec["uid"])
    os.setuid(spec["uid"])
    # Changing credentials clears PDEATHSIG; reinstall it and close the race.
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) or os.getppid() != spec["supervisor_pid"]:
        raise OSError("supervisor_lost_after_privilege_drop")
    os.umask(0o077)
    restrict_syscalls()
    if os.read(spec["gate_fd"], 1) != b"1":
        raise OSError("execution_registration_missing")
    os.close(spec["gate_fd"])
    os.execve(spec["argv"][0], spec["argv"], spec["environment"])


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        os.write(2, b"sandbox_setup_failed\n")
        os._exit(125)
