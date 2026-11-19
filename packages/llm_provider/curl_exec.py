"""Internal fixed curl exec with Linux parent-death protection; no shell or key arguments."""

import ctypes
import fcntl
import json
import math
import os
import signal
import sys

# Linux UAPI: some Python builds omit these constants despite host kernel/libc support.
ADD_SEALS, GET_SEALS, SEAL_ALL = 1033, 1034, 0x000F


def sealed_fd(name, raw):
    if sys.platform != "linux":
        raise ValueError("curl_memfd_requires_linux")
    libc = ctypes.CDLL(None, use_errno=True)
    create = getattr(libc, "memfd_create", None)
    if create is None:
        raise ValueError("curl_memfd_unavailable")
    create.argtypes, create.restype = [ctypes.c_char_p, ctypes.c_uint], ctypes.c_int
    fd = create(name.encode(), 0x0001 | 0x0002)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "curl_memfd_failed")
    try:
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.lseek(fd, 0, os.SEEK_SET)
        fcntl.fcntl(fd, ADD_SEALS, SEAL_ALL)
        return fd
    except BaseException:
        os.close(fd)
        raise


def install_input():
    """Avoid curl config-line limits: sealed anonymous body FD, secret config on stdin only."""
    raw = sys.stdin.buffer.read(16_777_217)
    if len(raw) > 16_777_216:
        raise ValueError("curl_stdin_bound")
    body_fd, config_fd = None, None
    try:
        lines = []
        for line in raw.decode().splitlines():
            name, separator, encoded = line.partition(" = ")
            if not separator or name not in {
                "url",
                "proxy",
                "request",
                "user-agent",
                "header",
                "data-binary",
            }:
                raise ValueError("curl_stdin_format")
            value = json.loads(encoded)
            if not isinstance(value, str):
                raise ValueError("curl_stdin_value")
            if name == "data-binary":
                if body_fd is not None:
                    raise ValueError("curl_body_duplicate")
                body_fd = sealed_fd("mmagent-curl-body", value.encode())
                value = "@/proc/self/fd/" + str(body_fd)
            lines.append(name + " = " + json.dumps(value, ensure_ascii=False) + "\n")
        config = "".join(lines).encode()
        if len(config) > 65536:
            raise ValueError("curl_config_bound")
        config_fd = sealed_fd("mmagent-curl-stdin", config)
        os.dup2(config_fd, 0)
        if body_fd is not None:
            os.set_inheritable(body_fd, True)
        return body_fd
    except BaseException:
        if body_fd is not None:
            os.close(body_fd)
        raise
    finally:
        if config_fd is not None:
            os.close(config_fd)


def parent_death_kill(parent):
    """Arm before reading credentials; close the parent-exit-before-prctl race."""
    if parent <= 1 or sys.platform != "linux":
        raise ValueError("curl_parent_binding")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent:
        raise ValueError("curl_parent_binding")


def main():
    from .curl_client import BINARY, WRITE_OUT, CurlClient, verify_binary

    parent, sha, timeout, maximum = sys.argv[1:]
    parent, timeout, maximum = int(parent), float(timeout), int(maximum)
    if not math.isfinite(timeout) or not 0 < timeout <= 600 or not 1 <= maximum <= 2_000_000:
        return 125
    verify_binary(CurlClient(binary_checksum=sha))
    parent_death_kill(parent)
    body_fd = install_input()
    try:
        os.execv(
            str(BINARY),
            [
                str(BINARY),
                "--disable",
                "--config",
                "-",
                "--http1.1",
                "--silent",
                "--proto",
                "=https",
                "--max-redirs",
                "0",
                "--retry",
                "0",
                "--connect-timeout",
                str(min(20.0, timeout)),
                "--max-time",
                str(timeout),
                "--max-filesize",
                str(maximum),
                "--include",
                "--suppress-connect-headers",
                "--write-out",
                WRITE_OUT,
            ],
        )
    finally:
        if body_fd is not None:
            os.close(body_fd)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError):
        raise SystemExit(125) from None
