"""Owned launch gate. No sandbox, no network or GPU memory isolation claims."""

import json
import os
import resource
import sys


def main():
    with open(sys.argv[1]) as stream:
        spec = json.load(stream)
    gate = int(sys.argv[2])
    if os.read(gate, 1) != b"1":
        return 125
    os.close(gate)
    os.umask(0o077)
    os.sched_setaffinity(0, spec["cpus"])
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (spec["file_bytes"], spec["file_bytes"]))
    os.chdir(spec["cwd"])
    os.execve(spec["argv"][0], spec["argv"], spec["environment"])


if __name__ == "__main__":
    sys.exit(main())
