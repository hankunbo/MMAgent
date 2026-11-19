"""11A's own direct-exit observer. Original lease retains sole cleanup authority."""

import json
import sys
from pathlib import Path

import pytest

from db import manage
from tests.database.pytest_plugin import _LEASE


def write(root, name, value):
    (root / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def process_ticks(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


class Observer:
    def __init__(self, out):
        self.out, self.reports, self.clusters = out, [], []
        self.lease = None

    def pytest_sessionstart(self, session):
        self.lease = session.config.stash[_LEASE]
        self.temp_root = Path(session.config.option.basetemp).parent
        original = self.lease.start

        def start(port):
            cluster = original(port)
            state = manage._read_test_state()
            pid = int((self.lease.root / "data" / "postmaster.pid").read_text().splitlines()[0])
            self.clusters.append(
                dict(
                    cluster_id=cluster,
                    root=str(self.lease.root),
                    pid=pid,
                    start_ticks=process_ticks(pid),
                    state_cluster_id=state["cluster_id"],
                )
            )
            write(self.out, "clusters.json", self.clusters)
            return cluster

        self.lease.start = start
        session.config.add_cleanup(lambda: setattr(self.lease, "start", original))

    def pytest_runtest_logreport(self, report):
        self.reports.append(
            dict(
                nodeid=report.nodeid,
                when=report.when,
                outcome=report.outcome,
                duration=report.duration,
            )
        )
        write(self.out, "test-reports.json", self.reports)

    def finish(self, status):
        checks = [
            dict(
                **c,
                root_removed=not Path(c["root"]).exists(),
                process_stopped=process_ticks(c["pid"]) != c["start_ticks"],
            )
            for c in self.clusters
        ]
        clean = all(c["root_removed"] and c["process_stopped"] for c in checks)
        clean = clean and (self.lease is not None and not self.lease.root.exists())
        temp_removed = not self.temp_root.exists()
        clean = clean and temp_removed
        write(
            self.out,
            "cleanup.json",
            dict(
                verified=clean,
                clusters=checks,
                root=str(self.lease.root) if self.lease else None,
                temp_root=str(self.temp_root),
                temp_removed=temp_removed,
            ),
        )
        write(self.out, "pytest-return.json", dict(direct_exit=status))
        return 0 if clean else 2


def main():
    out = Path(sys.argv[1]).resolve()
    if not out.is_dir() or not (out / "collector-owner.json").is_file():
        raise SystemExit("11A collector-owned output required")
    observer = Observer(out)
    status = int(pytest.main(sys.argv[2:], plugins=[observer]))
    integrity = observer.finish(status)
    (out / "pytest.exit").write_text(str(status) + "\n")
    (out / "cleanup.exit").write_text(str(integrity) + "\n")
    return status or integrity


if __name__ == "__main__":
    sys.exit(main())
